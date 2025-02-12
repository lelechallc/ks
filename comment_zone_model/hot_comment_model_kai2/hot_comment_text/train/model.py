import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'predict'], default='train')
parser.add_argument('--dryrun', dest="dryrun", const=True, default=False, nargs='?')
parser.add_argument('--with_kai', action="store_true")
args = parser.parse_args()

SEQ_SIZE = 8

# 1. define sparse input
if args.mode == 'train':
    import tensorflow.compat.v1 as tf
    import kai.tensorflow as kai

    user_embedding = kai.nn.new_embedding("user_embedding", dim=4, slots=[101, 102])
    photo_embedding = kai.nn.new_embedding("photo_embedding", dim=64, slots=[103, 104])
    comment_id_embedding = tf.reshape(kai.nn.new_embedding("c_id_embedding", dim=64, expand=SEQ_SIZE, slots=[201, 202]), [-1, SEQ_SIZE, 64 * 2])
    comment_info_embedding = tf.reshape(kai.nn.new_embedding("c_info_embedding", dim=32, expand=SEQ_SIZE, slots=[203, 204, 205, 206, 207, 208]), [-1, SEQ_SIZE, 32 * 6])
    padding_input = kai.nn.get_dense_fea("mask_pack", dim=SEQ_SIZE, dtype=tf.float32)  # [-1, seq]
else:
    import tensorflow as tf
    from mio_tensorflow.config import MioConfig
    if not args.dryrun and not args.with_kai:
        # monkey patch
        import mio_tensorflow.patch as mio_tensorflow_patch
        mio_tensorflow_patch.apply()

    base_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), './base.yaml')
    config = MioConfig.from_base_yaml(base_config, clear_embeddings=True, clear_params=True,
                                    dryrun=args.dryrun, label_with_kv=True, grad_no_scale=False,
                                    with_kai=args.with_kai)
    compress_kwargs = dict(compress_group="USER")

    user_embedding = config.new_embedding("user_embedding", dim=4, slots=[101, 102], **compress_kwargs)
    photo_embedding = config.new_embedding("photo_embedding", dim=64, slots=[103, 104], **compress_kwargs)
    comment_id_embedding = config.new_embedding("c_id_embedding", dim=64, slots=[201, 202])   # [-1, dim]
    comment_info_embedding = config.new_embedding("c_info_embedding", dim=32, slots=[203, 204, 205, 206, 207, 208])
    padding_input = tf.reshape(config.get_extra_param(name="mask_pack", size=1), [1, -1])  # [1, n]

def simple_dense_network(name, inputs, units, dropout=0, act=tf.nn.tanh, last_act=tf.nn.sigmoid, stop_gradient=False):
    if stop_gradient:
        output = tf.stop_gradient(inputs)
    else:
        output = inputs
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if dropout > 0:
            output = tf.layers.dropout(output, dropout, training=(args.mode == 'train'))
        for i, unit in enumerate(units):
            # output = tf.layers.Dense(unit, act, name='dense_{}_{}'.format(name, i))(output)
            if i == len(units) - 1:
                act = last_act
            output = tf.layers.dense(output, unit, activation=act,
                                  kernel_initializer=tf.glorot_uniform_initializer())
        return output
    

def norm(name, x, eps=1e-5):
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    dim = x.shape.as_list()[-1]
    mean, var = tf.nn.moments(x, axes=[-1], keep_dims=True)
    x = (x - mean) * tf.math.rsqrt(var + eps)
    gamma = tf.get_variable(f'gamma_{name}', shape=[dim], initializer=tf.initializers.ones())
    beta = tf.get_variable(f'beta_{name}', shape=[dim], initializer=tf.initializers.ones())
    output = gamma * x + beta
    return output
  
  
def multi_head_attention(name, queries, keys, padding, num_units=None, num_heads=8):
    """
    queries: [-1, q_seq, dim], q_seq = 1 or k_seq
    keys: [-1, k_seq, dim]
    padding 是由 0 / 1 组成的 [-1, k_seq] 的矩阵
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]
        q_seq = tf.shape(queries)[1]
        k_seq = tf.shape(keys)[1]
        
        Q = tf.layers.dense(queries, num_units) # [-1, q_seq, dim]
        K = tf.layers.dense(keys, num_units)  # [-1, k_seq, dim]
        V = tf.layers.dense(keys, num_units)  # [-1, k_seq, dim]

        assert num_units % num_heads == 0
        depth = num_units // num_heads
        Q_ = tf.transpose(tf.reshape(Q, [-1, q_seq, num_heads, depth]), perm=[0, 2, 1, 3])
        K_ = tf.transpose(tf.reshape(K, [-1, k_seq, num_heads, depth]), perm=[0, 2, 1, 3])
        V_ = tf.transpose(tf.reshape(V, [-1, k_seq, num_heads, depth]), perm=[0, 2, 1, 3])   # [-1, num_heads, k_seq, depth]

        # [-1, num_heads, k_seq, k_seq]
        outputs = tf.matmul(Q_, K_, transpose_b=True) / tf.math.sqrt(tf.cast(depth, tf.float32))
        
        # padding mask
        # [-1, 1, k_seq, 1] -> [-1, num_heads, k_seq, k_seq]
        padding = (1 - tf.tile(padding[:, None, :, None], [1, num_heads, 1, k_seq])) * -1e9
        outputs = outputs + padding

        outputs = tf.nn.softmax(outputs, axis=-1)
        attention = tf.transpose(tf.matmul(outputs, V_), perm=[0, 2, 1, 3])  # [-1, k_seq, num_heads, depth]
        attention = tf.reshape(attention, [-1, k_seq, num_units])  # [-1, k_seq, num_units]

        # add and norm
        attention = norm(name=f"{name}_norm", x=attention + K)
        return attention
    

# define model structure
if args.mode == 'train':
    queries = tf.expand_dims(tf.concat([user_embedding, photo_embedding], axis=-1), 1)   # [-1, 1, dim]
    keys = tf.concat([comment_id_embedding, comment_info_embedding], axis=-1)  # [-1, seq, dim]
else:
    queries = tf.expand_dims(tf.slice(tf.concat([user_embedding, photo_embedding], axis=-1), [0, 0], [1, -1]), 0)   # [1, 1, dim]
    keys = tf.expand_dims(tf.concat([comment_id_embedding, comment_info_embedding], axis=-1), 0)  # [1, seq, dim]

attention_output = multi_head_attention("multi_head_attention", queries, keys, padding_input, num_units=128)

expand_xtr = tf.reshape(simple_dense_network("expand_xtr", attention_output, [1], 0.0, act=tf.nn.sigmoid, last_act=tf.nn.sigmoid), [-1, 1])   #[-1, 1]
like_xtr = tf.reshape(simple_dense_network("like_xtr", attention_output, [1], 0.0, act=tf.nn.sigmoid, last_act=tf.nn.sigmoid), [-1, 1])
reply_xtr = tf.reshape(simple_dense_network("reply_xtr", attention_output, [1], 0.0, act=tf.nn.sigmoid, last_act=tf.nn.sigmoid), [-1, 1])

if args.mode == 'train':
    # define label input and define metrics
    expand_label = tf.reshape(kai.nn.get_dense_fea("expandAction_v_pack", dim=SEQ_SIZE, dtype=tf.float32), [-1, 1])
    like_label = tf.reshape(kai.nn.get_dense_fea("likeAction_v_pack", dim=SEQ_SIZE, dtype=tf.float32), [-1, 1])
    reply_label = tf.reshape(kai.nn.get_dense_fea("replyAction_v_pack", dim=SEQ_SIZE, dtype=tf.float32), [-1, 1])
    sample_weight = tf.reshape(kai.nn.get_dense_fea("sample_weight_pack", dim=SEQ_SIZE, dtype=tf.float32), [-1, 1])
    targets = [
        ('expand_predict', expand_xtr, expand_label, sample_weight, "auc"),
        ('like_predict', like_xtr, like_label, sample_weight, "auc"),
        ('reply_predict', reply_xtr, reply_label, sample_weight, "auc")
    ]

    metric_name, preds, labels, weights, metric_type = zip(*targets)


    # 5. define optimizer
    loss = tf.losses.log_loss(labels, preds, weights, reduction="weighted_sum")
    optimizer = kai.nn.optimizer.Adam(1e-3)
    optimizer.minimize(loss)

    # 6. finish define model structure 
    kai.build_model(optimizer=[optimizer], metrics=targets)
else:
    targets = [
      ("expand_xtr", expand_xtr),
      ("like_xtr", like_xtr),
      ("reply_xtr", reply_xtr),
    ]
    q_names, preds = zip(*targets)
    config.dump_predict_config('./predict/config', targets, input_type=3, extra_preds=q_names)

