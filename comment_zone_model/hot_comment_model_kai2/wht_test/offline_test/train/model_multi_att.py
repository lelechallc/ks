import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'predict'], default='train')
parser.add_argument('--dryrun', dest="dryrun", const=True, default=False, nargs='?')
parser.add_argument('--with_kai', action="store_true")
args = parser.parse_args()

SEQ_SIZE = 16

# 1. define sparse input
if args.mode == 'train':
    import tensorflow.compat.v1 as tf
    import kai.tensorflow as kai

    user_embedding = kai.nn.new_embedding("user_embedding", dim=4, slots=[101, 102])
    photo_embedding = kai.nn.new_embedding("photo_embedding", dim=64, slots=[103, 104])
    
    comment_id_list = []
    comment_info_list = []
    comment_content_list = []
    for slot in [201, 202]:
        comment_id_list.append(tf.reshape(kai.nn.new_embedding(f"comment_id_{slot}", dim=64, expand=SEQ_SIZE, slots=[slot]), [-1, SEQ_SIZE, 64]))

    for slot in [203, 204, 205, 206, 207, 208]:
        comment_info_list.append(tf.reshape(kai.nn.new_embedding(f"comment_info_{slot}", dim=8, expand=SEQ_SIZE, slots=[slot]), [-1, SEQ_SIZE, 8]))

    for slot in [209, 210, 211, 212, 213, 214]:
        comment_content_list.append(tf.reshape(kai.nn.new_embedding(f"comment_content_{slot}", dim=4, expand=SEQ_SIZE, slots=[slot]), [-1, SEQ_SIZE, 4]))
    
    comment_id_embedding = tf.concat(comment_id_list, axis=-1)  # [-1, seq, 64*2]
    comment_info_embedding = tf.concat(comment_info_list, axis=-1)  # [-1, seq, 8*6]
    comment_content_embedding = tf.concat(comment_content_list, axis=-1)    # [-1, seq, 4*6]
    
    padding_input = kai.nn.get_dense_fea("mask_pack", dim=SEQ_SIZE, dtype=tf.float32)  # [-1, seq]  (1,1,...,0,0)组成的序列
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
    comment_info_embedding = config.new_embedding("c_info_embedding", dim=8, slots=[203, 204, 205, 206, 207, 208])
    comment_content_embedding = config.new_embedding("c_content_embedding", dim=4, slots=[209, 210, 211, 212, 213, 214])
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
    

def layer_norm(name, x, eps=1e-5):
    """layer normalization

    Args:
        name (str): _description_
        x (Tensor): [-1, k_seq, num_units]
        eps (float, optional): Defaults to 1e-5.

    Returns:
        output (Tensor): same shape as x
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        dim = x.shape.as_list()[-1]
        mean, var = tf.nn.moments(x, axes=[-1], keep_dims=True)
        x = (x - mean) * tf.math.rsqrt(var + eps)
        gamma = tf.get_variable(f'gamma_{name}', shape=[dim], initializer=tf.initializers.ones())
        beta = tf.get_variable(f'beta_{name}', shape=[dim], initializer=tf.initializers.ones())
        output = gamma * x + beta
        return output


def multi_head_attention(name, queries, keys, padding_mask, num_units=None, num_heads=8, add_embedding=None):
    """ 
    queries: [-1, q_seq, h] q_seq=k_seq
    keys:    [-1, k_seq, h]
    values:  [-1, k_seq, h]
    padding_mask: [-1, k_seq]. 是由 0 / 1 组成的 mask 的矩阵, padding位值为0, 非padding位值为1
    num_units: int. 等于 num_heads * depth
    num_heads: int
    add_embedding: Tensor, 残差连接的input_tensor
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]
        q_seq = tf.shape(queries)[1]
        k_seq = tf.shape(keys)[1]
        
        # 可学习参数1
        Q = tf.layers.dense(queries, num_units) # [-1, q_seq, dim]
        K = tf.layers.dense(keys, num_units)    # [-1, k_seq, dim]
        V = tf.layers.dense(keys, num_units)    # [-1, k_seq, dim]

        assert num_units % num_heads == 0
        depth = num_units // num_heads
        Q_ = tf.transpose(tf.reshape(Q, [-1, q_seq, num_heads, depth]), perm=[0, 2, 1, 3])   # [-1, num_heads, q_seq, depth]
        K_ = tf.transpose(tf.reshape(K, [-1, k_seq, num_heads, depth]), perm=[0, 2, 1, 3])   # [-1, num_heads, k_seq, depth]
        V_ = tf.transpose(tf.reshape(V, [-1, k_seq, num_heads, depth]), perm=[0, 2, 1, 3])   # [-1, num_heads, k_seq, depth]

        # [-1, num_heads, k_seq, k_seq]
        outputs = tf.matmul(Q_, K_, transpose_b=True) / tf.math.sqrt(tf.cast(depth, tf.float32))    # [-1, num_heads, q_seq, k_seq]
        
        # padding mask
        # [-1, k_seq] -> [-1, 1, k_seq, 1] -> [-1, num_heads, k_seq, k_seq]
        padding_mask = (1 - tf.tile(padding_mask[:, None, :, None], [1, num_heads, 1, k_seq])) * -1e9 # [-1, num_heads, k_seq, k_seq] 原值1转换为0，原值0转换为-1e9
        outputs = outputs + padding_mask   

        outputs = tf.nn.softmax(outputs, axis=-1)   # [-1, num_heads, k_seq, k_seq]
        attention = tf.transpose(tf.matmul(outputs, V_), perm=[0, 2, 1, 3])  # [-1, k_seq, num_heads, depth]
        attention = tf.reshape(attention, [-1, k_seq, num_units])  # [-1, k_seq, num_units]

        # attention = tf.layers.dense(attention, attention.shape[-1]) # [-1, k_seq, num_units]

        # add and norm
        # 可学习参数2
        if add_embedding is not None:
            attention = layer_norm(name=f"{name}_norm", x=attention + tf.layers.dense(add_embedding, num_units))
        else:
            attention = layer_norm(name=f"{name}_norm", x=attention + K)
        return attention
    

# define model structure
if args.mode == 'train':
    attention_input = tf.concat([comment_id_embedding, comment_info_embedding, comment_content_embedding], axis=-1)  # [-1, seq, dim]
    user_input = tf.tile(tf.expand_dims(tf.concat([user_embedding, photo_embedding], axis=-1), 1), [1, SEQ_SIZE, 1])   # [-1, seq, dim]
else:
    attention_input = tf.expand_dims(tf.concat([comment_id_embedding, comment_info_embedding, comment_content_embedding], axis=-1), 0)  # [1, seq, dim]
    user_input = tf.expand_dims(tf.concat([user_embedding, photo_embedding], axis=-1), 0)  # [1, seq, dim]


attention_output = multi_head_attention("multi_head_attention", attention_input, attention_input, padding_input, num_units=128)   # [-1, seq, dim]
# attention_output = attention_input
field_input = tf.concat([user_input, attention_output], axis=-1)

expand_seq_xtr = simple_dense_network("expand_xtr", field_input, [64, 32, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)   #[-1, seq, 1]
like_seq_xtr = simple_dense_network("like_xtr", field_input, [64, 32, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
reply_seq_xtr = simple_dense_network("reply_xtr", field_input, [64, 32, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)

expand_xtr = tf.squeeze(expand_seq_xtr, [-1])   # [-1, seq]
like_xtr = tf.squeeze(like_seq_xtr, [-1])
reply_xtr = tf.squeeze(reply_seq_xtr, [-1])


if args.mode == 'train':

    class TensorPrintHook(kai.training.RunHookBase):
        def __init__(self, debug_tensor_map):
            self.has_print = False
            self.debug_tensor_map = debug_tensor_map

        def before_pass_run(self, pass_run_context):
            """
            每个 pass 只会 print 一次
            """
            self.has_print = False
        
        def before_step_run(self, step_run_context):
            return kai.training.StepRunArgs(fetches=self.debug_tensor_map)
        
        def after_step_run(self, step_run_context, step_run_values):
            if not self.has_print:
                for k, v in step_run_values.result.items():
                    print(f"{k} = {v}")
                self.has_print = True

    # define label input and define metrics
    expand_label = kai.nn.get_dense_fea("expand_v_pack", dim=SEQ_SIZE, dtype=tf.float32)   # [-1, seq]
    like_label = kai.nn.get_dense_fea("like_v_pack", dim=SEQ_SIZE, dtype=tf.float32)
    reply_label = kai.nn.get_dense_fea("reply_v_pack", dim=SEQ_SIZE, dtype=tf.float32)
    sample_weight = kai.nn.get_dense_fea("sample_weight_pack", dim=SEQ_SIZE, dtype=tf.float32)
    targets = [
        ('expand_predict', expand_xtr, expand_label, sample_weight, "auc"),
        ('like_predict', like_xtr, like_label, sample_weight, "auc"),
        ('reply_predict', reply_xtr, reply_label, sample_weight, "auc")
    ]

    metric_name, preds, labels, weights, metric_type = zip(*targets)


    # 5. define optimizer
    loss = tf.losses.log_loss(labels, preds, weights, reduction="weighted_sum")

    debug_tensor = {
        "padding_input": tf.slice(padding_input, [0, 0], [2, -1]),
        "expand_xtr": tf.slice(expand_xtr, [0, 0], [2, -1]),
        "expand_label": tf.slice(expand_label, [0, 0], [2, -1]),
        "sample_weight": tf.slice(sample_weight, [0, 0], [2, -1]),
        "attention_input_embedding": tf.squeeze(tf.slice(attention_input, [0, 0, 0], [1, -1, 4])),
        "expand_positive_xtr": tf.slice(tf.boolean_mask(expand_xtr, tf.reduce_sum(expand_label, axis=-1) > 0), [0, 0], [2, -1]),
        "expand_positive_label": tf.slice(tf.boolean_mask(expand_label, tf.reduce_sum(expand_label, axis=-1) > 0), [0, 0], [2, -1]),
    }
    kai.add_run_hook(TensorPrintHook(debug_tensor), "debug_tensor_hook")

    optimizer = kai.nn.optimizer.Adam(1e-3)
    optimizer.minimize(loss)

    # 6. finish define model structure 
    kai.build_model(optimizer=[optimizer], metrics=targets)
else:
    targets = [
      ("expand_xtr", tf.reshape(expand_xtr, [-1, 1])),
      ("like_xtr", tf.reshape(like_xtr, [-1, 1])),
      ("reply_xtr", tf.reshape(reply_xtr, [-1, 1])),
    ]
    q_names, preds = zip(*targets)
    config.dump_predict_config('./predict/config', targets, input_type=3, extra_preds=q_names)

