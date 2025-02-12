import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'predict'], default='train')
parser.add_argument('--dryrun', dest="dryrun", const=True, default=False, nargs='?')
parser.add_argument('--with_kai', action="store_true")
args = parser.parse_args()

SEQ_SIZE = 20
eps = 1e-6

# 1. define sparse input
if args.mode == 'train':
    import tensorflow.compat.v1 as tf
    import kai.tensorflow as kai

    user_embedding = kai.nn.new_embedding("user_embedding", dim=4, slots=[101, 102])
    photo_embedding = kai.nn.new_embedding("photo_embedding", dim=64, slots=[103, 104])
    
    comment_id_list = []
    comment_info_list = []
    comment_content_list = []
    comment_rank_pxtr_list = []
    for slot in [201, 202]:
        comment_id_list.append(tf.reshape(kai.nn.new_embedding(f"comment_id_{slot}", dim=64, expand=SEQ_SIZE, slots=[slot]), [-1, SEQ_SIZE, 64]))

    for slot in [203, 204, 205, 206, 207, 208]:
        comment_info_list.append(tf.reshape(kai.nn.new_embedding(f"comment_info_{slot}", dim=8, expand=SEQ_SIZE, slots=[slot]), [-1, SEQ_SIZE, 8]))

    for slot in [209, 210, 211, 212, 213, 214]:
        comment_content_list.append(tf.reshape(kai.nn.new_embedding(f"comment_content_{slot}", dim=4, expand=SEQ_SIZE, slots=[slot]), [-1, SEQ_SIZE, 4]))

    for slot in [241, 242, 243, 244, 245, 246, 247]:
        comment_rank_pxtr_list.append(tf.reshape(kai.nn.new_embedding(f"comment_rank_pxtr_{slot}", dim=8, expand=SEQ_SIZE, slots=[slot]), [-1, SEQ_SIZE, 8]))

    comment_id_embedding = tf.concat(comment_id_list, axis=-1)
    comment_info_embedding = tf.concat(comment_info_list, axis=-1)
    comment_content_embedding = tf.concat(comment_content_list, axis=-1)
    comment_rank_pxtr_embedding = tf.concat(comment_rank_pxtr_list, axis=-1)

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
    comment_info_embedding = config.new_embedding("c_info_embedding", dim=8, slots=[203, 204, 205, 206, 207, 208])
    comment_content_embedding = config.new_embedding("c_content_embedding", dim=4, slots=[209, 210, 211, 212, 213, 214])
    comment_rank_pxtr_embedding = config.new_embedding("c_rank_pxtr_embedding", dim=8, slots=[241, 242, 243, 244, 245, 246, 247])
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

def simple_dense_network_poso(name, poso_input, inputs, units, dropout=0, act=tf.nn.tanh, last_act=tf.nn.sigmoid, stop_gradient=False):
    if stop_gradient:
        output = tf.stop_gradient(inputs)
    else:
        output = inputs
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if dropout > 0:
            output = tf.layers.dropout(output, dropout, training=(args.mode == 'train'))
        for i, unit in enumerate(units):
            if i == len(units) - 1:
                act = last_act
                output = tf.layers.dense(output, unit, activation=act,
                                    kernel_initializer=tf.glorot_uniform_initializer())
            else:
                output = tf.layers.dense(output, unit, activation=act,
                                    kernel_initializer=tf.glorot_uniform_initializer())
                poso_output = tf.layers.dense(poso_input, unit, activation=act,
                                    kernel_initializer=tf.glorot_uniform_initializer())
                output = output * poso_output
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
  
  
def multi_head_attention(name, queries, keys, padding, num_units=None, num_heads=8, add_embedding=None):
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
        # attention = norm(name=f"{name}_norm", x=attention + K)
        if add_embedding is not None:
            attention = norm(name=f"{name}_norm", x=attention + tf.layers.dense(add_embedding, num_units))
        else:
            attention = norm(name=f"{name}_norm", x=attention + K)
        return attention
    

# define model structure
if args.mode == 'train':
    attention_input = tf.concat([comment_id_embedding, comment_info_embedding, comment_content_embedding, comment_rank_pxtr_embedding], axis=-1)  # [-1, seq, dim]
    user_input = tf.tile(tf.expand_dims(tf.concat([user_embedding, photo_embedding], axis=-1), 1), [1, SEQ_SIZE, 1])   # [-1, seq, dim]
else:
    attention_input = tf.expand_dims(tf.concat([comment_id_embedding, comment_info_embedding, comment_content_embedding, comment_rank_pxtr_embedding], axis=-1), 0)  # [1, seq, dim]
    user_input = tf.expand_dims(tf.concat([user_embedding, photo_embedding], axis=-1), 0)  # [1, seq, dim]
    comment_content_embedding = tf.expand_dims(comment_content_embedding, 0)


attention_output = multi_head_attention("multi_head_attention", attention_input, attention_input, padding_input, num_units=128)   # [-1, seq, dim]
# attention_output = attention_input
field_input = tf.concat([user_input, attention_output], axis=-1)

expand_seq_xtr = simple_dense_network("expand_xtr", field_input, [64, 32, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)   #[-1, seq, 1]
like_seq_xtr = simple_dense_network("like_xtr", field_input, [64, 32, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
reply_seq_xtr = simple_dense_network("reply_xtr", field_input, [64, 32, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
copy_seq_xtr = simple_dense_network("copy_xtr", field_input, [64, 32, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)   #[-1, seq, 1]
share_seq_xtr = simple_dense_network("share_xtr", field_input, [64, 32, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
audience_seq_xtr = simple_dense_network("audience_xtr", field_input, [64, 32, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
continuous_seq_xtr = simple_dense_network("continuous_xtr", field_input, [64, 32, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
staytime_seq_xtr = simple_dense_network("staytime_xtr", field_input, [64, 32, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)

expand_xtr = tf.squeeze(expand_seq_xtr, [-1])   # [-1, seq]
like_xtr = tf.squeeze(like_seq_xtr, [-1])
reply_xtr = tf.squeeze(reply_seq_xtr, [-1])
copy_xtr = tf.squeeze(copy_seq_xtr, [-1])   # [-1, seq]
share_xtr = tf.squeeze(share_seq_xtr, [-1])
audience_xtr = tf.squeeze(audience_seq_xtr, [-1])
continuous_xtr = tf.squeeze(continuous_seq_xtr, [-1])
staytime_xtr = tf.squeeze(staytime_seq_xtr, [-1]) 

expand_xtr = tf.reshape(expand_xtr, [-1, 1])
like_xtr = tf.reshape(like_xtr, [-1, 1])
reply_xtr = tf.reshape(reply_xtr, [-1, 1])
copy_xtr = tf.reshape(copy_xtr, [-1, 1])
share_xtr = tf.reshape(share_xtr, [-1, 1])
audience_xtr = tf.reshape(audience_xtr, [-1, 1])
continuous_xtr = tf.reshape(continuous_xtr, [-1, 1])
staytime_xtr = tf.reshape(staytime_xtr, [-1, 1])
staytime_xtr = tf.clip_by_value(staytime_xtr,  eps, 1-eps)

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

    expand_label = tf.reshape(kai.nn.get_dense_fea("expand_v_pack", dim=SEQ_SIZE, dtype=tf.float32), [-1, 1])   # [-1, 1]
    like_label = tf.reshape(kai.nn.get_dense_fea("like_v_pack", dim=SEQ_SIZE, dtype=tf.float32), [-1, 1])
    reply_label = tf.reshape(kai.nn.get_dense_fea("reply_v_pack", dim=SEQ_SIZE, dtype=tf.float32), [-1, 1])
    copy_label = tf.reshape(kai.nn.get_dense_fea("copy_v_pack", dim=SEQ_SIZE, dtype=tf.float32), [-1, 1])   # [-1, 1]
    share_label = tf.reshape(kai.nn.get_dense_fea("share_v_pack", dim=SEQ_SIZE, dtype=tf.float32), [-1, 1])
    audience_label = tf.reshape(kai.nn.get_dense_fea("audience_v_pack", dim=SEQ_SIZE, dtype=tf.float32), [-1, 1])
    continuous_expand_label = tf.reshape(kai.nn.get_dense_fea("continuous_v_pack", dim=SEQ_SIZE, dtype=tf.float32), [-1, 1])
    staytime_label = tf.reshape(kai.nn.get_dense_fea("staytime_v_pack", dim=SEQ_SIZE, dtype=tf.float32), [-1, 1])  

    staytime_label = tf.clip_by_value(staytime_label / 1000, 0, 20)

    show_label = tf.reshape(kai.nn.get_dense_fea("show_v_pack", dim=SEQ_SIZE, dtype=tf.float32), [-1, 1])   # [-1, 1]
    ones = tf.ones_like(show_label, dtype=tf.float32)
    zeros = tf.zeros_like(show_label, dtype=tf.float32)
    train_weight = tf.where(show_label > 0, ones, zeros)

    targets = [
        ('expand_predict', expand_xtr, expand_label, train_weight, "auc"),
        ('like_predict', like_xtr, like_label, train_weight, "auc"),
        ('reply_predict', reply_xtr, reply_label, train_weight, "auc"),
        ('copy_predict', copy_xtr, copy_label, train_weight, "auc"),
        ('share_predict', share_xtr, share_label, train_weight, "auc"),
        ('audience_predict', audience_xtr, audience_label, train_weight, "auc"),
        ('continuous_predict', continuous_xtr, continuous_expand_label, train_weight, "auc"),
    ]


    metric_name, preds, labels, weights, metric_type = zip(*targets)


    # 5. define optimizer
    pxtr_loss = tf.losses.log_loss(labels, preds, weights, reduction="weighted_sum")
    time_pos_loss = tf.losses.log_loss(ones, staytime_xtr, staytime_label, reduction="weighted_sum")
    time_neg_loss = tf.losses.log_loss(zeros, staytime_xtr, train_weight, reduction="weighted_sum")
    tf.summary.scalar("pxtr_loss", pxtr_loss) 
    tf.summary.scalar("time_loss", time_pos_loss + time_neg_loss) 

    loss = pxtr_loss + time_pos_loss + time_neg_loss

    debug_tensor = {
        "padding_input": tf.slice(padding_input, [0, 0], [2, -1]),
        "expand_xtr": tf.slice(expand_xtr, [0, 0], [2, -1]),
        "expand_label": tf.slice(expand_label, [0, 0], [2, -1]),
        "sample_weight": tf.slice(show_label, [0, 0], [2, -1]),
        "attention_input_embedding": tf.squeeze(tf.slice(attention_input, [0, 0, 0], [1, -1, 4])),
        # "expand_positive_xtr": tf.slice(tf.boolean_mask(expand_xtr, tf.reduce_sum(expand_label, axis=-1) > 0), [0, 0], [2, -1]),
        # "expand_positive_label": tf.slice(tf.boolean_mask(expand_label, tf.reduce_sum(expand_label, axis=-1) > 0), [0, 0], [2, -1]),
    }
    kai.add_run_hook(TensorPrintHook(debug_tensor), "debug_tensor_hook")

    optimizer = kai.nn.optimizer.Adam(1e-3)
    optimizer.minimize(loss)

    # 6. finish define model structure 
    kai.build_model(optimizer=[optimizer], metrics=targets)
else:
    targets = [
      ("expand_xtr", expand_xtr),
      ("like_xtr", like_xtr),
      ("reply_xtr", reply_xtr),
      ("copy_xtr", copy_xtr),
      ("share_xtr", share_xtr),
      ("audience_xtr", audience_xtr),
      ("continuous_xtr", continuous_xtr),
      ("staytime_xtr", staytime_xtr / (1 - staytime_xtr)),
    ]
    q_names, preds = zip(*targets)
    config.dump_predict_config('./predict/config', targets, input_type=3, extra_preds=q_names)
