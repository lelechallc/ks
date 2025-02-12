import os
import argparse

import kai.tensorflow as kai
import tensorflow.compat.v1 as tf

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'predict'], default='train')
parser.add_argument('--dryrun', dest="dryrun", const=True, default=False, nargs='?')
parser.add_argument('--with_kai', action="store_true")
args = parser.parse_args()


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

# Autodis Emb
def autodis_embedding(name, meta_embedding, numerical_inputs, alpha, tau):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # [-1, 16] [-1, 16, 100, 8]
        dim = 5
        numerical_inputs = tf.reshape(numerical_inputs, [-1, 16, 1])
        h = tf.layers.dense(numerical_inputs, dim, activation=tf.nn.leaky_relu, use_bias=False,
                            kernel_initializer=tf.glorot_uniform_initializer())
        # [-1, 16, 200]

        p_h = tf.layers.dense(h, dim, activation=None, use_bias=False,
                              kernel_initializer=tf.glorot_uniform_initializer()) + alpha * h

        p_h = p_h / tau

        p_h = tf.expand_dims(tf.nn.softmax(p_h, axis=-1), -1)
        # [-1, 16, 100, 1]

        output = tf.reduce_sum(p_h * meta_embedding, -2, keep_dims=False)
        # [-1, 16, 8]

        output = tf.reshape(output, [-1, 64])

        return output

# 1. define sparse input
if args.mode == 'train':
    import tensorflow.compat.v1 as tf
    import kai.tensorflow as kai

    user_embedding = kai.nn.new_embedding("user_embedding", dim=4, slots=[101, 102])
    comment_id_embedding = kai.nn.new_embedding("c_id_embedding", dim=64, slots=[201, 202])
    comment_info_embedding = kai.nn.new_embedding("c_info_embedding", dim=32, slots=[203, 204, 205, 206, 207, 209])
    position_embedding = kai.nn.new_embedding("position_embedding", dim=8, slots=[208])
    # 测试是Train走的if阶段，test也是走的if阶段
    print("############## if阶段 ##############")
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
    comment_id_embedding = config.new_embedding("c_id_embedding", dim=64, slots=[201, 202])
    comment_info_embedding = config.new_embedding("c_info_embedding", dim=32, slots=[203, 204, 205, 206, 207, 209])
    position_embedding = config.new_embedding("position_embedding", dim=8, slots=[208])
    print("############## else阶段 ##############")

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
    
# define model structure
field_input = tf.concat([user_embedding, comment_id_embedding, comment_info_embedding, position_embedding], -1)
expand_xtr = simple_dense_network("expand_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
like_xtr = simple_dense_network("like_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
reply_xtr = simple_dense_network("reply_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)

if args.mode == 'train':
    # define label input and define metrics
    expand_label = kai.nn.get_dense_fea("expandaction", dim=1, dtype=tf.float32)
    like_label = kai.nn.get_dense_fea("likeaction", dim=1, dtype=tf.float32)
    reply_label = kai.nn.get_dense_fea("replyaction", dim=1, dtype=tf.float32)

    # expand_label = tf.cast(kai.nn.get_dense_fea("expandAction", dim=1, dtype=tf.int64), tf.float32)
    # like_label = tf.cast(kai.nn.get_dense_fea("likeAction", dim=1, dtype=tf.int64), tf.float32)
    # reply_label = tf.cast(kai.nn.get_dense_fea("replyAction", dim=1, dtype=tf.int64), tf.float32)

    sample_weight = kai.nn.get_dense_fea("sample_weight", dim=1, dtype=tf.float32)

    # 按照photo hash方式进行评估
    # sample_weight = tf.ones_like(expand_label, dtype=tf.float32)
    # ones = tf.ones_like(expand_label, dtype=tf.float32)
    # photo_hash = kai.nn.get_dense_fea("photo_hash", dim=1, dtype=tf.int64)

    # 七元组photo_hash
    # targets = [
    #     ('expand_predict', expand_xtr, expand_label, sample_weight, "auc", -ones, photo_hash),
    #     ('like_predict', like_xtr, like_label, sample_weight, "auc", -ones, photo_hash),
    #     ('reply_predict', reply_xtr, reply_label, sample_weight, "auc", -ones, photo_hash)
    # ]

    # 在send_to_mio_learner的user_hash_attr = "photo_hash"，来在photo hash维度评估
    targets = [
        ('expand_predict', expand_xtr, expand_label, sample_weight, "auc"),
        ('like_predict', like_xtr, like_label, sample_weight, "auc"),
        ('reply_predict', reply_xtr, reply_label, sample_weight, "auc")
    ]

    # metric_name, preds, labels, weights, metric_type, _, _ = zip(*targets)
    metric_name, preds, labels, weights, metric_type = zip(*targets)


    debug_tensor = {
        "expand_xtr": tf.slice(expand_xtr, [0, 0], [2, -1]),
        "like_xtr": tf.slice(like_xtr, [0, 0], [2, -1]),
        "reply_xtr": tf.slice(reply_xtr, [0, 0], [2, -1]),

        "expand_label": tf.slice(expand_label, [0, 0], [2, -1]),
        "like_label": tf.slice(like_label, [0, 0], [2, -1]),
        "reply_label": tf.slice(reply_label, [0, 0], [2, -1]),

        "sample_weight": tf.slice(sample_weight, [0, 0], [2, -1]),
    }

    kai.add_run_hook(TensorPrintHook(debug_tensor), "debug_tensor_hook")


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

