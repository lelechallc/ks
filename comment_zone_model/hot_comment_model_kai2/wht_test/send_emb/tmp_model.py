"""在个性化模型udp model基础上增加copy、share、audience、连续展开目标; 增加了photo_author_id特征；去掉了时序区样本（因为还没有这三个目标）；
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'predict'], default='train')
parser.add_argument('--dryrun', dest="dryrun", const=True, default=False, nargs='?')
parser.add_argument('--with_kai', action="store_true")
args = parser.parse_args()

# 1. define sparse input
if args.mode == 'train':
    import tensorflow.compat.v1 as tf
    import kai.tensorflow as kai

    # user_embedding = kai.nn.new_embedding("user_embedding", dim=4, slots=[101])
    comment_id_embedding = kai.nn.new_embedding("c_id_embedding", dim=64, slots=[201])
    # comment_info_embedding = kai.nn.new_embedding("c_info_embedding", dim=32, slots=[203, 204, 205, 206, 207, 209])
    # position_embedding = kai.nn.new_embedding("position_embedding", dim=8, slots=[208])
    # comment_udp_id_embedding = kai.nn.new_embedding("c_udp_id_embedding", dim=64, slots=[103, 104, 105, 106])
    cmt_emb = kai.nn.get_dense_fea('cmt_emb', dim=64, dtype=tf.float32)
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

    # user_embedding = config.new_embedding("user_embedding", dim=4, slots=[101, 102], **compress_kwargs)
    # comment_id_embedding = config.new_embedding("c_id_embedding", dim=64, slots=[201, 202])
    # comment_info_embedding = config.new_embedding("c_info_embedding", dim=32, slots=[203, 204, 205, 206, 207, 209])
    # position_embedding = config.new_embedding("position_embedding", dim=8, slots=[208])
    # comment_udp_id_embedding = config.new_embedding("c_udp_id_embedding", dim=64, slots=[103, 104, 105, 106])
    comment_id_embedding = config.new_embedding("c_id_embedding", dim=64, slots=[201])
    cmt_emb = config.get_dense_fea('cmt_emb', dim=64, dtype=tf.float32)



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
field_input = tf.concat([comment_id_embedding, cmt_emb], -1)
# expand_xtr = simple_dense_network("expand_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
like_xtr = simple_dense_network("like_xtr", field_input, [64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
# reply_xtr = simple_dense_network("reply_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
# copy_xtr = simple_dense_network("copy_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
# share_xtr = simple_dense_network("share_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
# audience_xtr = simple_dense_network("audience_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
# continuous_expand_xtr = simple_dense_network("continuous_expand_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)


if args.mode == 'train':
    # # define label input and define metrics
    # expand_label = tf.cast(kai.nn.get_dense_fea("expandAction", dim=1, dtype=tf.int64), tf.float32)
    # like_label = tf.cast(kai.nn.get_dense_fea("likeAction", dim=1, dtype=tf.int64), tf.float32)
    # reply_label = tf.cast(kai.nn.get_dense_fea("replyAction", dim=1, dtype=tf.int64), tf.float32)
    # sample_weight = kai.nn.get_dense_fea("sample_weight", dim=1, dtype=tf.float32)

    # define label input and define metrics
    # sample_weight = kai.nn.get_dense_fea("new_sample_weight", dim=1, dtype=tf.float32)
    # ones = tf.ones_like(sample_weight, dtype=tf.float32)
    # zeros = tf.zeros_like(sample_weight, dtype=tf.float32)

    # expandAction_first = tf.cast(kai.nn.get_dense_fea("expandAction_first", dim=1, dtype=tf.int64), tf.float32)
    # expand_label = tf.where(expandAction_first > 0, ones, zeros)
    # continuous_expand_label = tf.where(expandAction_first > 1, ones, zeros)
    
    like_first_label = kai.nn.get_dense_fea("like_action", dim=1, dtype=tf.float32)
    # like_second_label = tf.cast(kai.nn.get_dense_fea("likeAction_second", dim=1, dtype=tf.int64), tf.float32)
    like_first_label = tf.nn.dropout(like_first_label, 0.5)
    ones = tf.ones_like(like_first_label, dtype=tf.float32)
    zeros = tf.zeros_like(like_first_label, dtype=tf.float32)
    like_label = tf.where((like_first_label > 0), ones, zeros)


    vv = kai.nn.get_dense_fea("vv", dim=1, dtype=tf.float32)
    cid = tf.cast(kai.nn.get_dense_fea("comment_id", dim=1, dtype=tf.float32), dtype=tf.int64)

    # reply_first_label = tf.cast(kai.nn.get_dense_fea("replyAction_first", dim=1, dtype=tf.int64), tf.float32)
    # reply_second_label = tf.cast(kai.nn.get_dense_fea("replyAction_second", dim=1, dtype=tf.int64), tf.float32)
    # reply_label = tf.where((reply_first_label > 0) | (reply_second_label > 0), ones, zeros)

    # copy_first_label = tf.cast(kai.nn.get_dense_fea("copyAction_first", dim=1, dtype=tf.int64), tf.float32)
    # copy_second_label = tf.cast(kai.nn.get_dense_fea("copyAction_second", dim=1, dtype=tf.int64), tf.float32)
    # copy_label = tf.where((copy_first_label > 0) | (copy_second_label > 0), ones, zeros)

    # share_first_label = tf.cast(kai.nn.get_dense_fea("shareAction_first", dim=1, dtype=tf.int64), tf.float32)
    # share_second_label = tf.cast(kai.nn.get_dense_fea("shareAction_second", dim=1, dtype=tf.int64), tf.float32)
    # share_label = tf.where((share_first_label > 0) | (share_second_label > 0), ones, zeros)

    # audience_first_label = tf.cast(kai.nn.get_dense_fea("audienceAction_first", dim=1, dtype=tf.int64), tf.float32)
    # audience_second_label = tf.cast(kai.nn.get_dense_fea("audienceAction_second", dim=1, dtype=tf.int64), tf.float32)
    # audience_label = tf.where((audience_first_label > 0) | (audience_second_label > 0), ones, zeros)


    # targets = [
    #     ('expand_predict', expand_xtr, expand_label, sample_weight, "auc"),
    #     ('like_predict', like_xtr, like_label, sample_weight, "auc"),
    #     ('reply_predict', reply_xtr, reply_label, sample_weight, "auc"),
    #     ('copy_predict', copy_xtr, copy_label, sample_weight, "auc"),
    #     ('share_predict', share_xtr, share_label, sample_weight, "auc")
    # ]

    targets = [
        # ('expand_predict', expand_xtr, expand_label, ones, "auc"),
        ('like_predict', like_xtr, like_label, ones, "auc"),
        # ('reply_predict', reply_xtr, reply_label, ones, "auc"),
        # ('copy_predict', copy_xtr, copy_label, ones, "auc"),
        # ('share_predict', share_xtr, share_label, ones, "auc"),
        # ('audience_predict', audience_xtr, audience_label, ones, "auc"),
        # ('continuous_expand_predict', continuous_expand_xtr, continuous_expand_label, ones, "auc"),
    ]

    metric_name, preds, labels, weights, metric_type = zip(*targets)

    # 5. define optimizer
    loss = tf.losses.log_loss(labels, preds, weights, reduction="weighted_sum")
    optimizer = kai.nn.optimizer.Adam(1e-3)
    optimizer.minimize(loss)

    eval_targets = [
        # ('expand_predict', expand_xtr, expand_label, ones, "auc"),
        ('like_predict', like_xtr, like_label, ones, "auc"),
    ]

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

    debug_tensor = {
        "cmt_emb": tf.slice(cmt_emb, [0, 0], [1, -1]),
        "cid": cid[0],
        'vv': vv[0]
    }
    kai.add_run_hook(TensorPrintHook(debug_tensor), "debug_tensor_hook")


    # 6. finish define model structure 
    kai.build_model(optimizer=[optimizer], metrics=eval_targets)
else:
    targets = [
    #   ("expand_xtr", expand_xtr),
      ("like_xtr", like_xtr),
    #   ("reply_xtr", reply_xtr),
    #   ("copy_xtr", copy_xtr),
    #   ("share_xtr", share_xtr),
    #   ("audience_xtr", audience_xtr),
    #   ("continuous_expand_xtr", continuous_expand_xtr),
    ]
    q_names, preds = zip(*targets)
    config.dump_predict_config('./predict/config', targets, input_type=3, extra_preds=q_names)

