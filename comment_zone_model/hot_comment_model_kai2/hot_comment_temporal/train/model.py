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
    from kconf.get_config import get_double_config
    import numpy as np

    user_embedding = kai.nn.new_embedding("user_embedding", dim=4, slots=[101, 102])
    comment_id_embedding = kai.nn.new_embedding("c_id_embedding", dim=64, slots=[201, 202])
    comment_info_embedding = kai.nn.new_embedding("c_info_embedding", dim=32, slots=[203, 204, 205, 206, 207, 209])
    position_embedding = kai.nn.new_embedding("position_embedding", dim=8, slots=[208])
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

    class GetKconfHook(kai.training.RunHookBase):
        def __init__(self, tensor_names):
            self.tensor_names = tensor_names

        def before_pass_run(self, pass_run_context):
            """
            每个 pass 只会 print 一次
            """
            self.has_print = False

        def before_step_run(self, step_run_context):
            min_reply_fea = float(get_double_config("cc.knowledgeGraph.commentTrainExpandLabel"))
            random_keep_fea = float(get_double_config("cc.knowledgeGraph.commentTrainRandomKeep"))

            if not self.has_print:
                print("cc.knowledgeGraph.commentTrainExpandLabel kconf value: ", min_reply_fea)
                print("cc.knowledgeGraph.commentTrainRandomKeep kconf value: ", random_keep_fea)
                self.has_print = True

            values = [np.array([min_reply_fea]), np.array([random_keep_fea])]
            feed_dict = dict(zip(self.tensor_names, values))
            return kai.training.StepRunArgs(feed_dict=feed_dict)

    # define label input and define metrics
    sample_weight = kai.nn.get_dense_fea("new_sample_weight", dim=1, dtype=tf.float32)
    ones = tf.ones_like(sample_weight, dtype=tf.float32)
    zeros = tf.zeros_like(sample_weight, dtype=tf.float32)

    expand_label = tf.cast(kai.nn.get_dense_fea("expandAction_first", dim=1, dtype=tf.int64), tf.float32)
    expand_label = tf.where(expand_label > 0, ones, zeros)
    
    like_first_label = tf.cast(kai.nn.get_dense_fea("likeAction_first", dim=1, dtype=tf.int64), tf.float32)
    like_second_label = tf.cast(kai.nn.get_dense_fea("likeAction_second", dim=1, dtype=tf.int64), tf.float32)
    like_label = tf.where((like_first_label > 0) | (like_second_label > 0), ones, zeros)

    reply_first_label = tf.cast(kai.nn.get_dense_fea("replyAction_first", dim=1, dtype=tf.int64), tf.float32)
    reply_second_label = tf.cast(kai.nn.get_dense_fea("replyAction_second", dim=1, dtype=tf.int64), tf.float32)
    reply_label = tf.where((reply_first_label > 0) | (reply_second_label > 0), ones, zeros)

    # data clean
    # reply_dense_fea = tf.cast(kai.nn.get_dense_fea("reply_cnt", dim=1, dtype=tf.int64), tf.float32)
    # min_reply_fea = tf.placeholder_with_default(input=tf.constant([1.0]), shape=[1], name="min_reply_fea")
    # random_keep_fea = tf.placeholder_with_default(input=tf.constant([1.0]), shape=[1], name="random_keep_fea")
    # kai.add_run_hook(GetKconfHook([min_reply_fea.name, random_keep_fea.name]), "get_kconf_hook")

    # random_value = tf.random_uniform(shape=tf.shape(expand_label), dtype=tf.float32)

    # # refine expand label
    # expand_label = tf.where((reply_dense_fea < min_reply_fea) & (random_value > random_keep_fea), zeros, expand_label)

    length = tf.cast(tf.shape(expand_label)[0], dtype=tf.float32)
    expand_cnt = tf.reduce_sum(expand_label)
    like_cnt = tf.reduce_sum(like_label)
    reply_cnt = tf.reduce_sum(reply_label)

    targets = [
        ('expand_predict', expand_xtr, expand_label, ones, "auc"),
        ('like_predict', like_xtr, like_label, ones, "auc"),
        ('reply_predict', reply_xtr, reply_label, ones, "auc")
    ]

    metric_name, preds, labels, weights, metric_type = zip(*targets)


    # 5. define optimizer
    loss = tf.losses.log_loss(labels, preds, weights, reduction="weighted_sum")
    optimizer = kai.nn.optimizer.Adam(1e-3)
    optimizer.minimize(loss)

    realshow_cnt = tf.cast(kai.nn.get_dense_fea("realshow_cnt", dim=1, dtype=tf.int64), tf.float32)
    vv_80 = tf.where(realshow_cnt > 80, ones, zeros)
    vv_200 = tf.where(realshow_cnt > 200, ones, zeros)

    comment_genre = tf.cast(kai.nn.get_dense_fea("comment_genre", dim=1, dtype=tf.int64), tf.float32)
    pic_comment = tf.where(comment_genre > 0, ones, zeros)

    eval_targets = [
        ('expand_predict', expand_xtr, expand_label, ones, "auc"),
        ('like_predict', like_xtr, like_label, ones, "auc"),
        ('reply_predict', reply_xtr, reply_label, ones, "auc"),

        ('vv_80_expand_predict', expand_xtr, expand_label, vv_80, "auc"),
        ('vv_80_like_predict', like_xtr, like_label, vv_80, "auc"),
        ('vv_80_reply_predict', reply_xtr, reply_label, vv_80, "auc"),
        ('com_vv_80_expand_predict', expand_xtr, expand_label, 1 - vv_80, "auc"),
        ('com_vv_80_like_predict', like_xtr, like_label, 1 - vv_80, "auc"),
        ('com_vv_80_reply_predict', reply_xtr, reply_label, 1 - vv_80, "auc"),

        ('vv_200_expand_predict', expand_xtr, expand_label, vv_200, "auc"),
        ('vv_200_like_predict', like_xtr, like_label, vv_200, "auc"),
        ('vv_200_reply_predict', reply_xtr, reply_label, vv_200, "auc"),
        ('com_vv_200_expand_predict', expand_xtr, expand_label, 1 - vv_200, "auc"),
        ('com_vv_200_like_predict', like_xtr, like_label, 1 - vv_200, "auc"),
        ('com_vv_200_reply_predict', reply_xtr, reply_label, 1 - vv_200, "auc"),

        ('pic_expand_predict', expand_xtr, expand_label, pic_comment, "auc"),
        ('pic_like_predict', like_xtr, like_label, pic_comment, "auc"),
        ('pic_reply_predict', reply_xtr, reply_label, pic_comment, "auc"),
        ('text_expand_predict', expand_xtr, expand_label,  1 - pic_comment, "auc"),
        ('text_like_predict', like_xtr, like_label, 1 - pic_comment, "auc"),
        ('text_reply_predict', reply_xtr, reply_label, 1 - pic_comment, "auc"),
    ]

    # debug_tensor = {
    #     "expand_weight": length / expand_cnt,
    #     "reply_weight": length / reply_cnt,
    #     "like_weight": length / like_cnt,
    #     "min_reply_fea": min_reply_fea,
    #     "random_keep_fea": random_keep_fea,
    # }
    # kai.add_run_hook(TensorPrintHook(debug_tensor), "debug_tensor_hook")

    # 6. finish define model structure 
    kai.build_model(optimizer=[optimizer], metrics=eval_targets)
else:
    targets = [
      ("expand_xtr", expand_xtr),
      ("like_xtr", like_xtr),
      ("reply_xtr", reply_xtr),
    ]
    q_names, preds = zip(*targets)
    config.dump_predict_config('./predict/config', targets, input_type=3, extra_preds=q_names)

