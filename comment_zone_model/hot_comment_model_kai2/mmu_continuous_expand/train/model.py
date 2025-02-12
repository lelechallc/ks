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

    user_embedding = kai.nn.new_embedding("user_embedding", dim=4, slots=[101, 102])
    comment_id_embedding = kai.nn.new_embedding("c_id_embedding", dim=64, slots=[201, 202])
    comment_cnt_embedding = kai.nn.new_embedding("c_cnt_embedding", dim=32, slots=[203, 204, 205, 209])
    comment_xtr_embedding = kai.nn.new_embedding("c_xtr_embedding", dim=32, slots=[206, 207])
    # comment_mmu_score_embedding = kai.nn.new_embedding("c_mmu_score_embedding", dim=32, slots=[407, 408, 409])
    comment_mmu_score_embedding = kai.nn.new_embedding("c_mmu_score_embedding", dim=32, slots=[408])

    position_embedding = kai.nn.new_embedding("position_embedding", dim=8, slots=[208])

    # mmu提供的comment_content embedding
    mmu_hetu_content_emb = kai.nn.get_dense_fea("mmu_hetu_content_emb", dim=128, dtype=tf.float32)
    mmu_clip_content_emb = kai.nn.get_dense_fea("mmu_clip_content_emb", dim=256, dtype=tf.float32)
    mmu_bert_content_emb = kai.nn.get_dense_fea("mmu_bert_content_emb", dim=256, dtype=tf.float32)

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
    comment_cnt_embedding = config.new_embedding("c_cnt_embedding", dim=32, slots=[203, 204, 205, 209])
    comment_xtr_embedding = config.new_embedding("c_xtr_embedding", dim=32, slots=[206, 207])
    position_embedding = config.new_embedding("position_embedding", dim=8, slots=[208])
    # comment_mmu_score_embedding = config.new_embedding("c_mmu_score_embedding", dim=32, slots=[407, 408, 409])
    comment_mmu_score_embedding = config.new_embedding("c_mmu_score_embedding", dim=32, slots=[408])

    mmu_hetu_content_emb = config.get_extra_param("mmu_hetu_content_emb", size=128)
    mmu_clip_content_emb = config.get_extra_param("mmu_clip_content_emb", size=256)
    mmu_bert_content_emb = config.get_extra_param("mmu_bert_content_emb", size=256)

def simple_dense_network(name, inputs, units, dropout=0.0, act=tf.nn.tanh, last_act=tf.nn.sigmoid, stop_gradient=False):
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

mmu_content_emb_out = simple_dense_network("mmu_content_emb", tf.concat([mmu_hetu_content_emb, mmu_clip_content_emb, mmu_bert_content_emb], axis=-1), [256, 64], 0.0, act=tf.nn.leaky_relu, last_act=None)
# define model structure
field_input = tf.concat([user_embedding, comment_id_embedding, comment_cnt_embedding, comment_xtr_embedding, position_embedding, comment_mmu_score_embedding, mmu_content_emb_out], -1)

expand_xtr = simple_dense_network("expand_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
like_xtr = simple_dense_network("like_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
reply_xtr = simple_dense_network("reply_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
continuous_expand_xtr = simple_dense_network("continuous_expand_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)

if args.mode == 'train':
    # define label input and define metrics
    # expand_label = kai.nn.get_dense_fea("expandAction_v", dim=1, dtype=tf.float32)
    # like_label = kai.nn.get_dense_fea("likeAction_v", dim=1, dtype=tf.float32)
    # reply_label = kai.nn.get_dense_fea("replyAction_v", dim=1, dtype=tf.float32)

    # sample_weight = kai.nn.get_dense_fea("sample_weight", dim=1, dtype=tf.float32)

    # 修正new label
    sample_weight = kai.nn.get_dense_fea("new_sample_weight", dim=1, dtype=tf.float32)
    ones = tf.ones_like(sample_weight, dtype=tf.float32)
    zeros = tf.zeros_like(sample_weight, dtype=tf.float32)

    expand_action_cnt = kai.nn.get_dense_fea("expandAction_first_v", dim=1, dtype=tf.float32)
    expand_label = tf.where(expand_action_cnt > 0.0, ones, zeros)

    continuous_expand_label = tf.where(expand_action_cnt > 1.0, ones, zeros)
    
    like_first_label = kai.nn.get_dense_fea("likeAction_first_v", dim=1, dtype=tf.float32)
    like_second_label = kai.nn.get_dense_fea("likeAction_second_v", dim=1, dtype=tf.float32)
    like_label = tf.where((like_first_label > 0.0) | (like_second_label > 0.0), ones, zeros)

    reply_first_label = kai.nn.get_dense_fea("replyAction_first_v", dim=1, dtype=tf.float32)
    reply_second_label = kai.nn.get_dense_fea("replyAction_second_v", dim=1, dtype=tf.float32)
    reply_label = tf.where((reply_first_label > 0.0) | (reply_second_label > 0.0), ones, zeros)

    targets = [
        ('expand_predict', expand_xtr, expand_label, ones, "auc"),
        ('like_predict', like_xtr, like_label, ones, "auc"),
        ('reply_predict', reply_xtr, reply_label, ones, "auc"),
        ('continuous_expand_predict', continuous_expand_xtr, continuous_expand_label, ones, "auc"),
    ]

    metric_name, preds, labels, weights, metric_type = zip(*targets)

    # 5. define optimizer
    loss = tf.losses.log_loss(labels, preds, weights, reduction="weighted_sum")
    optimizer = kai.nn.optimizer.Adam(1e-3)
    optimizer.minimize(loss)

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
        "mmu_hetu_content_emb": tf.slice(mmu_hetu_content_emb, [0, 0], [1, -1]),
        "mmu_clip_content_emb": tf.slice(mmu_clip_content_emb, [0, 0], [1, -1]),
        "mmu_bert_content_emb": tf.slice(mmu_bert_content_emb, [0, 0], [1, -1]),
        "mmu_content_emb_out": tf.slice(mmu_content_emb_out, [0, 0], [1, -1]),
        "comment_mmu_score_embedding": tf.slice(comment_mmu_score_embedding, [0, 0], [1, -1]),
    }
    kai.add_run_hook(TensorPrintHook(debug_tensor), "debug_tensor_hook")

    # 6. finish define model structure 
    kai.build_model(optimizer=[optimizer], metrics=targets)
else:
    targets = [
      ("expand_xtr", expand_xtr),
      ("like_xtr", like_xtr),
      ("reply_xtr", reply_xtr),
      ("continuous_expand_xtr", continuous_expand_xtr),
    ]
    q_names, preds = zip(*targets)
    config.dump_predict_config('./predict/config', targets, input_type=3, extra_preds=q_names)

