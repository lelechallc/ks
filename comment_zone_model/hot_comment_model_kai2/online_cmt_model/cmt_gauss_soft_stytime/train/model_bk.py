import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'predict', 'eval'], default='train')
parser.add_argument('--dryrun', dest="dryrun", const=True, default=False, nargs='?')
parser.add_argument('--with_kai', action="store_true")
args = parser.parse_args()

# 1. define sparse input
if args.mode == 'train' or args.mode == 'eval':
    import tensorflow.compat.v1 as tf
    import kai.tensorflow as kai
    # kai.set_load_dense_func(default_load_dense_func)

    user_embedding = kai.nn.new_embedding("user_embedding", dim=4, slots=[101, 102])
    pid_embedding = kai.nn.new_embedding("pid_embedding", dim=64, slots=[103])
    aid_embedding = kai.nn.new_embedding("aid_embedding", dim=64, slots=[104])
    uid_embedding = kai.nn.new_embedding("uid_embedding", dim=64, slots=[105])
    did_embedding = kai.nn.new_embedding("did_embedding", dim=64, slots=[106])
    context_embedding = kai.nn.new_embedding("context_embedding", dim=32, slots=[110, 111])

    comment_id_embedding = kai.nn.new_embedding("c_id_embedding", dim=64, slots=[201, 202])
    comment_info_embedding = kai.nn.new_embedding("c_info_embedding", dim=32, slots=[203, 204, 205, 206, 207, 209])
    position_embedding = kai.nn.new_embedding("position_embedding", dim=8, slots=[208])
    comment_genre_embedding = kai.nn.new_embedding("comment_genre_embedding", dim=8, slots=[250])
    comment_length_embedding = kai.nn.new_embedding("comment_length_embedding", dim=32, slots=[251])
    
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
    pid_embedding = config.new_embedding("pid_embedding", dim=64, slots=[103], **compress_kwargs)
    aid_embedding = config.new_embedding("aid_embedding", dim=64, slots=[104], **compress_kwargs)
    uid_embedding = config.new_embedding("uid_embedding", dim=64, slots=[105], **compress_kwargs)
    did_embedding = config.new_embedding("did_embedding", dim=64, slots=[106], **compress_kwargs)
    context_embedding = config.new_embedding("context_embedding", dim=32, slots=[110, 111], **compress_kwargs)

    comment_id_embedding = config.new_embedding("c_id_embedding", dim=64, slots=[201, 202])
    comment_info_embedding = config.new_embedding("c_info_embedding", dim=32, slots=[203, 204, 205, 206, 207, 209])
    position_embedding = config.new_embedding("position_embedding", dim=8, slots=[208])
    comment_genre_embedding = config.new_embedding("comment_genre_embedding", dim=8, slots=[250])
    comment_length_embedding = config.new_embedding("comment_length_embedding", dim=32, slots=[251])


def simple_dense_network(name, inputs, units, dropout=0, act=tf.nn.tanh, last_act=tf.nn.sigmoid, stop_gradient=False):
    if stop_gradient:
        output = tf.stop_gradient(inputs)
    else:
        output = inputs
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        for i, unit in enumerate(units):
            if dropout > 0 and i>0 and i<len(units) - 1:
                output = tf.layers.dropout(output, dropout, training=(args.mode == 'train'))
            if i == len(units) - 1:
                act = last_act
            output = tf.layers.dense(output, unit, activation=act,
                                  kernel_initializer=tf.glorot_uniform_initializer())
        return output
    
# define model structure
field_input = tf.concat([user_embedding, comment_id_embedding, comment_info_embedding, position_embedding, pid_embedding, aid_embedding, uid_embedding, did_embedding, context_embedding, comment_genre_embedding, comment_length_embedding], -1)
input_wo_pos = tf.concat([user_embedding, comment_id_embedding, comment_info_embedding, pid_embedding, aid_embedding, uid_embedding, did_embedding, context_embedding, comment_genre_embedding, comment_length_embedding], -1)
input_did_pos = tf.concat([did_embedding, position_embedding], -1)
expand_xtr = simple_dense_network("expand_xtr", field_input, [256, 128, 64, 1], 0.3, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
like_xtr = simple_dense_network("like_xtr", field_input, [256, 128, 64, 1], 0.3, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
# like_second_xtr = simple_dense_network("like_second_xtr", field_input, [256, 128, 64, 1], 0.3, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
reply_xtr = simple_dense_network("reply_xtr", field_input, [256, 128, 64, 1], 0.3, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
# reply_second_xtr = simple_dense_network("reply_task_second_xtr", field_input, [256, 128, 64, 1], 0.3, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
copy_xtr = simple_dense_network("copy_xtr", field_input, [256, 128, 64, 1], 0.3, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
share_xtr = simple_dense_network("share_xtr", field_input, [256, 128, 64, 1], 0.3, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
audience_xtr = simple_dense_network("audience_xtr", field_input, [256, 128, 64, 1], 0.3, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
continuous_expand_xtr = simple_dense_network("continuous_expand_xtr", field_input, [256, 128, 64, 1], 0.3, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
# hate_xtr = simple_dense_network("hate_xtr", field_input, [256, 128, 64, 1], 0.3, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
# sub_at_xtr = simple_dense_network("sub_at_xtr", field_input, [256, 128, 64, 1], 0.3, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
staytime_logit = simple_dense_network("staytime", input_wo_pos, [256, 128, 61], 0.3, act=tf.nn.leaky_relu, last_act=None)
staytime_pos_bias_logit = simple_dense_network("staytime_pos_bias", input_did_pos, [128, 61], 0.3, act=tf.nn.leaky_relu, last_act=None)
if args.mode == 'train' or args.mode == 'eval':
  staytime_logit = staytime_logit + staytime_pos_bias_logit
staytime_pred = tf.nn.softmax(staytime_logit)

buk_nums = 61

if args.mode == 'train' or args.mode == 'eval':
    # define label input and define metrics
    recall_type = kai.nn.get_dense_fea("recall_type", dim=1, dtype=tf.float32)
    ones = tf.ones_like(recall_type, dtype=tf.float32)
    zeros = tf.zeros_like(recall_type, dtype=tf.float32)

    expandAction_first = kai.nn.get_dense_fea("expandAction_first", dim=1, dtype=tf.float32)
    expand_label = tf.where(expandAction_first > 0, ones, zeros)
    continuous_expand_label = tf.where(expandAction_first > 1, ones, zeros)
    
    like_first_label = kai.nn.get_dense_fea("likeAction_first", dim=1, dtype=tf.float32)
    like_second_label = kai.nn.get_dense_fea("likeAction_second", dim=1, dtype=tf.float32)
    like_label = tf.where((like_first_label > 0) | (like_second_label > 0), ones, zeros)

    reply_first_label = kai.nn.get_dense_fea("replyAction_first", dim=1, dtype=tf.float32)
    reply_second_label = kai.nn.get_dense_fea("replyAction_second", dim=1, dtype=tf.float32)
    reply_label = tf.where((reply_first_label > 0) | (reply_second_label > 0), ones, zeros)

    copy_first_label = kai.nn.get_dense_fea("copyAction_first", dim=1, dtype=tf.float32)
    copy_second_label = kai.nn.get_dense_fea("copyAction_second", dim=1, dtype=tf.float32)
    copy_label = tf.where((copy_first_label > 0) | (copy_second_label > 0), ones, zeros)

    share_first_label = kai.nn.get_dense_fea("shareAction_first", dim=1, dtype=tf.float32)
    share_second_label = kai.nn.get_dense_fea("shareAction_second", dim=1, dtype=tf.float32)
    share_label = tf.where((share_first_label > 0) | (share_second_label > 0), ones, zeros)

    audience_first_label = kai.nn.get_dense_fea("audienceAction_first", dim=1, dtype=tf.float32)
    audience_second_label = kai.nn.get_dense_fea("audienceAction_second", dim=1, dtype=tf.float32)
    audience_label = tf.where((audience_first_label > 0) | (audience_second_label > 0), ones, zeros)

    staytime_label = kai.nn.get_dense_fea("stayDurationMs", dim=1, dtype=tf.float32)
    staytime_label = tf.clip_by_value(staytime_label / 1000, 0, 60)

    targets = [
        ('expand_predict', expand_xtr, expand_label, ones, "auc", 1.0),
        ('like_predict', like_xtr, like_label, ones, "auc", 1.0),
        ('reply_predict', reply_xtr, reply_label, ones, "auc", 1.0),
        ('copy_predict', copy_xtr, copy_label, ones, "auc", 1.0),
        ('share_predict', share_xtr, share_label, ones, "auc", 1),
        ('audience_predict', audience_xtr, audience_label, ones, "auc", 1.0),
        ('continuous_expand_predict', continuous_expand_xtr, continuous_expand_label, ones, "auc", 1.0),
    ]

    metric_name, preds, labels, weights, metric_type, loss_weight = zip(*targets)

    # 5. define optimizer
    loss = 0.0
    for metric_name, preds, labels, weights, metric_type, loss_weight in targets:
      loss_task =  tf.losses.log_loss(labels, preds, weights, reduction="weighted_sum")
      loss += loss_weight * loss_task
      tf.summary.scalar("task_{}_loss".format(metric_name), tf.reduce_mean(loss_task))
      tf.summary.scalar("task_{}_pred".format(metric_name), tf.reduce_mean(preds))
      tf.summary.scalar("task_{}_label".format(metric_name), tf.reduce_mean(labels))
    # stay_time loss
    loss_weights  = {"staytime": 1.0}
    sigma = 0.1
    normal_dist = tf.distributions.Normal(0.0, sigma)
    task_label = tf.squeeze(staytime_label, axis=[-1])
    print("label_shape", task_label.shape)
    label_bins = tf.tile(
        tf.range(buk_nums, dtype=tf.float32)[tf.newaxis, :],
        [tf.shape(task_label)[0], 1]) - task_label[:, tf.newaxis]
    print("label_bins_shape", label_bins.shape)  # [?, buk_nums]
    softy = normal_dist.prob(label_bins)
    print("soft_shape", softy.shape)
    soft_label = softy / tf.reduce_sum(softy, axis=-1, keepdims=True)
    print("soft_label_shape", soft_label.shape)
    bin_val = tf.tile(tf.range(buk_nums, dtype=tf.float32)[tf.newaxis, :], [tf.shape(task_label)[0], 1])
    print("task_preds_idx_shape", staytime_pred.shape)
    staytime_pred = tf.reduce_sum(bin_val * staytime_pred, axis=-1)[:, tf.newaxis]
    print("task_pred_new_shape", staytime_pred.shape) # [?, 1]
    print(staytime_logit.shape)  # [?, buk_nums]
    print(soft_label.shape)  # [?, buk_nums]
    loss_task = tf.nn.softmax_cross_entropy_with_logits(
        labels=soft_label, logits=staytime_logit)
    tf.summary.scalar("task_{}_loss".format("staytime"), tf.reduce_mean(loss_task))
    loss += loss_weights["staytime"] * loss_task
    optimizer = kai.nn.optimizer.Adam(1e-3)
    optimizer.minimize(loss)


    comment_genre = kai.nn.get_dense_fea("comment_genre", dim=1, dtype=tf.float32)
    pic_comment = tf.where(comment_genre > 0, ones, zeros)

    eval_targets = [
        ('expand_predict', expand_xtr, expand_label, ones, "auc"),
        ('like_predict', like_xtr, like_label, ones, "auc"),
        ('reply_predict', reply_xtr, reply_label, ones, "auc"),
        ('copy_predict', copy_xtr, copy_label, ones, "auc"),
        ('share_predict', share_xtr, share_label, ones, "auc"),
        ('audience_predict', audience_xtr, audience_label, ones, "auc"),
        ('continuous_expand_predict', continuous_expand_xtr, continuous_expand_label, ones, "auc"),
        ('duration_predict', staytime_pred, staytime_label, ones, 'linear_regression'),

        # ('expand_hot', expand_xtr, expand_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        # ('like_hot', like_xtr, like_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        # ('reply_hot', reply_xtr, reply_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        # ('copy_hot', copy_xtr, copy_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        # ('share_hot', share_xtr, share_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        # ('audience_hot', audience_xtr, audience_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        # ('continuous_expand_hot', continuous_expand_xtr, continuous_expand_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),

        # ('expand_climb', expand_xtr, expand_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        # ('like_climb', like_xtr, like_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        # ('reply_climb', reply_xtr, reply_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        # ('copy_climb', copy_xtr, copy_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        # ('share_climb', share_xtr, share_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        # ('audience_climb', audience_xtr, audience_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        # ('continuous_expand_climb', continuous_expand_xtr, continuous_expand_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
    ]

    # 6. finish define model structure 
    kai.build_model(optimizer=[optimizer], metrics=eval_targets)
else:
    bin_val = tf.tile(tf.range(buk_nums, dtype=tf.float32)[tf.newaxis, :],
                                [tf.shape(staytime_pred)[0], 1])
    print("bin_val", bin_val.shape)
    print("pre task pred", task_preds[task_name].shape)
    staytime_pred = tf.reduce_sum(bin_val * staytime_pred, axis=-1)[:, tf.newaxis]
    print("task pred", task_pred.shape)
    targets = [
        ("expand_xtr", expand_xtr),
        ("like_xtr", like_xtr),
        # ("like_second_xtr", like_second_xtr),
        ("reply_xtr", reply_xtr),
        # ("reply_second_xtr", reply_second_xtr),
        ("copy_xtr", copy_xtr),
        ("share_xtr", share_xtr),
        ("audience_xtr", audience_xtr),
        ("continuous_expand_xtr", continuous_expand_xtr),
        # ("hate_xtr", hate_xtr),
        # ("like_xtr", 1-(1-like_first_xtr)*(1-like_second_xtr)),
        # ("reply_xtr", 1-(1-reply_first_xtr)*(1-reply_second_xtr)),
        # ("sub_at_xtr", sub_at_xtr),
        ("duration_predict", staytime_pred),
    ]
    q_names, preds = zip(*targets)
    config.dump_predict_config('./predict/config', targets, input_type=3, extra_preds=q_names)
