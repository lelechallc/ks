""" 在v2基础上 增加 地理位置、请求时间 特征，不修正label.
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

    # common attr
    user_embedding = kai.nn.new_embedding("user_embedding", dim=4, slots=[101, 102])
    context_embedding1 = kai.nn.new_embedding("context_embedding1", dim=32, slots=[114])    # city
    context_embedding2 = kai.nn.new_embedding("context_embedding2", dim=8, slots=[115, 116])  # request_hour, request_day
    comment_udp_id_embedding = kai.nn.new_embedding("c_udp_id_embedding", dim=64, slots=[103, 104, 105, 106])

    # item attr
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

    # common attr
    user_embedding = config.new_embedding("user_embedding", dim=4, slots=[101, 102], **compress_kwargs)
    context_embedding1 = config.new_embedding("context_embedding1", dim=32, slots=[114], **compress_kwargs)  # city
    context_embedding2 = config.new_embedding("context_embedding2", dim=8, slots=[115, 116], **compress_kwargs)  # request_hour, request_day
    comment_udp_id_embedding = config.new_embedding("c_udp_id_embedding", dim=64, slots=[103, 104, 105, 106], **compress_kwargs)
    
    # item attr
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
field_input = tf.concat([user_embedding, comment_id_embedding, comment_info_embedding, position_embedding, comment_udp_id_embedding,
                         context_embedding1, context_embedding2], -1)
expand_xtr = simple_dense_network("expand_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
like_xtr = simple_dense_network("like_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
reply_xtr = simple_dense_network("reply_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
copy_xtr = simple_dense_network("copy_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
share_xtr = simple_dense_network("share_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
audience_xtr = simple_dense_network("audience_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
continuous_expand_xtr = simple_dense_network("continuous_expand_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)

if args.mode == 'train':
    # define label input and define metrics
    sample_weight = kai.nn.get_dense_fea("sample_weight", dim=1, dtype=tf.float32)
    ones = tf.ones_like(sample_weight, dtype=tf.float32)
    zeros = tf.zeros_like(sample_weight, dtype=tf.float32)

    ### 新标签
    # subAtAction_first = kai.nn.get_dense_fea("subAtAction_first", dim=1, dtype=tf.float32)
    # subAtAction_second = kai.nn.get_dense_fea("subAtAction_second", dim=1, dtype=tf.float32)
    # sub_at_label = tf.where((subAtAction_first > 0) | (subAtAction_second > 0), ones, zeros)

    # hateAction_first = kai.nn.get_dense_fea("hateAction_first", dim=1, dtype=tf.float32)
    # hateAction_second = kai.nn.get_dense_fea("hateAction_second", dim=1, dtype=tf.float32)
    # cancelHateAction_first = kai.nn.get_dense_fea("cancelHateAction_first", dim=1, dtype=tf.float32)
    # cancelHateAction_second = kai.nn.get_dense_fea("cancelHateAction_second", dim=1, dtype=tf.float32)
    # hate_label = tf.where((hateAction_first-cancelHateAction_first > 0) | (hateAction_second-cancelHateAction_second > 0), ones, zeros)

    # expandAction_first = kai.nn.get_dense_fea("expandAction_first", dim=1, dtype=tf.float32)
    # expand_label = tf.where(expandAction_first > 0, ones, zeros)
    # continuous_expand_label = tf.where(expandAction_first > 1, ones, zeros)

    # like_first_label = kai.nn.get_dense_fea("likeAction_first", dim=1, dtype=tf.float32)
    # like_second_label = kai.nn.get_dense_fea("likeAction_second", dim=1, dtype=tf.float32)
    # cancelLikeAction_first = kai.nn.get_dense_fea("cancelLikeAction_first", dim=1, dtype=tf.float32)
    # cancelLikeAction_second = kai.nn.get_dense_fea("cancelLikeAction_second", dim=1, dtype=tf.float32)
    # like_label = tf.where((like_first_label-cancelLikeAction_first > 0) | (like_second_label-cancelLikeAction_second > 0), ones, zeros)

    # replyTaskAction_first = kai.nn.get_dense_fea("replyTaskAction_first", dim=1, dtype=tf.float32)
    # replyTaskAction_second = kai.nn.get_dense_fea("replyTaskAction_second", dim=1, dtype=tf.float32)
    # reply_label = tf.where((replyTaskAction_first > 0) | (replyTaskAction_second > 0), ones, zeros)

    # copy_first_label = kai.nn.get_dense_fea("copyAction_first", dim=1, dtype=tf.float32)
    # copy_second_label = kai.nn.get_dense_fea("copyAction_second", dim=1, dtype=tf.float32)
    # copy_label = tf.where((copy_first_label > 0) | (copy_second_label > 0), ones, zeros)

    # share_first_label = kai.nn.get_dense_fea("shareAction_first", dim=1, dtype=tf.float32)
    # share_second_label = kai.nn.get_dense_fea("shareAction_second", dim=1, dtype=tf.float32)
    # share_label = tf.where((share_first_label > 0) | (share_second_label > 0), ones, zeros)

    # audience_first_label = kai.nn.get_dense_fea("audienceAction_first", dim=1, dtype=tf.float32)
    # audience_second_label = kai.nn.get_dense_fea("audienceAction_second", dim=1, dtype=tf.float32)
    # audience_label = tf.where((audience_first_label > 0) | (audience_second_label > 0), ones, zeros)


    ## 老标签
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

    targets = [
        ('expand_predict', expand_xtr, expand_label, ones, "auc"),
        ('like_predict', like_xtr, like_label, ones, "auc"),
        ('reply_predict', reply_xtr, reply_label, ones, "auc"),
        ('copy_predict', copy_xtr, copy_label, ones, "auc"),
        ('share_predict', share_xtr, share_label, ones, "auc"),
        ('audience_predict', audience_xtr, audience_label, ones, "auc"),
        ('continuous_expand_predict', continuous_expand_xtr, continuous_expand_label, ones, "auc"),

    ]

    metric_name, preds, labels, weights, metric_type = zip(*targets)

    # 5. define optimizer
    loss = tf.losses.log_loss(labels, preds, weights, reduction="weighted_sum")
    optimizer = kai.nn.optimizer.Adam(1e-3)
    optimizer.minimize(loss)

    recall_type = kai.nn.get_dense_fea("recall_type", dim=1, dtype=tf.float32)

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

        ('expand_hot', expand_xtr, expand_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('like_hot', like_xtr, like_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('reply_hot', reply_xtr, reply_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('copy_hot', copy_xtr, copy_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('share_hot', share_xtr, share_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('audience_hot', audience_xtr, audience_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('continuous_expand_hot', continuous_expand_xtr, continuous_expand_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),

        ('expand_climb', expand_xtr, expand_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('like_climb', like_xtr, like_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('reply_climb', reply_xtr, reply_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('copy_climb', copy_xtr, copy_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('share_climb', share_xtr, share_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('audience_climb', audience_xtr, audience_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('continuous_expand_climb', continuous_expand_xtr, continuous_expand_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),

        # ('pic_expand_predict', expand_xtr, expand_label, pic_comment, "auc"),
        # ('pic_like_predict', like_xtr, like_label, pic_comment, "auc"),
        # ('pic_reply_predict', reply_xtr, reply_label, pic_comment, "auc"),
        # ('pic_copy_predict', copy_xtr, copy_label, pic_comment, "auc"),
        # ('pic_share_predict', share_xtr, share_label, pic_comment, "auc"),
        # ('pic_audience_predict', audience_xtr, audience_label, pic_comment, "auc"),
        # ('pic_continuous_expand_predict', continuous_expand_xtr, continuous_expand_label, pic_comment, "auc"),

        # ('text_expand_predict', expand_xtr, expand_label,  1 - pic_comment, "auc"),
        # ('text_like_predict', like_xtr, like_label, 1 - pic_comment, "auc"),
        # ('text_reply_predict', reply_xtr, reply_label, 1 - pic_comment, "auc"),
        # ('text_copy_predict', copy_xtr, copy_label, 1-pic_comment, "auc"),
        # ('text_share_predict', share_xtr, share_label, 1-pic_comment, "auc"),
        # ('text_audience_predict', audience_xtr, audience_label, 1-pic_comment, "auc"),
        # ('text_continuous_expand_predict', continuous_expand_xtr, continuous_expand_label, 1-pic_comment, "auc"),
    ]

    # 6. finish define model structure 
    kai.build_model(optimizer=[optimizer], metrics=eval_targets)
else:
    targets = [
        ("expand_xtr", expand_xtr),
        ("like_xtr", like_xtr),
        ("reply_xtr", reply_xtr),
        ("copy_xtr", copy_xtr),
        ("share_xtr", share_xtr),
        ("audience_xtr", audience_xtr),
        ("continuous_expand_xtr", continuous_expand_xtr),
    ]
    q_names, preds = zip(*targets)
    config.dump_predict_config('./predict/config', targets, input_type=3, extra_preds=q_names)

