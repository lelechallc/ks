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
    position_embedding = kai.nn.new_embedding("position_embedding", dim=8, slots=[208])
    comment_udp_id_embedding = kai.nn.new_embedding("c_udp_id_embedding", dim=64, slots=[103, 105, 106])
    
    # gender_emb = kai.nn.new_embedding("gender_embedding", dim=4, slots=[101])
    # age_emb = kai.nn.new_embedding("age_embedding", dim=4, slots=[102])

    # cid_emb = kai.nn.new_embedding("c_id_embedding", dim=64, slots=[201])
    # c_author_emb = kai.nn.new_embedding("c_author_embedding", dim=64, slots=[202])
    # like_cnt_emb = kai.nn.new_embedding("c_like_cnt_embedding", dim=32, slots=[203])
    # reply_cnt_emb = kai.nn.new_embedding("c_reply_cnt_embedding", dim=32, slots=[204])
    # time_emb = kai.nn.new_embedding("c_time_embedding", dim=32, slots=[205])
    # dislike_cnt_emb = kai.nn.new_embedding("c_dislike_cnt_embedding", dim=32, slots=[209])

    # ltr_emb = kai.nn.new_embedding("c_ltr_embedding", dim=32, slots=[206])
    # rtr_emb = kai.nn.new_embedding("c_rtr_embedding", dim=32, slots=[207])
    # show_emb = kai.nn.new_embedding("c_show_embedding", dim=8, slots=[208])

    # photo_id_emb = kai.nn.new_embedding("photo_id_embedding", dim=64, slots=[103])
    # uid_emb = kai.nn.new_embedding("uid_embedding", dim=64, slots=[105])
    # device_id_emb = kai.nn.new_embedding("device_id_embedding", dim=64, slots=[106])


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
    comment_udp_id_embedding = config.new_embedding("c_udp_id_embedding", dim=64, slots=[103, 105, 106])


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

# input_emb_list = [gender_emb, age_emb, cid_emb, c_author_emb, 
#                   like_cnt_emb, reply_cnt_emb, time_emb, dislike_cnt_emb,
#                   ltr_emb, rtr_emb, show_emb, photo_id_emb, uid_emb, device_id_emb]
emb_name_list = [
    "gender_emb", "age_emb", "cid_emb", "c_author_emb",
    "like_cnt_emb", "reply_cnt_emb", "time_emb", "dislike_cnt_emb",
    "ltr_emb", "rtr_emb", "show_emb", "photo_id_emb", "uid_emb", "device_id_emb"
]

# define model structure
field_input = tf.concat([user_embedding, comment_id_embedding, comment_cnt_embedding, comment_xtr_embedding, position_embedding, comment_udp_id_embedding], -1)

# field_input = tf.concat(input_emb_list, -1)

expand_xtr = simple_dense_network("expand_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
like_xtr = simple_dense_network("like_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
reply_xtr = simple_dense_network("reply_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)

if args.mode == 'train':
    # define label input and define metrics

    # expand_label = tf.cast(kai.nn.get_dense_fea("expandAction", dim=1, dtype=tf.int64), tf.float32)
    # like_label = tf.cast(kai.nn.get_dense_fea("likeAction", dim=1, dtype=tf.int64), tf.float32)
    # reply_label = tf.cast(kai.nn.get_dense_fea("replyAction", dim=1, dtype=tf.int64), tf.float32)
    # sample_weight = kai.nn.get_dense_fea("sample_weight", dim=1, dtype=tf.float32)

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

    recall_type_tag = tf.cast(kai.nn.get_dense_fea("recall_type", dim=1, dtype=tf.int64), tf.float32)

    eval_targets = [
        ('expand_predict', expand_xtr, expand_label, sample_weight, "auc"),
        ('like_predict', like_xtr, like_label, sample_weight, "auc"),
        ('reply_predict', reply_xtr, reply_label, sample_weight, "auc"),

        ('expand_time', expand_xtr, expand_label, tf.where(tf.less_equal(recall_type_tag, 0.9), ones, zeros), "auc"),
        ('expand_base', expand_xtr, expand_label, tf.where(tf.less_equal(0.9, recall_type_tag), ones, zeros), "auc"),
        ('expand_hot', expand_xtr, expand_label, tf.where(tf.less_equal(recall_type_tag, 1.1), tf.where(tf.less_equal(0.9, recall_type_tag), ones, zeros), zeros), "auc"),
        ('expand_climb', expand_xtr, expand_label, tf.where(tf.less_equal(recall_type_tag, 4.1), tf.where(tf.less_equal(1.1, recall_type_tag), ones, zeros), zeros), "auc"),

        ('like_time', like_xtr, like_label, tf.where(tf.less_equal(recall_type_tag, 0.9), ones, zeros), "auc"),
        ('like_base', like_xtr, like_label, tf.where(tf.less_equal(0.9, recall_type_tag), ones, zeros), "auc"),
        ('like_hot', like_xtr, like_label, tf.where(tf.less_equal(recall_type_tag, 1.1), tf.where(tf.less_equal(0.9, recall_type_tag), ones, zeros), zeros), "auc"),
        ('like_climb', like_xtr, like_label, tf.where(tf.less_equal(recall_type_tag, 4.1), tf.where(tf.less_equal(1.1, recall_type_tag), ones, zeros), zeros), "auc"),

        ('reply_time', reply_xtr, reply_label, tf.where(tf.less_equal(recall_type_tag, 0.9), ones, zeros), "auc"),
        ('reply_base', reply_xtr, reply_label, tf.where(tf.less_equal(0.9, recall_type_tag), ones, zeros), "auc"),
        ('reply_hot', reply_xtr, reply_label, tf.where(tf.less_equal(recall_type_tag, 1.1), tf.where(tf.less_equal(0.9, recall_type_tag), ones, zeros), zeros), "auc"),
        ('reply_climb', reply_xtr, reply_label, tf.where(tf.less_equal(recall_type_tag, 4.1), tf.where(tf.less_equal(1.1, recall_type_tag), ones, zeros), zeros), "auc"),
    ]

    
    # for emb, emb_name in zip(input_emb_list, emb_name_list):
    #     tf.summary.scalar(emb_name, tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(emb), axis=1))))
    # tf.summary.scalar('loss', loss)

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

