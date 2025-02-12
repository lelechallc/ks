""" 在v2基础上，把点赞、回复一二级拆开，增加hate、subat、单评论停留时长信号、单列的uid、pid embedding。
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'predict', 'eval'], default='train')
parser.add_argument('--dryrun', dest="dryrun", const=True, default=False, nargs='?')
parser.add_argument('--with_kai', action="store_true")
args = parser.parse_args()


def default_load_dense_func(
    warmup_weight: dict, warmup_extra: dict, ps_weight: dict, ps_extra: dict, tf_weight: dict,
        load_option):
    """
    该函数的功能是聚合从各处加载的dense模型，生成最终的weight与extra。
    这个函数只有加载init_model_path指定的模型时会被调用。

    该函数遵循以下基本逻辑:
    1、若有warmup_weight/extra, 优先使用
    2、若无warmup，weight优先使用tf_weight, extra优先使用ps_extra
    3、若warmup中dense参数非全量，则会使用tf_weight进行补全, extra使用ps_extra补全

    Arguments:
        warmup_weight {[dict]} -- [从保存的model加载得到的weight]
        warmup_extra {[dict]} -- [从保存的model加载得到的extra]
        ps_weight {[dict]} -- [从参数服务器上拉取的weight]
        ps_extra {[dict]} -- [从参数服务器上拉取的extra]
        tf_weight {[dict]} -- [tensorflow本地通过初始化op生成的weight]
        load_option {dict} -- [控制当次 load 行为的所有配置]

    Raises:
        path: [description]

    Returns:
        weight [dcit] -- [最终聚合得到的weight]
        extra  [dict] -- [最终聚合得到的extra]
    """
    # 加载其他路径模型，宽松检查；加载当前路径模型，严格检查。
    strict = not (load_option.load_from_other_model_path)

    if strict:
        tf_weight_keys = set(tf_weight.keys())
        warmup_weight_keys = set(warmup_weight.keys())
        warmup_extra_keys = set(warmup_extra.keys())
        ps_extra_keys = set(ps_extra.keys())
        # assert len(tf_weight) == len(warmup_weight), "参数数量不匹配：{} vs {}，模型多定义了[{}]，weight 多加载了[{}]".format(
        #     len(tf_weight), len(warmup_weight),
        #     tf_weight_keys.difference(warmup_weight_keys), warmup_weight_keys.difference(tf_weight_keys))
        if len(tf_weight) != len(warmup_weight):
            print("参数数量不匹配：{} vs {}，模型多定义了[{}]，weight 多加载了[{}]".format(
                len(tf_weight), len(warmup_weight),
                tf_weight_keys.difference(warmup_weight_keys), warmup_weight_keys.difference(tf_weight_keys)))

        assert len(tf_weight) == len(warmup_extra), "参数数量不匹配：{} vs {}，模型多定义了[{}]，extra 多加载了[{}]".format(
            len(tf_weight), len(warmup_extra),
            tf_weight_keys.difference(warmup_extra_keys), warmup_extra_keys.difference(tf_weight_keys))
        assert len(tf_weight) == len(ps_extra), "参数数量不匹配：{} vs {}，模型多定义了[{}]，ps extra 多拉取了[{}]".format(
            len(tf_weight), len(ps_extra),
            tf_weight_keys.difference(ps_extra_keys), ps_extra_keys.difference(tf_weight_keys))
        for name in tf_weight:
            assert name in warmup_weight, "加载的权重中没有 {}".format(name)
            assert tf_weight[name].size == warmup_weight[name].size, "{} 的 weight size 不匹配：模型定义了 {} vs 加载了 {}".format(
                name, tf_weight[name].size, warmup_weight[name].size)
            assert name in warmup_extra, "加载的优化器参数中没有 {}".format(name)
            assert ps_extra[name].size == warmup_extra[name].size, "{} 的 extra size 不匹配：模型定义了 {} vs 加载了 {}".format(
                name, ps_extra[name].size, warmup_extra[name].size)

    weight = None
    extra = None
    dense_variable_nums = len(tf_weight)

    for var_name in list(warmup_weight):
        if var_name not in tf_weight:
            print("加载的 dense variable({}) 在运行时不存在，其值被忽略。".format(var_name))  # noqa
            del warmup_weight[var_name]
            del warmup_extra[var_name]
        elif warmup_weight[var_name].size != tf_weight[var_name].size:
            print("加载的 dense variable({}) size ({} vs {}) 不匹配，其值被忽略".format(var_name, warmup_weight[var_name].size, tf_weight[var_name].size))  # noqa
            del warmup_weight[var_name]
            del warmup_extra[var_name]
    weight = warmup_weight

    for var_name in list(warmup_extra):
        if var_name not in ps_extra:
            print("加载的 dense variable extra({}) 在运行时不存在，其值被忽略。".format(var_name))  # noqa
            del warmup_extra[var_name]
        elif warmup_extra[var_name].size != ps_extra[var_name].size:
            print("加载的 dense variable extra({}) size ({} vs {}) 不匹配，其值被忽略".format(var_name, warmup_extra[var_name].size, ps_extra[var_name].size))  # noqa
            del warmup_extra[var_name]
    extra = warmup_extra

    if len(weight) < dense_variable_nums:
        for var_name, var in tf_weight.items():
            if var_name not in weight:
                print(
                    "加载的权重中未找到 {}, 使用 tensorflow 初始化的值".format(var_name))
                weight[var_name] = var

    if len(extra) < dense_variable_nums:
        for var_name, var in ps_extra.items():
            if var_name not in extra:
                print(
                    "加载的优化器参数中未找到 {}, 使用 ps 初始化的值".format(var_name))
                extra[var_name] = var

    assert len(weight) == dense_variable_nums
    assert len(extra) == dense_variable_nums
    print("加载的参数：weight={}, extra={}".format(weight.keys(), extra.keys()))
    return weight, extra


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

    slide_pid_emb = kai.nn.get_dense_fea("pid_emb", dim=64, dtype=tf.float32)
    slide_uid_emb = kai.nn.get_dense_fea("uid_emb", dim=64, dtype=tf.float32)


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

    slide_pid_emb = kai.nn.get_dense_fea("pid_emb", dim=64, dtype=tf.float32)
    slide_uid_emb = kai.nn.get_dense_fea("uid_emb", dim=64, dtype=tf.float32)


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
field_input = tf.concat([slide_pid_emb, slide_uid_emb, user_embedding, comment_id_embedding, comment_info_embedding, position_embedding, pid_embedding, aid_embedding, uid_embedding, did_embedding, context_embedding, comment_genre_embedding, comment_length_embedding], -1)
input_wo_pos = tf.concat([slide_pid_emb, slide_uid_emb, user_embedding, comment_id_embedding, comment_info_embedding, pid_embedding, aid_embedding, uid_embedding, did_embedding, context_embedding, comment_genre_embedding, comment_length_embedding], -1)
input_did_pos = tf.concat([did_embedding, position_embedding], -1)
expand_xtr = simple_dense_network("expand_xtr", field_input, [256, 128, 64, 1], 0.3, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
like_first_xtr = simple_dense_network("like_first_xtr", field_input, [256, 128, 64, 1], 0.3, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
like_second_xtr = simple_dense_network("like_second_xtr", field_input, [256, 128, 64, 1], 0.3, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
reply_first_xtr = simple_dense_network("reply_task_first_xtr", field_input, [256, 128, 64, 1], 0.3, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
reply_second_xtr = simple_dense_network("reply_task_second_xtr", field_input, [256, 128, 64, 1], 0.3, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
copy_xtr = simple_dense_network("copy_xtr", field_input, [256, 128, 64, 1], 0.3, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
share_xtr = simple_dense_network("share_xtr", field_input, [256, 128, 64, 1], 0.3, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
audience_xtr = simple_dense_network("audience_xtr", field_input, [256, 128, 64, 1], 0.3, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
continuous_expand_xtr = simple_dense_network("continuous_expand_xtr", field_input, [256, 128, 64, 1], 0.3, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
hate_xtr = simple_dense_network("hate_xtr", field_input, [256, 128, 64, 1], 0.3, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
sub_at_xtr = simple_dense_network("sub_at_xtr", field_input, [256, 128, 64, 1], 0.3, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
duration_predict = simple_dense_network("duration_predict", input_wo_pos, [256, 128, 64, 1], 0.3, act=tf.nn.leaky_relu, last_act=tf.nn.relu)
duration_pos_bias_predict = simple_dense_network("duration_pos_bias_predict", input_did_pos, [128, 64, 1], 0.3, act=tf.nn.leaky_relu, last_act=tf.nn.relu)

if args.mode == 'train' or args.mode == 'eval':
    duration_predict = duration_predict + duration_pos_bias_predict

    # define label input and define metrics
    sample_weight = kai.nn.get_dense_fea("sample_weight", dim=1, dtype=tf.float32)
    ones = tf.ones_like(sample_weight, dtype=tf.float32)
    zeros = tf.zeros_like(sample_weight, dtype=tf.float32)

    expandAction_first = kai.nn.get_dense_fea("expandAction_first", dim=1, dtype=tf.float32)
    expand_label = tf.where(expandAction_first > 0, ones, zeros)
    continuous_expand_label = tf.where(expandAction_first > 1, ones, zeros)

    likeAction_first = kai.nn.get_dense_fea("likeAction_first", dim=1, dtype=tf.float32)
    likeAction_second = kai.nn.get_dense_fea("likeAction_second", dim=1, dtype=tf.float32)
    cancelLikeAction_first = kai.nn.get_dense_fea("cancelLikeAction_first", dim=1, dtype=tf.float32)
    cancelLikeAction_second = kai.nn.get_dense_fea("cancelLikeAction_second", dim=1, dtype=tf.float32)
    # like_label = tf.where((likeAction_first - cancelLikeAction_first > 0) | (likeAction_second > 0), ones, zeros)
    like_first_label = tf.where(likeAction_first - cancelLikeAction_first > 0, ones, zeros)
    like_second_label = tf.where(likeAction_second - cancelLikeAction_second > 0, ones, zeros)

    hateAction_first = kai.nn.get_dense_fea("hateAction_first", dim=1, dtype=tf.float32)
    # hateAction_second = kai.nn.get_dense_fea("hateAction_second", dim=1, dtype=tf.float32)
    cancelHateAction_first = kai.nn.get_dense_fea("cancelHateAction_first", dim=1, dtype=tf.float32)
    # cancelHateAction_second = kai.nn.get_dense_fea("cancelHateAction_second", dim=1, dtype=tf.float32)
    hate_label = tf.where(hateAction_first - cancelHateAction_first > 0, ones, zeros)

    # replyAction_first = kai.nn.get_dense_fea("replyAction_first", dim=1, dtype=tf.float32)
    # replyAction_second = kai.nn.get_dense_fea("replyAction_second", dim=1, dtype=tf.float32)
    # reply_label = tf.where((replyAction_first > 0) | (replyAction_second > 0), ones, zeros)
    # reply_first_label = tf.where(replyAction_first > 0, ones, zeros)
    # reply_second_label = tf.where(replyAction_second > 0, ones, zeros)

    replyTask_first = kai.nn.get_dense_fea("replyTaskAction_first", dim=1, dtype=tf.float32)
    replyTask_second = kai.nn.get_dense_fea("replyTaskAction_second", dim=1, dtype=tf.float32)
    # reply_label = tf.where((replyTask_first > 0) | (replyTask_second > 0), ones, zeros)
    reply_first_label = tf.where(replyTask_first > 0, ones, zeros)
    reply_second_label = tf.where(replyTask_second > 0, ones, zeros)

    copy_first_label = kai.nn.get_dense_fea("copyAction_first", dim=1, dtype=tf.float32)
    copy_second_label = kai.nn.get_dense_fea("copyAction_second", dim=1, dtype=tf.float32)
    copy_label = tf.where((copy_first_label > 0) | (copy_second_label > 0), ones, zeros)

    share_first_label = kai.nn.get_dense_fea("shareAction_first", dim=1, dtype=tf.float32)
    share_second_label = kai.nn.get_dense_fea("shareAction_second", dim=1, dtype=tf.float32)
    share_label = tf.where((share_first_label > 0) | (share_second_label > 0), ones, zeros)

    audience_first_label = kai.nn.get_dense_fea("audienceAction_first", dim=1, dtype=tf.float32)
    audience_second_label = kai.nn.get_dense_fea("audienceAction_second", dim=1, dtype=tf.float32)
    audience_label = tf.where((audience_first_label > 0) | (audience_second_label > 0), ones, zeros)

    subAtAction_first = kai.nn.get_dense_fea("subAtAction_first", dim=1, dtype=tf.float32)
    subAtAction_second = kai.nn.get_dense_fea("subAtAction_second", dim=1, dtype=tf.float32)
    sub_at_label = tf.where((subAtAction_first > 0) | (subAtAction_second > 0), ones, zeros)

    duration_label = kai.nn.get_dense_fea("stayDurationMs", dim=1, dtype=tf.float32)
    duration_label = tf.clip_by_value(duration_label / 1000, 0, 60)

    targets = [
        ('expand_predict', expand_xtr, expand_label, ones, "auc"),
        ('like_first_predict', like_first_xtr, like_first_label, ones, "auc"),
        ('like_second_predict', like_second_xtr, like_second_label, ones, "auc"),
        ('reply_first_predict', reply_first_xtr, reply_first_label, ones, "auc"),
        ('reply_second_predict', reply_second_xtr, reply_second_label, ones, "auc"),
        ('copy_predict', copy_xtr, copy_label, ones, "auc"),
        ('share_predict', share_xtr, share_label, ones, "auc"),
        ('audience_predict', audience_xtr, audience_label, ones, "auc"),
        ('continuous_expand_predict', continuous_expand_xtr, continuous_expand_label, ones, "auc"),
        ('hate_predict', hate_xtr, hate_label, ones, "auc"),
        ('sub_at_predict', sub_at_xtr, sub_at_label, ones, "auc"),
    ]

    metric_name, preds, labels, weights, metric_type = zip(*targets)

    # 5. define optimizer
    loss = tf.losses.log_loss(labels, preds, weights, reduction="weighted_sum")
    loss_duration = tf.losses.huber_loss(duration_label, duration_predict, weights=1.0, delta=3.0)
    loss = loss + loss_duration
    optimizer = kai.nn.optimizer.AdamW(1e-3)
    optimizer.minimize(loss)

    recall_type = kai.nn.get_dense_fea("recall_type", dim=1, dtype=tf.float32)
    # ones = tf.ones_like(expand_label, dtype=tf.float32)
    # zeros = tf.zeros_like(expand_label, dtype=tf.float32)

    # comment_genre = tf.cast(kai.nn.get_dense_fea("comment_genre", dim=1, dtype=tf.int64), tf.float32)
    # pic_comment = tf.where(comment_genre > 0, ones, zeros)


    eval_targets = [
        ('expand_predict', expand_xtr, expand_label, ones, "auc"),
        ('like_first_predict', like_first_xtr, like_first_label, ones, "auc"),
        ('like_second_predict', like_second_xtr, like_second_label, ones, "auc"),
        ('reply_first_predict', reply_first_xtr, reply_first_label, ones, "auc"),
        ('reply_second_predict', reply_second_xtr, reply_second_label, ones, "auc"),
        ('copy_predict', copy_xtr, copy_label, ones, "auc"),
        ('share_predict', share_xtr, share_label, ones, "auc"),
        ('audience_predict', audience_xtr, audience_label, ones, "auc"),
        ('continuous_expand_predict', continuous_expand_xtr, continuous_expand_label, ones, "auc"),
        ('hate_predict', hate_xtr, hate_label, ones, "auc"),
        ('sub_at_predict', sub_at_xtr, sub_at_label, ones, "auc"),
        ('duration_predict', duration_predict, duration_label, ones, 'linear_regression'),

        # ('expand_hot', expand_xtr, expand_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        # ('like_first_hot', like_first_xtr, like_first_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        # ('like_second_hot', like_second_xtr, like_second_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        # ('reply_first_hot', reply_first_xtr, reply_first_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        # ('reply_second_hot', reply_second_xtr, reply_second_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        # ('copy_hot', copy_xtr, copy_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        # ('share_hot', share_xtr, share_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        # ('audience_hot', audience_xtr, audience_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        # ('continuous_expand_hot', continuous_expand_xtr, continuous_expand_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        # ('hate_hot', hate_xtr, hate_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        # ('sub_at_hot', sub_at_xtr, sub_at_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        # ('duration_hot', duration_predict, duration_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "linear_regression"),

        # ('expand_climb', expand_xtr, expand_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        # ('like_first_climb', like_first_xtr, like_first_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        # ('like_second_climb', like_second_xtr, like_second_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        # ('reply_first_climb', reply_first_xtr, reply_first_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        # ('reply_second_climb', reply_second_xtr, reply_second_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        # ('copy_climb', copy_xtr, copy_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        # ('share_climb', share_xtr, share_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        # ('audience_climb', audience_xtr, audience_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        # ('continuous_expand_climb', continuous_expand_xtr, continuous_expand_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        # ('hate_climb', hate_xtr, hate_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        # ('sub_at_climb', sub_at_xtr, sub_at_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        # ('duration_climb', duration_predict, duration_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
    ]

    # 6. finish define model structure 
    kai.build_model(optimizer=[optimizer], metrics=eval_targets)
else:
    targets = [
        ("expand_xtr", expand_xtr),
        ("like_first_xtr", like_first_xtr),
        ("like_second_xtr", like_second_xtr),
        ("reply_first_xtr", reply_first_xtr),
        ("reply_second_xtr", reply_second_xtr),
        ("copy_xtr", copy_xtr),
        ("share_xtr", share_xtr),
        ("audience_xtr", audience_xtr),
        ("continuous_expand_xtr", continuous_expand_xtr),
        ("hate_xtr", hate_xtr),
        # ("like_xtr", 1-(1-like_first_xtr)*(1-like_second_xtr)),
        # ("reply_xtr", 1-(1-reply_first_xtr)*(1-reply_second_xtr)),
        ("sub_at_xtr", sub_at_xtr),
        ("duration_predict", duration_predict),
    ]
    q_names, preds = zip(*targets)
    config.dump_predict_config('./predict/config', targets, input_type=3, extra_preds=q_names)
