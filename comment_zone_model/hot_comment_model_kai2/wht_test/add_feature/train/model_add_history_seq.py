""" 在v2基础上 增加 内容 特征，不修正label.
"""

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'predict'], default='train')
parser.add_argument('--dryrun', dest="dryrun", const=True, default=False, nargs='?')
parser.add_argument('--with_kai', action="store_true")
args = parser.parse_args()

SEQ_LEN = 20

# 1. define sparse input
if args.mode == 'train':
    import tensorflow.compat.v1 as tf
    import kai.tensorflow as kai

    ## common attr
    user_embedding = kai.nn.new_embedding("user_embedding", dim=4, slots=[101, 102])
    pid_emb = kai.nn.new_embedding("pid_emb", dim=64, slots=[103])
    photo_aid_emb = kai.nn.new_embedding("photo_aid_emb", dim=64, slots=[104])
    uid_emb = kai.nn.new_embedding("uid_emb", dim=64, slots=[105])
    did_emb = kai.nn.new_embedding("did_emb", dim=64, slots=[106])

    user_continuous_cid_fea0 = kai.nn.new_embedding("user_continuous_cid_fea", dim=64, expand=SEQ_LEN, slots=[300])
    user_continuous_cid_fea = tf.reshape(user_continuous_cid_fea0, [-1, SEQ_LEN, 64])     
    padding_input = kai.nn.get_dense_fea("mask_pack", dim=SEQ_LEN, dtype=tf.float32)  # [-1, seq]

    ## item attr
    comment_id_embedding = kai.nn.new_embedding("c_id_embedding", dim=64, slots=[201])
    author_id_embedding = kai.nn.new_embedding("a_id_embedding", dim=64, slots=[202])
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

    ## common attr
    user_embedding = config.new_embedding("user_embedding", dim=4, slots=[101, 102], **compress_kwargs)
    pid_emb = config.new_embedding("pid_emb", dim=64, slots=[103], **compress_kwargs)
    photo_aid_emb = config.new_embedding("photo_aid_emb", dim=64, slots=[104], **compress_kwargs)
    uid_emb = config.new_embedding("uid_emb", dim=64, slots=[105], **compress_kwargs)
    did_emb = config.new_embedding("did_emb", dim=64, slots=[106], **compress_kwargs)

    user_continuous_cid_fea0 = config.new_embedding("user_continuous_cid_fea", dim=64, expand=SEQ_LEN, slots=[300], **compress_kwargs)
    user_continuous_cid_fea = tf.reshape(user_continuous_cid_fea0, [-1, SEQ_LEN, 64])     
    padding_input = config.get_dense_fea("mask_pack", dim=SEQ_LEN, dtype=tf.float32, **compress_kwargs)  # [-1, seq]

    ## item attr
    comment_id_embedding = config.new_embedding("c_id_embedding", dim=64, slots=[201])
    author_id_embedding = config.new_embedding("a_id_embedding", dim=64, slots=[202])
    comment_info_embedding = config.new_embedding("c_info_embedding", dim=32, slots=[203, 204, 205, 206, 207, 209])
    position_embedding = config.new_embedding("position_embedding", dim=8, slots=[208])

    # comment_content_segs0 = config.new_embedding("comment_content_segs", dim=64, expand=SEQ_LEN, slots=[300])    # [-1, 1280]  
    # comment_content_segs = tf.reshape(comment_content_segs0, [-1, SEQ_LEN, 64])     

    # seq_pos_embedding0 = config.new_embedding("seq_pos_embedding", dim=64, expand=SEQ_LEN, slots=[301])   # [-1, 1280] 
    # seq_pos_embedding = tf.reshape(seq_pos_embedding0, [-1, SEQ_LEN, 64])

    # cmt_token_emb_seq = comment_content_segs + seq_pos_embedding

    # padding_input = config.get_dense_fea("mask_pack", dim=SEQ_LEN, dtype=tf.float32)  # [-1, seq]



def attention(query, key, padding_mask):
    """
    计算注意力加权和
    :param query: 形状为 [-1, 1, h] 的张量
    :param key: 形状为 [-1, L, h] 的张量
    :param padding_mask: 形状为 [-1, L] 的张量，0表示padding
    :return: 形状为 [-1, h] 的注意力加权和结果
    """
    # 获取维度信息
    _, L, h = tf.shape(key)[0], tf.shape(key)[1], tf.shape(key)[2]
    
    # 计算注意力分数
    # 将 query 转换为 [-1, h, 1] 以便进行矩阵乘法
    query_reshaped = tf.transpose(query, [0, 2, 1])
    # 计算 query 和 key 的点积，结果形状为 [-1, L, 1]
    attention_scores = tf.matmul(key, query_reshaped)
    # 将形状调整为 [-1, L]
    attention_scores = tf.squeeze(attention_scores, axis=-1)
    
    # 应用 padding mask
    # 将 padding_mask 中的 0 转换为一个很大的负数（这里使用 -1e9）
    paddings = tf.ones_like(attention_scores) * (-1e9)
    attention_scores = tf.where(tf.equal(padding_mask, 0), paddings, attention_scores)
    
    # 应用 softmax 获取注意力权重
    attention_weights = tf.nn.softmax(attention_scores, axis=-1)
    
    # 计算注意力加权和
    # 将 attention_weights 的形状调整为 [-1, L, 1] 以便进行广播
    attention_weights = tf.expand_dims(attention_weights, axis=-1)
    # 进行加权和，结果形状为 [-1, L, h]
    weighted_sum = tf.multiply(key, attention_weights)
    # 沿着序列长度维度求和，结果形状为 [-1, h]
    result = tf.reduce_sum(weighted_sum, axis=1)
    
    return result


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


# attention_output = multi_head_attention("multi_head_attention", cmt_token_emb_seq, padding_mask=padding_input, num_units=64, num_heads=8)   # [-1, seq, dim]
# cls_output = attention_output[:, 0, :]  # [-1, dim]

# hadamard_uid = tf.multiply(cls_output, uid_emb)
# hadamard_cid = tf.multiply(cls_output, comment_id_embedding)
# inner_uid = tf.expand_dims(tf.reduce_sum(hadamard_uid, axis=1), 1)
# inner_cid = tf.expand_dims(tf.reduce_sum(hadamard_cid, axis=1), 1)

attention_output = attention(tf.expand_dims(comment_id_embedding, axis=1), user_continuous_cid_fea, padding_input)
    
# define model structure
used_feats = [user_embedding, pid_emb, photo_aid_emb, uid_emb, did_emb,
            comment_id_embedding, author_id_embedding, comment_info_embedding, position_embedding,
            attention_output
            ]
field_input = tf.concat(used_feats, -1)
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

    # comment_genre = kai.nn.get_dense_fea("comment_genre", dim=1, dtype=tf.float32)
    # pic_comment = tf.where(comment_genre > 0, ones, zeros)
    

    eval_targets = [
        ('expand_predict', expand_xtr, expand_label, ones, "auc"),
        ('like_predict', like_xtr, like_label, ones, "auc"),
        ('reply_predict', reply_xtr, reply_label, ones, "auc"),
        ('copy_predict', copy_xtr, copy_label, ones, "auc"),
        ('share_predict', share_xtr, share_label, ones, "auc"),
        ('audience_predict', audience_xtr, audience_label, ones, "auc"),
        ('continuous_expand_predict', continuous_expand_xtr, continuous_expand_label, ones, "auc"),

    #     ('expand_hot', expand_xtr, expand_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
    #     ('like_hot', like_xtr, like_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
    #     ('reply_hot', reply_xtr, reply_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
    #     ('copy_hot', copy_xtr, copy_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
    #     ('share_hot', share_xtr, share_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
    #     ('audience_hot', audience_xtr, audience_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
    #     ('continuous_expand_hot', continuous_expand_xtr, continuous_expand_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),

    #     ('expand_climb', expand_xtr, expand_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
    #     ('like_climb', like_xtr, like_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
    #     ('reply_climb', reply_xtr, reply_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
    #     ('copy_climb', copy_xtr, copy_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
    #     ('share_climb', share_xtr, share_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
    #     ('audience_climb', audience_xtr, audience_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
    #     ('continuous_expand_climb', continuous_expand_xtr, continuous_expand_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
    ]

    # class TensorPrintHook(kai.training.RunHookBase):
    #     def __init__(self, debug_tensor_map):
    #         self.has_print = False
    #         self.debug_tensor_map = debug_tensor_map

    #     def begin(self, stream_context):
    #         pass

    #     def before_pass_run(self, pass_run_context):
    #         """
    #         每个 pass 只会 print 一次
    #         """
    #         self.has_print = False

    #         total=0
    #         for i, feat in enumerate(self.debug_tensor_map['used_features']):
    #             # print(f'feat_{i}: dim={feat.shape[-1]}')
    #             total += feat.shape[-1]
    #         print(f'total size={total}')

    #         for i, feat in enumerate(self.debug_tensor_map['debug_features']):
    #             print(f'debug_features_{i}: dim={feat.shape}')

    #     def before_step_run(self, step_run_context):
    #         return kai.training.StepRunArgs(fetches=self.debug_tensor_map)

    #     def after_step_run(self, step_run_context, step_run_values):
    #         if not self.has_print:
    #             for k, v in step_run_values.result.items():
    #                 if k == 'print_features':
    #                     print(v)
    #             self.has_print = True
            
    # debug_tensor = {
    #     'used_features': used_feats,
    #     'debug_features': [comment_content_segs0, comment_content_segs, seq_pos_embedding0, seq_pos_embedding, padding_input],
    #     # 'print_features': [comment_content_segs[:3, :6, :2], seq_pos_embedding[:3,:6,:2]]
    # }
    # kai.add_run_hook(TensorPrintHook(debug_tensor), "debug_tensor_hook")

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

