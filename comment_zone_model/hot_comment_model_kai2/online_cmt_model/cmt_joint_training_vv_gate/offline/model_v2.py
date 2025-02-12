""" 在v2基础上，增加单评论停留时长信号。并且每个tower的输入都不包含曝光位置特征！仅结构没有用ple！
"""
import os
import argparse
import tensorflow as tf
import bert.modeling as modeling
import bert.tokenization_zh as tokenization_zh
import numpy as np
# from tqdm import tqdm
import json

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'predict', 'eval'], default='train')
parser.add_argument('--dryrun', dest="dryrun", const=True, default=False, nargs='?')
parser.add_argument('--with_kai', action="store_true")
args = parser.parse_args()

print("args.mode", "\t", args.mode)
# 1. define sparse input
if args.mode == 'train' or args.mode == 'eval':
    import tensorflow.compat.v1 as tf
    import kai.tensorflow as kai

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
    
    token_input_ids = kai.nn.get_dense_fea("token_input_ids", dim=27, dtype=tf.float32)
    token_input_mask = kai.nn.get_dense_fea("token_input_mask", dim=27, dtype=tf.float32)
    token_sep_ids = kai.nn.get_dense_fea("token_sep_ids", dim=27, dtype=tf.float32)

    bert_slot_emb =  kai.nn.new_embedding("bert_id_embedding", dim=256, slots=[252])
    gate_emb = kai.nn.new_embedding("gate_embedding", dim=32, slots=[260])

    bert_config_file = "./bert/Robert256/bert_config.json"
    
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
    
    batch_size = tf.shape(comment_id_embedding)[0]
    token_input_ids = token_input_mask = token_sep_ids = tf.ones([batch_size, 27], dtype=tf.float32)
    # token_input_ids = config.get_extra_param(name="token_input_ids", size=27)
    # token_input_mask = config.get_extra_param(name="token_input_mask", size=27)
    # token_sep_ids = config.get_extra_param(name="token_sep_ids", size=27)

    bert_slot_emb =  config.new_embedding("bert_id_embedding", dim=256, slots=[252])
    gate_emb = kai.nn.new_embedding("gate_embedding", dim=32, slots=[260])

    bert_config_file = "./train/bert/Robert256/bert_config.json"


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
            # output = tf.layers.dense(output, unit, activation=act,
            #                       kernel_initializer=tf.glorot_uniform_initializer(), name = f"dense_{i}")
        return output

def get_bert_emb(token_input_ids, token_input_mask, token_sep_ids):
    token_input_ids  = tf.cast(token_input_ids, dtype= tf.int32)
    token_input_mask  = tf.cast(token_input_mask, dtype= tf.int32)
    token_sep_ids  = tf.cast(token_sep_ids, dtype= tf.int32)

    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    model = modeling.BertModel(
        config=bert_config,
        is_training=True,    
        input_ids=token_input_ids,  
        input_mask=token_input_mask, 
        token_type_ids=token_sep_ids, 
        use_one_hot_embeddings=False,
        is_stop_gradient =True,
        stop_layer_idx = 4
    )
    hidden_states = model.get_sequence_output() # 输出的embedding
    print("hidden_states shape", hidden_states.shape)
    bert_output_emb= hidden_states[:,0,:]
    print("bert_output_emb shape", bert_output_emb.shape)
    return bert_output_emb

if args.mode == 'train' or args.mode == 'eval':
    bert_output_emb = get_bert_emb(token_input_ids, token_input_mask, token_sep_ids)
    # config.custom_gradients[bert_slot_emb] = bert_output_emb
    # config.custom_opt[bert_slot_emb] = {
    #                             "opt_type": "AssignAdd", 
    #                             "decay_rate": 0.0,
    #                             "add_rate": 1.0
    #                             }
else:
    bert_output_emb = tf.cond(
        tf.constant(False),
        lambda: get_bert_emb(token_input_ids, token_input_mask, token_sep_ids),
        lambda: bert_slot_emb
    )


#------define model structure
# input_wo_pos = tf.concat([user_embedding, comment_id_embedding, comment_info_embedding, pid_embedding, aid_embedding, uid_embedding, did_embedding, context_embedding, comment_genre_embedding, comment_length_embedding], -1)
cid_inpput =  simple_dense_network("cid_input", comment_id_embedding, [128], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.leaky_relu)
# action_input = simple_dense_network("action_input", input_wo_pos, [256, 128], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.leaky_relu)
content_input = simple_dense_network("content_input", bert_output_emb, [128], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.leaky_relu)
gate_input = simple_dense_network("gate_input", gate_emb, [1], 0.0, act=tf.nn.sigmoid, last_act=tf.nn.sigmoid)
fuse_input = gate_input * cid_inpput + (1 - gate_input) * content_input
input_wo_pos = tf.concat([user_embedding, fuse_input, comment_info_embedding, pid_embedding, aid_embedding, uid_embedding, did_embedding, context_embedding, comment_genre_embedding, comment_length_embedding], -1)

expand_xtr = simple_dense_network("expand_xtr", fuse_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
like_xtr = simple_dense_network("like_xtr", fuse_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
reply_xtr = simple_dense_network("reply_xtr", fuse_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
copy_xtr = simple_dense_network("copy_xtr", fuse_input, [256, 128,64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
share_xtr = simple_dense_network("share_xtr", fuse_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
audience_xtr = simple_dense_network("audience_xtr", fuse_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
continuous_expand_xtr = simple_dense_network("continuous_expand_xtr", fuse_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
hate_xtr = simple_dense_network("hate_xtr", fuse_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
# report_xtr = simple_dense_network("report_xtr", input_wo_pos, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)

# duration
duration_predict = simple_dense_network("duration_predict", fuse_input, [64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.relu)
input_did_pos = tf.concat([did_embedding, position_embedding, context_embedding], -1)
duration_pos_bias_predict = simple_dense_network("duration_pos_bias_predict", input_did_pos, [128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.relu)

if args.mode == 'train' or args.mode == 'eval':
    duration_predict = duration_predict + duration_pos_bias_predict

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

    duration_label = kai.nn.get_dense_fea("stayDurationMs", dim=1, dtype=tf.float32)
    duration_label = tf.clip_by_value(duration_label / 1000, 0, 60)

    hate_la = kai.nn.get_dense_fea("hate_label", dim=1, dtype=tf.float32)
    hate_label = tf.where(hate_la > 0.0, ones, zeros)

    # report_first_label = kai.nn.get_dense_fea("reportAction_first", dim=1, dtype=tf.float32)
    # report_second_label = kai.nn.get_dense_fea("reportAction_second", dim=1, dtype=tf.float32)
    # report_label = tf.where((report_first_label > 0.0) | (report_second_label > 0.0), ones, zeros)




    targets = [
        ('expand_predict', expand_xtr, expand_label, ones, "auc"),
        ('like_predict', like_xtr, like_label, ones, "auc"),
        ('reply_predict', reply_xtr, reply_label, ones, "auc"),
        ('copy_predict', copy_xtr, copy_label, ones, "auc"),
        ('share_predict', share_xtr, share_label, ones, "auc"),
        ('audience_predict', audience_xtr, audience_label, ones, "auc"),
        ('continuous_expand_predict', continuous_expand_xtr, continuous_expand_label, ones, "auc"),
        ('hate_predict', hate_xtr, hate_label, ones, "auc"),
        # ('report_predict', report_xtr, report_label, ones, "auc"),
    ]

    metric_name, preds, labels, weights, metric_type = zip(*targets)

    # 5. define optimizer
    loss = tf.losses.log_loss(labels, preds, weights, reduction="weighted_sum")
    loss_duration = tf.losses.huber_loss(duration_label, duration_predict, weights=1.0, delta=3.0)
    loss = loss + loss_duration
    # ------只更新bert最后n层
    # update_var_list = []
    # un_update_var_list = []
    # total_sparse_var = kai.get_sparse_trainable_variables(return_dict=True)
    # for sparse_var_name, sparse_var_tensor in total_sparse_var.items():
    #     update_var_list.append(sparse_var_tensor)
    
    # total_dense_var = kai.get_dense_trainable_variables()
    # for var in total_dense_var:
    #     if "bert" not in var.name:
    #         update_var_list.append(var)
    #     else:
    #         if "layer_5" in var.name:
    #            update_var_list.append(var) 
    #         else:
    #            un_update_var_list.append(var) 

    
    # optimizer = kai.nn.optimizer.Adam(1e-3)
    # optimizer.minimize(loss, var_list=update_var_list)
    # un_optimizer = kai.nn.optimizer.Adam(0.0)
    # un_optimizer.minimize(loss, var_list=un_update_var_list)

    # ---- 全部更新
    optimizer_all = kai.nn.optimizer.Adam(1e-3)
    optimizer_all.minimize(loss, var_list  = kai.get_dense_trainable_variables() + list(set(kai.get_sparse_trainable_variables()) - set([bert_slot_emb])))
    custom_grad_dict = {}
    custom_grad_dict[bert_slot_emb.name] = bert_output_emb
    assign_add_opt = kai.optimizer.AssignAddOptimizer(decay_rate=0, add_rate=1)
    assign_add_opt.minimize(loss, var_list=[bert_slot_emb], custom_gradient=custom_grad_dict)

    eval_targets = [
        ('expand_predict', expand_xtr, expand_label, ones, "auc"),
        ('like_predict', like_xtr, like_label, ones, "auc"),
        ('reply_predict', reply_xtr, reply_label, ones, "auc"),
        ('copy_predict', copy_xtr, copy_label, ones, "auc"),
        ('share_predict', share_xtr, share_label, ones, "auc"),
        ('audience_predict', audience_xtr, audience_label, ones, "auc"),
        ('continuous_expand_predict', continuous_expand_xtr, continuous_expand_label, ones, "auc"),
        ('duration_predict', duration_predict, duration_label, ones, 'linear_regression'),
        ('hate_predict', hate_xtr, hate_label, ones, "auc"),
        # ('report_predict', report_xtr, report_label, ones, "auc"),

        ('expand_hot', expand_xtr, expand_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('like_hot', like_xtr, like_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('reply_hot', reply_xtr, reply_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('copy_hot', copy_xtr, copy_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('share_hot', share_xtr, share_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('audience_hot', audience_xtr, audience_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('continuous_expand_hot', continuous_expand_xtr, continuous_expand_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('duration_hot', duration_predict, duration_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), 'linear_regression'),
        ('hate_hot', hate_xtr, hate_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),

        ('expand_climb', expand_xtr, expand_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('like_climb', like_xtr, like_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('reply_climb', reply_xtr, reply_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('copy_climb', copy_xtr, copy_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('share_climb', share_xtr, share_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('audience_climb', audience_xtr, audience_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('continuous_expand_climb', continuous_expand_xtr, continuous_expand_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('duration_climb', duration_predict, duration_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), 'linear_regression'),
        ('hate_climb', hate_xtr, hate_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
    ]

    # 6. finish define model structure 
    kai.build_model(optimizer=[optimizer_all, assign_add_opt], metrics=eval_targets)
    # kai.build_model(optimizer=[optimizer, un_optimizer], metrics=eval_targets)
else:
    targets = [
        ("expand_xtr", expand_xtr),
        ("like_xtr", like_xtr),
        ("reply_xtr", reply_xtr),
        ("copy_xtr", copy_xtr),
        ("share_xtr", share_xtr),
        ("audience_xtr", audience_xtr),
        ("continuous_expand_xtr", continuous_expand_xtr),
        ("duration_predict", duration_predict),
        ('hate_xtr', hate_xtr),
        # ('report_xtr', report_xtr),
    ]
    q_names, preds = zip(*targets)
    config.dump_predict_config('./predict/config', targets, input_type=3, extra_preds=q_names)
