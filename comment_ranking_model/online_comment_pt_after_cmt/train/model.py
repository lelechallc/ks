from __future__ import print_function
MODEL_TRANS_ORIGIN='cpp'
from cProfile import label

import os
import logging
import argparse
import sys
import functools
import numpy as np
import math

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'predict', 'user_predict', 'photo_predict'], dest='mode', default='train')
parser.add_argument('--dryrun', dest='dryrun', const=True, default=False, nargs='?')
parser.add_argument('--with_kai', default=False)
parser.add_argument('--text', default=False)
parser.add_argument('--with_kai_v2', default=True)
args = parser.parse_known_args()[0]

if args.mode == 'train':
    import tensorflow.compat.v1 as tf
    import kai.tensorflow as config
    default_param_attr = config.nn.ParamAttr(initializer=config.nn.UniformInitializer(0.0001),
                                   access_method=config.nn.ProbabilityAccess(100.0),
                                   recycle_method=config.nn.UnseendaysRecycle(delete_after_unseen_days=7, delete_threshold=2.0, allow_dynamic_delete=True))
    config.nn.set_default_param_attr(default_param_attr)

    compress_kwargs = dict(common=True)

elif args.mode == 'predict':
    import tensorflow as tf
    if not args.dryrun and not args.with_kai:
        # monkey patch
        import mio_tensorflow.patch as mio_tensorflow_patch
        mio_tensorflow_patch.apply()
    from mio_tensorflow.config import MioConfig
    base_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), './base.yaml')
    config = MioConfig.from_base_yaml(base_config, clear_embeddings=True, clear_params=True,
                                    dryrun=args.dryrun, label_with_kv=True, grad_no_scale=False,
                                    with_kai=args.with_kai)  
    compress_kwargs = dict(compress_group="USER")
    
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

def multi_head_attention(name, queries, keys, num_units=None, num_heads=8, add_embedding=None):
    """
    queries: [-1, q_seq, dim], q_seq = 1 or k_seq
    keys: [-1, k_seq, dim]
    [2048, 1, 50, 48]
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]
        q_seq = tf.shape(queries)[1]  
        k_seq = tf.shape(keys)[1]
        
        Q = tf.layers.dense(queries, num_units) # [-1, q_seq, dim]     infer: [1, 2048, 48] 
        K = tf.layers.dense(keys, num_units)  # [-1, k_seq, dim]  # infer: [1, 50, 48]
        V = tf.layers.dense(keys, num_units)  # [-1, k_seq, dim]

        assert num_units % num_heads == 0
        depth = num_units // num_heads
        Q_ = tf.transpose(tf.reshape(Q, [-1, q_seq, num_heads, depth]), perm=[0, 2, 1, 3])  # [1, num_head, 2048, depth]
        K_ = tf.transpose(tf.reshape(K, [-1, k_seq, num_heads, depth]), perm=[0, 2, 1, 3])   #[1, num_head, 50, depth]
        V_ = tf.transpose(tf.reshape(V, [-1, k_seq, num_heads, depth]), perm=[0, 2, 1, 3])   # [1, num_heads, 50, depth]

        # [-1, num_heads, q_seq, k_seq]
        outputs = tf.matmul(Q_, K_, transpose_b=True) / tf.math.sqrt(tf.cast(depth, tf.float32)) # infer:[1, num_head, 2048, 50]
        outputs = tf.nn.softmax(outputs, axis=-1)
        attention = tf.transpose(tf.matmul(outputs, V_), perm=[0, 2, 1, 3])  # [-1, q_seq, num_heads, depth]  infer: [1, 2048, num_head, depth]
        attention = tf.reshape(attention, [-1, q_seq, num_units])  # [-1, q_seq, num_units]

        # add and norm
        # attention = norm(name=f"{name}_norm", x=attention + K)
        # if add_embedding is not None:
        #     attention = norm(name=f"{name}_norm", x=attention + tf.layers.dense(add_embedding, num_units))
        # else:
        #     #attention = norm(name=f"{name}_norm", x=attention + K)
        #     attention = norm(name=f"{name}_norm", x=attention)
    return attention

def tree_model_net(name, inputs, class_num, dropout, is_training):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        fc = tf.layers.dense(inputs, 64)
        fc = tf.nn.relu(fc)
        fc = tf.layers.dropout(fc, rate=dropout, training=is_training)
        fc = tf.layers.dense(fc, 32)
        fc = tf.nn.relu(fc)

        logits = tf.layers.dense(fc, class_num)
        res = tf.nn.sigmoid(logits)
    return res

def get_encoded_consumption_depth(label_encoding_predict, tree_num_intervals, begins, ends, name="get_encoded_consumption_depth"):
        height = int(math.log2(tree_num_intervals))
        encoded_prob_list = []
        temp_encoded_consumption_depth = (begins + ends) / 2.0  # bsx32
        encoded_consumption_depth = temp_encoded_consumption_depth
        
        for i in range(tree_num_intervals):
            temp = 0.0
            cur_code = 2**height - 1 + i
            for j in range(1, 1+height):
                classifier_branch = cur_code % 2
                classifier_idx = int((cur_code - 1) / 2)
                # update cur_code
                cur_code = classifier_idx
                if classifier_branch == 1:
                    temp += tf.log(1.0 - label_encoding_predict[:,classifier_idx]+ 0.00001)
                else:
                    temp += tf.log(label_encoding_predict[:,classifier_idx]+0.00001)
            encoded_prob_list.append(temp)

        encoded_prob = tf.exp(tf.stack(encoded_prob_list,axis=1))  # bs*tree_num_intervals
        encoded_consumption_depth = tf.reduce_sum(encoded_consumption_depth*encoded_prob,axis=-1,keepdims=True)

        e_x2 = tf.reduce_sum(tf.square(encoded_consumption_depth)*encoded_prob, axis=-1,keepdims=True)
        square_of_e_x = tf.square(encoded_consumption_depth)
        var = tf.sqrt(e_x2 - square_of_e_x)
        return encoded_consumption_depth, tf.reduce_sum(var)  

# photo features
  
photo_id_feature = [26]
author_id_feature = [128, 27]
photo_xtr_feature = [576, 577, 578, 579, 567, 71, 142]
photo_stat_feature = [110, 185, 685, 686, 673, 1118, 141]
photo_hetu_feature = [682, 683]
photo_cnt_feature = [786, 787]
photo_pxtr_feature = [(1001 + i) for i in range(29)]
photo_play_feature = [146, 147, 288, 418]
photo_bias_feaature = [498, 143, 603, 3621]

pid_emb = config.new_embedding("pid_emb", dim=32, slots=photo_id_feature)
aid_emb = config.new_embedding("aid_emb", dim=32, slots=author_id_feature)
pid_xtr = config.new_embedding("pid_xtr", dim=8, slots=photo_xtr_feature) # 9
pid_stat = config.new_embedding("pid_stat", dim=8, slots=photo_stat_feature)
pid_hetu = config.new_embedding("pid_hetu", dim=8, slots=photo_hetu_feature)
pid_cnt = config.new_embedding("pid_cnt", dim=8, slots=photo_cnt_feature)
pid_pxtr = config.new_embedding("pid_pxtr", dim=8, slots=photo_pxtr_feature)
top_bias = config.new_embedding("top_bias", dim=8, slots=photo_bias_feaature)
pid_play_f = config.new_embedding("pid_play_f", dim=8, slots=photo_play_feature)

photo_feature = [pid_emb, aid_emb, pid_xtr, pid_stat, pid_hetu, pid_cnt, top_bias, pid_pxtr, pid_play_f]  # [batchsize, dim]

photo_feature = tf.concat(photo_feature, axis=1)


user_id_feature = [38, 34]
user_stat_feature = [184, 35, 189]
device_stat_feature = [711, 712, 713, 714, 715, 716]

user_liveness_feature = [676, 677, 678, 679, 680, 681] # ExtractSignUserShowCountDay,ExtractSignUserShowCountHour,ExtractSignUserShowCountHalfHour,ExtractSignUserShowCountTenMinutes,ExtractSignUserShowCountFiveMinute,ExtractSignUserShowCountMinute
user_loc_feature = [182, 603] #ExtractSignUserProvCity ExtractSignTabInfo
user_view_id_feature = [290, 291] # ExtractSignUserHhEffectiveView,ExtractSignUserHhLongView

uid_emb = config.new_embedding("uid_emb", dim=32, slots=user_id_feature, **compress_kwargs)
uid_stat = config.new_embedding("uid_stat", dim=8, slots=user_stat_feature, **compress_kwargs)
did_stat = config.new_embedding("did_stat", dim=8, slots=device_stat_feature, **compress_kwargs)
uid_live_f = config.new_embedding("uid_live_f", dim=8, slots=user_liveness_feature, **compress_kwargs)
uid_loc_f = config.new_embedding("uid_loc_f", dim=8, slots=user_loc_feature, **compress_kwargs)
uid_viewid_f = config.new_embedding("uid_viewid_f", dim=32, slots=user_view_id_feature, **compress_kwargs)

short_term_tags = config.new_embedding("realshow_tags", dim=8, expand = 50, slots=[249], **compress_kwargs) # ExtractSignRecentRealshowTags
short_term_pids = config.new_embedding("short_term_pids", dim=32, expand=50, slots=[246], **compress_kwargs) # ExtractSignRecentRealshowPids
short_term_aids = config.new_embedding("short_term_aids", dim=32, expand=50, slots=[247], **compress_kwargs) # ExtractSignRecentRealshowAids
short_term_times = config.new_embedding("short_term_times", dim=8, expand=50, slots=[250], **compress_kwargs) # ExtractSignRecentRealshowTimestamps
realshow_pids = tf.reshape(short_term_pids, (-1, 50, 32))
realshow_aids = tf.reshape(short_term_aids, (-1, 50, 32))
realshow_tags = tf.reshape(short_term_tags, (-1, 50, 8))
realshow_times = tf.reshape(short_term_times, (-1, 50, 8))
user_realshow_list_input = tf.concat([realshow_pids, realshow_aids, realshow_tags, realshow_times], 2)
user_realshow_list_input_sum = tf.math.reduce_sum(user_realshow_list_input, axis = 1)

uid_like_id_seq = config.new_embedding('uid_like_list_id', dim=32, expand=50, slots=[904], **compress_kwargs)
uid_like_id_seq = tf.reshape(uid_like_id_seq, [-1, 50, 32])
uid_like_hetu1_seq = config.new_embedding('uid_like_list_hetu1', dim=8, expand=50, slots=[905], **compress_kwargs)
uid_like_hetu1_seq = tf.reshape(uid_like_hetu1_seq, [-1, 50, 8])
uid_like_hetu2_seq = config.new_embedding('uid_like_list_hetu2', dim=8, expand=50, slots=[906], **compress_kwargs)
uid_like_hetu2_seq = tf.reshape(uid_like_hetu2_seq, [-1, 50, 8])
uid_like_seq = tf.concat([uid_like_id_seq, uid_like_hetu1_seq, uid_like_hetu2_seq], axis=2) # [batchsize, 50, 48]

uid_follow_id_seq = config.new_embedding('uid_follow_list_id', dim=32, expand=50, slots=[907], **compress_kwargs)
uid_follow_id_seq = tf.reshape(uid_follow_id_seq, [-1, 50, 32])
uid_follow_hetu1_seq = config.new_embedding('uid_follow_list_hetu1', dim=8, expand=50, slots=[908], **compress_kwargs)
uid_follow_hetu1_seq = tf.reshape(uid_follow_hetu1_seq, [-1, 50, 8])
uid_follow_hetu2_seq = config.new_embedding('uid_follow_list_hetu2', dim=8, expand=50, slots=[909], **compress_kwargs)
uid_follow_hetu2_seq = tf.reshape(uid_follow_hetu2_seq, [-1, 50, 8])
uid_follow_seq = tf.concat([uid_follow_id_seq, uid_follow_hetu1_seq, uid_follow_hetu2_seq], axis=2) # [batchsize, 50, 48]

uid_forward_id_seq = config.new_embedding('uid_forward_list_id', dim=32, expand=50, slots=[910], **compress_kwargs)
uid_forward_id_seq = tf.reshape(uid_forward_id_seq, [-1, 50, 32])
uid_forward_hetu1_seq = config.new_embedding('uid_forward_list_hetu1', dim=8, expand=50, slots=[911], **compress_kwargs)
uid_forward_hetu1_seq = tf.reshape(uid_forward_hetu1_seq, [-1, 50, 8])
uid_forward_hetu2_seq = config.new_embedding('uid_forward_list_hetu2', dim=8, expand=50, slots=[912], **compress_kwargs)
uid_forward_hetu2_seq = tf.reshape(uid_forward_hetu2_seq, [-1, 50, 8])
uid_forward_seq = tf.concat([uid_forward_id_seq, uid_forward_hetu1_seq, uid_forward_hetu2_seq], axis=2) # [batchsize, 50, 48]

uid_comment_id_seq = config.new_embedding('uid_comment_list_id', dim=32, expand=50, slots=[913], **compress_kwargs)
uid_comment_id_seq = tf.reshape(uid_comment_id_seq, [-1, 50, 32])
uid_comment_hetu1_seq = config.new_embedding('uid_comment_list_hetu1', dim=8, expand=50, slots=[914], **compress_kwargs)
uid_comment_hetu1_seq = tf.reshape(uid_comment_hetu1_seq, [-1, 50, 8])
uid_comment_hetu2_seq = config.new_embedding('uid_comment_list_hetu2', dim=8, expand=50, slots=[915], **compress_kwargs)
uid_comment_hetu2_seq = tf.reshape(uid_comment_hetu2_seq, [-1, 50, 8])
uid_comment_seq = tf.concat([uid_comment_id_seq, uid_comment_hetu1_seq, uid_comment_hetu2_seq], axis=2) # [batchsize, 50, 48]
# [32] [B 32]
# [32*50]
uid_collect_id_seq = config.new_embedding('uid_collect_list_id', dim=32, expand=50, slots=[916], **compress_kwargs)
uid_collect_id_seq = tf.reshape(uid_collect_id_seq, [-1, 50, 32])
uid_collect_hetu1_seq = config.new_embedding('uid_collect_list_hetu1', dim=8, expand=50, slots=[917], **compress_kwargs)
uid_collect_hetu1_seq = tf.reshape(uid_collect_hetu1_seq, [-1, 50, 8])
uid_collect_hetu2_seq = config.new_embedding('uid_collect_list_hetu2', dim=8, expand=50, slots=[918], **compress_kwargs)
uid_collect_hetu2_seq = tf.reshape(uid_collect_hetu2_seq, [-1, 50, 8])
uid_collect_seq = tf.concat([uid_collect_id_seq, uid_collect_hetu1_seq, uid_collect_hetu2_seq], axis=2) # [batchsize, 50, 48]  [1, 50, 48]

uid_profile_enter_id_seq = config.new_embedding('uid_profile_enter_list_id', dim=32, expand=50, slots=[919], **compress_kwargs)
uid_profile_enter_id_seq = tf.reshape(uid_profile_enter_id_seq, [-1, 50, 32])
uid_profile_enter_hetu1_seq = config.new_embedding('uid_profile_enter_list_hetu1', dim=8, expand=50, slots=[920], **compress_kwargs)
uid_profile_enter_hetu1_seq = tf.reshape(uid_profile_enter_hetu1_seq, [-1, 50, 8])
uid_profile_enter_hetu2_seq = config.new_embedding('uid_profile_enter_list_hetu2', dim=8, expand=50, slots=[921], **compress_kwargs)
uid_profile_enter_hetu2_seq = tf.reshape(uid_profile_enter_hetu2_seq, [-1, 50, 8])
uid_profile_enter_seq = tf.concat([uid_profile_enter_id_seq, uid_profile_enter_hetu1_seq, uid_profile_enter_hetu2_seq], axis=2) # [batchsize, 50, 48]


#uid_seq = tf.reduce_sum(uid_seq, axis=1)
#uid_seq_output = simple_dense_network("seq_encoder", uid_seq, [32], act=tf.nn.leaky_relu, last_act=None)
if args.mode == 'train':
    target_pids = tf.expand_dims(tf.concat([pid_emb, pid_hetu], axis=1), 1) #[batchsize, 1, 48] []
elif args.mode == 'predict':
    target_pids = tf.expand_dims(tf.concat([pid_emb, pid_hetu], axis=1), 0) #[1, batchsize, 48] []
    uid_like_seq = tf.expand_dims(uid_like_seq[0,:,:], 0)
    uid_follow_seq = tf.expand_dims(uid_follow_seq[0,:,:], 0)
    uid_forward_seq = tf.expand_dims(uid_forward_seq[0,:,:], 0)
    uid_comment_seq = tf.expand_dims(uid_comment_seq[0,:,:], 0)
    uid_collect_seq = tf.expand_dims(uid_collect_seq[0,:,:], 0)
    uid_profile_enter_seq = tf.expand_dims(uid_profile_enter_seq[0,:,:], 0)

like_seq_output = multi_head_attention("like_seq_attention", target_pids, uid_like_seq, num_heads=4, num_units=32)  #[batchsize, 1, 32] infer: [1, batchsize, 32]
follow_seq_output = multi_head_attention("follow_seq_attention", target_pids, uid_follow_seq, num_heads=4, num_units=32)  #[batchsize, 1, 32]
forward_seq_output = multi_head_attention("forward_seq_attention", target_pids, uid_forward_seq, num_heads=4, num_units=32)  #[batchsize, 1, 32]
comment_seq_output = multi_head_attention("comment_seq_attention", target_pids, uid_comment_seq, num_heads=4, num_units=32)  #[batchsize, 1, 32]
collect_seq_output = multi_head_attention("collect_seq_attention", target_pids, uid_collect_seq, num_heads=4, num_units=32)  #[batchsize, 1, 32]
profile_enter_seq_output = multi_head_attention("profile_enter_seq_attention", target_pids, uid_profile_enter_seq, num_heads=4, num_units=32)  #[batchsize, 1, 32]

seq_output = tf.concat([like_seq_output, follow_seq_output, forward_seq_output, comment_seq_output, collect_seq_output, profile_enter_seq_output], axis=2) #[batchsize, 1, 32*6] infer: [1, batchsize, 32*6]
seq_output = tf.reshape(seq_output, [-1, 32*6])
# user_feature = [uid_emb, uid_stat, did_stat, uid_seq, uid_live_f, uid_loc_f, uid_viewid_f, user_realshow_list_input_sum]
# user_feature = tf.concat(user_feature, axis=1)


user_feature = [uid_emb, uid_stat, did_stat, uid_live_f, uid_loc_f, uid_viewid_f, user_realshow_list_input_sum, seq_output]
user_feature = tf.concat(user_feature, axis=1)


feature_input = tf.concat([user_feature, photo_feature], axis=1)

step = 0
tree_leaf_node_num = 8
class_num = tree_leaf_node_num-1
consumption_depth_quantile_list = [0, 1.94, 2.89, 3.87, 5.18, 6.86, 9.74, 15.99, 149.59]
##tpm loss  # params
var_w = 1.0
mse_weight = 1.0

print_tensor_opt = []
with tf.name_scope('model'):
        
    follow_logit = simple_dense_network("follow_layers", feature_input, [256, 128, 64, 1], act=tf.nn.leaky_relu, last_act=None)
    forward_inside_logit = simple_dense_network("forward_inside_layers", feature_input, [256, 128, 64, 1], act=tf.nn.leaky_relu, last_act=None)
    interact_logit = simple_dense_network("interact_layers", feature_input, [256, 128, 64, 1], act=tf.nn.leaky_relu, last_act=None)
    click_comment_logit = simple_dense_network("click_comment_layers", feature_input, [256, 128, 64, 1], act=tf.nn.leaky_relu, last_act=None)
    comment_time_score_logit = simple_dense_network("comment_time_layers", feature_input, [256, 128, 64, 1], act=tf.nn.leaky_relu, last_act=None)
    long_view_counter_factual_score_cmt_logit = simple_dense_network("long_view_wiz_cmt_layers", feature_input, [256, 128, 64, 1], act=tf.nn.leaky_relu, last_act=None)
    long_view_counter_factual_score_no_cmt_logit = simple_dense_network("long_view_wiz_no_cmt_layers", feature_input, [256, 128, 64, 1], act=tf.nn.leaky_relu, last_act=None)
    comment_top_net =  simple_dense_network("comment_top_net", feature_input, [256, 128], act=tf.nn.leaky_relu, last_act=tf.nn.leaky_relu)
  
    comment_unfold_score_logit = simple_dense_network("comment_unfold_layers", comment_top_net, [64, 1], act=tf.nn.leaky_relu, last_act=None)
    comment_like_score_logit = simple_dense_network("comment_like_layers", comment_top_net, [64, 1], act=tf.nn.leaky_relu, last_act=None)
    comment_content_copyward_score_logit = simple_dense_network("comment_content_copyward_layers", comment_top_net, [64, 1], act=tf.nn.leaky_relu, last_act=None)
    comment_effective_read_score_logit = simple_dense_network("comment_effective_read_layers", comment_top_net, [64, 1], act=tf.nn.leaky_relu, last_act=None)
    comment_slide_down_score_logit = simple_dense_network("comment_slide_down_layers", comment_top_net, [64, 1], act=tf.nn.leaky_relu, last_act=None)
    uplift_comment_consume_depth_score_logit = simple_dense_network("uplift_comment_consume_depth_layers", comment_top_net, [64, 1], act=tf.nn.leaky_relu, last_act=None)
    # uplift_click_comment_rate_score_logit = simple_dense_network("uplift_click_comment_rate_layers", comment_top_net, [64, 1], act=tf.nn.leaky_relu, last_act=None)
    uplift_comment_stay_duration_score_logit = simple_dense_network("uplift_comment_stay_duration_layers", comment_top_net, [64, 1], act=tf.nn.leaky_relu, last_act=None)
    playing_time_after_click_comment_score_logit = simple_dense_network("playing_time_after_click_comment_layers", comment_top_net, [64, 1], act=tf.nn.leaky_relu, last_act=None)

    # context aggregate
    effective_read_comment_fresh_logit = simple_dense_network("effective_read_comment_fresh_label_layers", comment_top_net, [64, 1], act=tf.nn.leaky_relu, last_act=None)

    eft_click_cmt_logit = simple_dense_network("eft_click_cmt", comment_top_net, [64, 1], act=tf.nn.leaky_relu, last_act=None)
    eft_write_cmt_logit = simple_dense_network("eft_write_cmt", comment_top_net, [64, 1], act=tf.nn.leaky_relu, last_act=None)

    comment_genre_layer = simple_dense_network("comment_genre_layers", feature_input, [256, 128], act=tf.nn.leaky_relu, last_act=tf.nn.leaky_relu)
    sub_comment_score_logit = simple_dense_network("sub_comment_layers", comment_genre_layer, [64, 1], act=tf.nn.leaky_relu, last_act=None)
    emoji_comment_score_logit = simple_dense_network("emoji_comment_layers", comment_genre_layer, [64, 1], act=tf.nn.leaky_relu, last_act=None)
    gif_comment_score_logit = simple_dense_network("gif_comment_layers", comment_genre_layer, [64, 1], act=tf.nn.leaky_relu, last_act=None)
    at_comment_score_logit = simple_dense_network("at_comment_layers", comment_genre_layer, [64, 1], act=tf.nn.leaky_relu, last_act=None)
    image_comment_score_logit = simple_dense_network("image_comment_layers", comment_genre_layer, [64, 1], act=tf.nn.leaky_relu, last_act=None)
    text_comment_score_logit = simple_dense_network("text_comment_layers", comment_genre_layer, [64, 1], act=tf.nn.leaky_relu, last_act=None)
    video_comment_score_logit = simple_dense_network("video_comment_layers", comment_genre_layer, [64, 1], act=tf.nn.leaky_relu, last_act=None)


    original_logits = [follow_logit, forward_inside_logit, interact_logit, click_comment_logit, comment_time_score_logit, 
                    comment_unfold_score_logit, comment_like_score_logit, comment_content_copyward_score_logit, comment_effective_read_score_logit,
                    sub_comment_score_logit, emoji_comment_score_logit, gif_comment_score_logit, at_comment_score_logit, 
                    image_comment_score_logit, text_comment_score_logit, video_comment_score_logit,
                    uplift_comment_consume_depth_score_logit, comment_slide_down_score_logit,uplift_comment_stay_duration_score_logit,
                    playing_time_after_click_comment_score_logit, effective_read_comment_fresh_logit,
                    eft_click_cmt_logit, eft_write_cmt_logit,long_view_counter_factual_score_cmt_logit,long_view_counter_factual_score_no_cmt_logit
                    ]

    logits = tf.concat(original_logits, axis=1) # [batchsize, len(original_logits)]
    context_logits = simple_dense_network("context_aware_logits_layers", logits, [64, len(original_logits)], act=tf.nn.relu, last_act=None)
    
    output = tf.nn.sigmoid(logits + context_logits)

    follow, forward_inside, interact, click_comment, comment_time_score, comment_unfold_score, comment_like_score, comment_content_copyward_score, \
    comment_effective_read_score, sub_comment_score, emoji_comment_score, gif_comment_score, at_comment_score, image_comment_score, \
    text_comment_score, video_comment_score, uplift_comment_consume_depth_score, comment_slide_down_score,uplift_comment_stay_duration_score, playing_time_after_click_comment_score, \
    effective_read_comment_fresh_score, eft_click_cmt_score, eft_write_cmt_score, long_view_counter_factual_score_cmt, long_view_counter_factual_score_no_cmt = tf.split(output, len(original_logits), axis=1)

## tpm
begins_tensor_tmp = tf.constant(consumption_depth_quantile_list[:-1], shape=[1, tree_leaf_node_num], dtype=tf.float32)
begins_tensor = tf.tile(begins_tensor_tmp, [tf.shape(follow)[0], 1])
ends_tensor_tmp = tf.constant(consumption_depth_quantile_list[1:], shape=[1, tree_leaf_node_num], dtype=tf.float32)
ends_tensor = tf.tile(ends_tensor_tmp, [tf.shape(follow)[0], 1])

## tpm consumption_depth
tpm_comment_consume_depth_pred = tree_model_net("tpm_comment_consume_depth_pred", comment_top_net, class_num=class_num, dropout=0, is_training=True)
tpm_comment_consume_depth, var = get_encoded_consumption_depth(tpm_comment_consume_depth_pred, tree_leaf_node_num, begins_tensor, ends_tensor)

if args.mode == 'train':
    import kai.tensorflow as kai
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

    my_step = config.get_step()
    
    slide_evtr = config.get_label('effective_view')
    time_weight = config.get_label('time_weight')
    one = tf.ones_like(slide_evtr, dtype=tf.float32)
    zero = tf.zeros_like(slide_evtr, dtype=tf.float32)

    comment_time = config.get_label('comment_watch_time')
    comment_stay_time = config.get_label('comment_stay_time')
    play_time = config.get_label('playing_time_s')
    comment_stay_time_s = config.get_label('comment_stay_time_s')
    play_time_wiz_cmt = tf.maximum(play_time - comment_stay_time_s , 0)
    playing_time_after_out_of_click_comment_label = config.get_label('playing_time_after_out_of_click_comment_label')

    effective_read_comment_fresh_label = config.get_label('fresh_true_sample_label')
    effective_read_comment_fresh_weight = config.get_label('effective_read_comment_fresh_weight')

    slide_click_comment_button = config.get_label('comment_effective_stay')
    slide_comment = config.get_label('comment')
    follow_label = config.get_label('follow')
    forward_inside_label = config.get_label('forward_inside')
    interact_label = config.get_label('interact_label')
    follow_weight = config.get_label('follow_weight')
    forward_inside_weight = config.get_label('forward_inside_weight')
    interact_weight = config.get_label('interact_weight')
    comment_action = config.get_label('comment_action_weight')
    
    comment_action_coeff = config.get_label('comment_action_coeff')
    comment_coeff = config.get_label('comment_coeff')
    comment_stay_coeff = config.get_label('comment_stay_coeff')

    comment_unfold = config.get_label("action_expand_secondary_comment_count")
    comment_like = config.get_label("action_like_comment")
    comment_content_copyward = config.get_label("comment_copyward")
    comment_effective_read = config.get_label("effective_read_comment")
    comment_slide_down = config.get_label("action_comment_slide_down")

    sub_comment = config.get_label("action_sub_comment")
    emoji_comment = config.get_label("action_emoji_comment")
    gif_comment = config.get_label("action_gif_comment")
    at_comment = config.get_label("action_at_comment")
    image_comment = config.get_label("action_image_comment")
    text_comment = config.get_label("action_text_comment")
    video_comment = config.get_label("action_video_comment")

    eft_click_cmt = config.get_label("eft_click_cmt")
    eft_write_cmt = config.get_label("eft_write_cmt")

    reward_comment_time = comment_time * comment_stay_coeff + comment_action * comment_action_coeff + slide_comment * comment_coeff 

    click_comment_mask = tf.where(slide_click_comment_button > 0, one, zero)
    comment_stay_time_mask = tf.where(comment_stay_time > 0, one, zero)
    no_comment_stay_time_mask = tf.where(comment_stay_time <= 0, one, zero)

    ## counter factual
    long_view = config.get_label("long_view")

    ## user cluster 
    user_comment_cluster_level = config.get_label("user_comment_cluster_level")
    low_user_comment_cluster_level = tf.where(tf.logical_or(user_comment_cluster_level <= 0, user_comment_cluster_level >=7), one, zero)
    mid_user_comment_cluster_level = tf.where(tf.logical_and(user_comment_cluster_level >= 4, user_comment_cluster_level <= 6), one, zero)
    high_user_comment_cluster_level = tf.where(tf.logical_and(user_comment_cluster_level >= 1, user_comment_cluster_level <=3), one, zero)

    ##tpm
    comment_watch_num = config.get_label("watch_comment_num")
    comment_consume_depth_xauc_label = tf.where(comment_watch_num > 5.18, one, zero)
    comment_watch_num_label_slot1 = config.get_label("comment_watch_num_label_slot1")
    comment_watch_num_label_slot2 = config.get_label("comment_watch_num_label_slot2")
    comment_watch_num_label_slot3 = config.get_label("comment_watch_num_label_slot3")
    comment_watch_num_label_slot4 = config.get_label("comment_watch_num_label_slot4")
    comment_watch_num_label_slot5 = config.get_label("comment_watch_num_label_slot5")
    comment_watch_num_label_slot6 = config.get_label("comment_watch_num_label_slot6")
    comment_watch_num_label_slot7 = config.get_label("comment_watch_num_label_slot7")
    label_slots = [
        comment_watch_num_label_slot1,
        comment_watch_num_label_slot2,
        comment_watch_num_label_slot3,
        comment_watch_num_label_slot4,
        comment_watch_num_label_slot5,
        comment_watch_num_label_slot6,
        comment_watch_num_label_slot7
    ]
    comment_watch_num_label = tf.concat(label_slots, axis=1)

    emp_comment_consume_depth = config.get_label("emp_comment_consume_depth")
    emp_click_comment_rate = config.get_label("emp_click_comment_rate")
    emp_comment_stay_duration = config.get_label("emp_comment_stay_duration")
    uplift_comment_consume_depth =  comment_watch_num - emp_comment_consume_depth
    uplift_comment_stay_duration =  comment_stay_time - emp_comment_stay_duration
    uplift_comment_consume_depth_label = tf.where(uplift_comment_consume_depth > 0, one, zero)
    uplift_comment_stay_duration_label = tf.where(uplift_comment_stay_duration > 0, one, zero)
    uplift_comment_consume_depth_weight =  tf.maximum(1.0, tf.abs(comment_watch_num - emp_comment_consume_depth))
    uplift_comment_stay_duration_weight =  tf.maximum(1.0, tf.abs(comment_stay_time - emp_comment_stay_duration) / 300)
    playing_time_after_out_of_click_comment_weight = config.get_label('playing_time_after_out_of_click_comment')/10

    ## 
    mask = tf.logical_and(playing_time_after_click_comment_score > 0, playing_time_after_click_comment_score < 1)
    filtered_scores = tf.boolean_mask(playing_time_after_click_comment_score, mask) 
    filtered_scores_label = tf.boolean_mask(playing_time_after_out_of_click_comment_label, mask)
    mask_one = tf.where(playing_time_after_click_comment_score == 0, one, zero)
    filtered_scores_mask_one = tf.boolean_mask(playing_time_after_click_comment_score, mask_one) 
    filtered_scores_label_mask_one = tf.boolean_mask(playing_time_after_out_of_click_comment_label, mask_one)

    targets = [
        ('click_comment', click_comment, slide_click_comment_button, one, 'auc'),
        ('comment_time_pos', comment_time_score, one, reward_comment_time * click_comment_mask, 'auc'),
        ('comment_time_neg', comment_time_score, zero, one * click_comment_mask, 'auc'),
        ('follow', follow, follow_label, follow_weight, 'auc'),
        ('forward_inside', forward_inside, forward_inside_label, forward_inside_weight, 'auc'),
        ('interact_once', interact, interact_label, interact_weight, 'auc'),
        ('comment_unfold', comment_unfold_score * tf.stop_gradient(click_comment), comment_unfold, one, 'auc'),
        ('comment_like', comment_like_score * tf.stop_gradient(click_comment), comment_like, one, 'auc'),
        ('comment_content_copyward', comment_content_copyward_score * tf.stop_gradient(click_comment), comment_content_copyward, one, 'auc'),
        ('comment_effective_read', comment_effective_read_score * tf.stop_gradient(click_comment), comment_effective_read, one, 'auc'),
        ('comment_slide_down', comment_slide_down_score * tf.stop_gradient(click_comment), comment_slide_down, one, 'auc'),
        ('uplift_comment_consume_depth', uplift_comment_consume_depth_score * tf.stop_gradient(click_comment), uplift_comment_consume_depth_label, uplift_comment_consume_depth_weight, 'auc'),
        ('uplift_comment_stay_duration', uplift_comment_stay_duration_score * tf.stop_gradient(click_comment), uplift_comment_stay_duration_label, uplift_comment_stay_duration_weight, 'auc'),
        ('playing_time_after_out_of_click_comment_label_click_space', playing_time_after_click_comment_score , playing_time_after_out_of_click_comment_label, comment_stay_time_mask, 'auc'),
        ('playing_time_after_out_of_click_comment_label_non_click_space', playing_time_after_click_comment_score , playing_time_after_click_comment_score, no_comment_stay_time_mask, 'auc'),

        ('effective_read_comment_fresh_label_pos', effective_read_comment_fresh_score, one, effective_read_comment_fresh_weight , 'auc'),
        ('effective_read_comment_fresh_label_neg', effective_read_comment_fresh_score, zero, effective_read_comment_fresh_label, 'auc'),
        ('long_view_counter_factual_score_cmt', long_view_counter_factual_score_cmt * tf.stop_gradient(click_comment) , long_view ,one , 'auc'),
        ('long_view_counter_factual_score_cmt', long_view_counter_factual_score_cmt , long_view ,  click_comment_mask / (click_comment + 1e-20) , 'auc'),
        ('long_view_counter_factual_score_no_cmt', long_view_counter_factual_score_no_cmt, (1-long_view), (1-click_comment_mask) / (1 - click_comment + 1e-20), 'auc'),

        ('sub_comment', sub_comment_score, sub_comment, one, 'auc'),
        ('emoji_comment', emoji_comment_score, emoji_comment, one, 'auc'),
        ('gif_comment', gif_comment_score, gif_comment, one, 'auc'),
        ('at_comment', at_comment_score, at_comment, one, 'auc'),
        ('image_comment', image_comment_score, image_comment, one, 'auc'),
        ('text_comment', text_comment_score, text_comment, one, 'auc'),
        ('eft_click_cmt', eft_click_cmt_score, eft_click_cmt, one, 'auc'),
        ('eft_write_cmt', eft_write_cmt_score, eft_write_cmt, one, 'auc'),
        ('eft_click_and_write_cmt', eft_click_cmt_score * eft_write_cmt_score, tf.where((eft_click_cmt > 0) & (eft_write_cmt > 0), one, zero), one, 'auc'),
    ]

    eval_targets = [
        ('click_comment', click_comment, slide_click_comment_button, one, 'auc'),
        ('follow', follow, follow_label, follow_weight, 'auc'),
        ('forward_inside', forward_inside, forward_inside_label, forward_inside_weight, 'auc'),
        ('interact_once', interact, interact_label, interact_weight, 'auc'),
        ('comment_unfold', comment_unfold_score * click_comment, comment_unfold, one, 'auc'),
        ('comment_like', comment_like_score * click_comment, comment_like, one, 'auc'),
        ('comment_content_copyward', comment_content_copyward_score * click_comment, comment_content_copyward, one, 'auc'),
        ('comment_effective_read', comment_effective_read_score * click_comment, comment_effective_read, one, 'auc'),
        ('comment_consume_depth_click_comment', tpm_comment_consume_depth * click_comment, comment_consume_depth_xauc_label, click_comment_mask, 'auc'),
        ('comment_consume_depth_total', tpm_comment_consume_depth * click_comment, comment_consume_depth_xauc_label, one, 'auc'),
        ('comment_slide_down', comment_slide_down_score * click_comment, comment_slide_down, one, 'auc'),
        ('uplift_comment_consume_depth', uplift_comment_consume_depth_score * click_comment, uplift_comment_consume_depth_label, one, 'auc'),
        ('uplift_comment_stay_duration', uplift_comment_stay_duration_score * click_comment, uplift_comment_stay_duration_label, one, 'auc'),
        ('long_view_counter_factual_score_cmt', long_view_counter_factual_score_cmt, long_view ,  one , 'auc'),
        ('long_view_counter_factual_score_no_cmt', long_view_counter_factual_score_no_cmt, (1-long_view), one, 'auc'),

        # ('effective_read_comment_fresh_label', effective_read_comment_fresh_score, effective_read_comment_fresh_label, one, 'auc'),
        ## 分人群auc
        ('comment_unfold_low_cluster', comment_unfold_score * click_comment, comment_unfold, low_user_comment_cluster_level, 'auc'),
        ('comment_unfold_mid_cluster', comment_unfold_score * click_comment, comment_unfold, mid_user_comment_cluster_level, 'auc'),
        ('comment_unfold_high_cluster', comment_unfold_score * click_comment, comment_unfold, high_user_comment_cluster_level, 'auc'),
        ('comment_like_low_cluster', comment_like_score * click_comment, comment_like, low_user_comment_cluster_level, 'auc'),
        ('comment_like_mid_cluster', comment_like_score * click_comment, comment_like, mid_user_comment_cluster_level, 'auc'),
        ('comment_like_high_cluster', comment_like_score * click_comment, comment_like, high_user_comment_cluster_level, 'auc'),
        ('comment_content_copyward_low_cluster', comment_content_copyward_score * click_comment, comment_content_copyward, low_user_comment_cluster_level, 'auc'),
        ('comment_content_copyward_mid_cluster', comment_content_copyward_score * click_comment, comment_content_copyward, mid_user_comment_cluster_level, 'auc'),
        ('comment_content_copyward_high_cluster', comment_content_copyward_score * click_comment, comment_content_copyward, high_user_comment_cluster_level, 'auc'),
        ('comment_effective_read_low_cluster', comment_effective_read_score * click_comment, comment_effective_read, low_user_comment_cluster_level, 'auc'),
        ('comment_effective_read_mid_cluster', comment_effective_read_score * click_comment, comment_effective_read, mid_user_comment_cluster_level, 'auc'),
        ('comment_effective_read_high_cluster', comment_effective_read_score * click_comment, comment_effective_read, high_user_comment_cluster_level, 'auc'),
        ('comment_slide_down_low_cluster', comment_slide_down_score * click_comment, comment_slide_down, low_user_comment_cluster_level, 'auc'),
        ('comment_slide_down_mid_cluster', comment_slide_down_score * click_comment, comment_slide_down, mid_user_comment_cluster_level, 'auc'),
        ('comment_slide_down_high_cluster', comment_slide_down_score * click_comment, comment_slide_down, high_user_comment_cluster_level, 'auc'),

        ('sub_comment', sub_comment_score, sub_comment, one, 'auc'),
        ('emoji_comment', emoji_comment_score, emoji_comment, one, 'auc'),
        ('gif_comment', gif_comment_score, gif_comment, one, 'auc'),
        ('at_comment', at_comment_score, at_comment, one, 'auc'),
        ('image_comment', image_comment_score, image_comment, one, 'auc'),
        ('text_comment', text_comment_score, text_comment, one, 'auc'),
        ('eft_click_cmt', eft_click_cmt_score, eft_click_cmt, one, 'auc'),
        ('eft_write_cmt', eft_write_cmt_score, eft_write_cmt, one, 'auc'),
        ('eft_click_and_write_cmt', eft_click_cmt_score * eft_write_cmt_score, tf.where((eft_click_cmt > 0) & (eft_write_cmt > 0), one, zero), one, 'auc'),
    ]

    comment_watch_num_log_loss = tf.losses.log_loss(comment_watch_num_label, tpm_comment_consume_depth_pred, click_comment_mask, reduction="weighted_sum")
    comment_watch_num_mse_loss = tf.reduce_sum(tf.square(comment_watch_num - tpm_comment_consume_depth * click_comment))
    tpm_train_losss_op = comment_watch_num_log_loss + comment_watch_num_mse_loss * mse_weight + var * var_w

    ## long_view_counter_factual loss
    long_view_counter_factual_loss_beta = 10
    long_view_counter_factual_regularization_loss = tf.abs(1 - (long_view_counter_factual_score_cmt + long_view_counter_factual_score_no_cmt))

    # 播看写递进约束 loss
    progressive_effective_click_comment_loss = tf.reduce_sum(tf.nn.relu(eft_write_cmt_score - eft_click_cmt_score))

    q_name, preds, labels, weights, auc = zip(*targets)

    total_loss = tf.losses.log_loss(labels, preds, weights, reduction="weighted_sum") + tpm_train_losss_op + progressive_effective_click_comment_loss + long_view_counter_factual_loss_beta * long_view_counter_factual_regularization_loss

    debug_tensor = {
        # "tpm_train_losss_op": tpm_train_losss_op,
        # "progressive_effective_click_comment_loss":  progressive_effective_click_comment_loss,
        # "total_loss": total_loss,
        "long_view_counter_factual_score_cmt": tf.slice(long_view_counter_factual_score_cmt, [0, 0], [20, -1]),
        "long_view_counter_factual_score_no_cmt": tf.slice(long_view_counter_factual_score_no_cmt, [0, 0], [20, -1]),
        "long_view_counter_factual_regularization_loss" : tf.slice(long_view_counter_factual_regularization_loss, [0, 0], [20, -1]),
        "long_view" : tf.slice(long_view, [0, 0], [20, -1]),
    }
    kai.add_run_hook(TensorPrintHook(debug_tensor), "debug_tensor_hook")

    with tf.control_dependencies(print_tensor_opt):
        if args.with_kai_v2:
                config.set_slot_param_attr([34, 35, 38, 184, 189], config.nn.ParamAttr(access_method=config.nn.ProbabilityAccess(100.0),
                                                       recycle_method=config.nn.UnseendaysRecycle(delete_after_unseen_days=30, delete_threshold=2.0, allow_dynamic_delete=False)))
                sparse_optimizer = config.optimizer.Adam(0.01)
                dense_optimizer = config.optimizer.Adam(0.0005)
                sparse_optimizer.minimize(total_loss, var_list=config.get_collection(config.GraphKeys.EMBEDDING_INPUT))
                dense_optimizer.minimize(total_loss, var_list=config.get_collection(config.GraphKeys.TRAINABLE_VARIABLES))
        else:
                optimizer = tf.train.GradientDescentOptimizer(1, name="opt")
                opt = optimizer.minimize(total_loss)

    if args.dryrun:
        config.mock_and_profile(tf.print(tf.shape(click_comment)), './training_log/', batch_sizes=[128, 288])
    elif args.with_kai:
        config.dump_kai_training_config('./training/conf', eval_targets, loss=total_loss, text=args.text, extra_ops=print_tensor_opt)
    elif args.with_kai_v2:
        config.build_model(optimizer=[sparse_optimizer, dense_optimizer], metrics=eval_targets)
    else:
        config.dump_training_config('./training/conf', eval_targets, opts=[opt, print_tensor_opt], text=args.text)


elif args.mode == 'predict':
    targets = [
        ('click_comment_score', click_comment),
        ('comment_stay_time_score', comment_time_score),
        ('follow', follow),
        ('forward_inside', forward_inside),
        ('interact_score', interact),
        ('comment_unfold', comment_unfold_score * click_comment),
        ('comment_like', comment_like_score * click_comment),
        ('comment_copyward', comment_content_copyward_score * click_comment),
        ('comment_effective_read', comment_effective_read_score * click_comment),
        ('comment_consume_depth', tpm_comment_consume_depth * click_comment),
        ('comment_slide_down', comment_slide_down_score * click_comment),
        ('uplift_comment_consume_depth_score', uplift_comment_consume_depth_score * click_comment),
        ('uplift_comment_stay_duration_score', uplift_comment_stay_duration_score * click_comment),
        ('playtime_after_click_comment_score', playing_time_after_click_comment_score * click_comment),
        ('effective_read_comment_fresh_score', effective_read_comment_fresh_score),
        ('long_view_counter_factual_score_cmt', long_view_counter_factual_score_cmt),
        ('long_view_counter_factual_score_no_cmt', long_view_counter_factual_score_no_cmt),

        ('sub_comment', sub_comment_score),
        ('emoji_comment', emoji_comment_score),
        ('gif_comment', gif_comment_score),
        ('at_comment', at_comment_score),
        ('image_comment', image_comment_score),
        ('text_comment', text_comment_score),
        # ('video_comment', video_comment_score),
        ('eft_click_cmt', eft_click_cmt_score),
        ('eft_write_cmt', eft_write_cmt_score),
    ]
    q_names, preds = zip(*targets)
    config.dump_predict_config('./predict/config', targets, input_type=3, extra_preds=q_names)
