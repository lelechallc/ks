from __future__ import print_function
from cProfile import label

import os
import logging
import argparse
import numpy
import sys
import functools
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['train', 'predict', 'user_predict', 'photo_predict'])
parser.add_argument('--dryrun', dest="dryrun", const=True, default=False, nargs='?')
parser.add_argument('--with_kai', action="store_true")
parser.add_argument('--text', action="store_true")
args = parser.parse_args()

if not args.dryrun and not args.with_kai:
    # monkey patch
    import mio_tensorflow.patch as mio_tensorflow_patch
    mio_tensorflow_patch.apply()

import tensorflow as tf
from tensorflow.keras.backend import expand_dims, repeat_elements, sum
from mio_tensorflow.config import MioConfig
from mio_tensorflow.variable import MioVariable, MioEmbedding
# import kai

logging.basicConfig()

base_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), './base.yaml')

# with kai 的时候 clear_params=False 会报错
config = MioConfig.from_base_yaml(base_config, clear_embeddings=True, clear_params=True,
                                  dryrun=args.dryrun, label_with_kv=True, grad_no_scale=False,
                                  with_kai=args.with_kai, predict=(args.mode != 'train'))

def batch_group_fm_quadratic(fm_input):
  summed_emb = tf.reduce_sum(fm_input, axis=1)
  summed_emb_squared = tf.square(summed_emb)
  squared_emb = tf.square(fm_input)
  squared_sum_emb = tf.reduce_sum(squared_emb, 1)
  fm_out = 0.5 * tf.subtract(summed_emb_squared, squared_sum_emb)
  return fm_out

def simple_dense_network(name, inputs, units, dropout=0, act=tf.nn.tanh, stop_gradient=False):
  if stop_gradient:
    output = tf.stop_gradient(inputs)
  else:
    output = inputs
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    if dropout > 0:
      output = tf.layers.dropout(output, dropout, training=(args.mode == 'train'))
    for i, unit in enumerate(units):
      # output = tf.layers.Dense(unit, act, name='dense_{}_{}'.format(name, i))(output)
      output = tf.layers.dense(output, unit, activation=act,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.5, seed=1)) #tf.glorot_normal_initializer ； glorot_uniform_initializer
    return output
  
def get_multi_trans_pxtr(pxtrs):
  """
  8种pxtrs变换
  """
  multi_trans_pxtr = [pxtrs] # 把原始的pxtrs数据也放进去
  multi_trans_pxtr.append(tf.tanh(pxtrs))                         # tanh
  multi_trans_pxtr.append(tf.sigmoid(pxtrs))                      # sigmoid
  multi_trans_pxtr.append(tf.sin(pxtrs))                          # sin
  # multi_trans_pxtr.append(tf.cos(pxtrs))                        # cos,感觉sin cos一个就可以了吧
  multi_trans_pxtr.append(tf.math.log(pxtrs))                     # log   
  # multi_trans_pxtr.append(tf.math.reciprocal(pxtrs))              # 1 / pxtrs 添加倒数训练时不收敛
  multi_trans_pxtr.append(tf.pow(pxtrs, 0.5))                     # 2 ^ pxtrs
  multi_trans_pxtr.append(tf.pow(2.0, pxtrs))                     # 2 ^ pxtrs
  multi_trans_pxtr.append(tf.pow(pxtrs, 2.0))                     # pxtrs ^ 2
  multi_trans_pxtr.append(tf.multiply(pxtrs, tf.sigmoid(pxtrs)))  # swish x * sigmoid(x)

  return tf.concat(multi_trans_pxtr, axis = 1) # 返回shape: ( batch_size * (len(pxtrs_float_list) * (1 + 8) ))

def weight_predtict_model(user_seq, photo_category, pxtrs):
  user_intent = intent_predictor(user_seq, 16)
  user_intent = simple_dense_network("intent_emb", user_intent, [16], act= tf.nn.sigmoid)
  pxtr_weights = ranking_ensemble_model(pxtrs, user_intent, photo_category)
  pred_score, temp_score = relative_quantile_ensemble(pxtr_weights, pxtrs)
  # pred_score_list = tf.unstack(pred_score, axis=-1)
  # pred_score_squeezed = tf.squeeze(pred_score_list, axis=-1)  # 压缩成形状为 [5, bz]
  # interact_score = simple_dense_network("ensemble_score", pxtrs, [1], act= tf.nn.softmax)
  return pxtr_weights, pred_score, temp_score

def ensemble(weights, pxtrs):
  ensemble_score = tf.reduce_prod(tf.pow(pxtrs, weights), axis = 1, keepdims=True)
  # ensemble_score = tf.reduce_sum(tf.multiply(pxtrs, weights), axis = 1, keepdims=True)
  temp_score= ensemble_score
  ensemble_score = simple_dense_network("ensemble_score", ensemble_score, [6], act= tf.nn.sigmoid)
  return ensemble_score, temp_score

def relative_quantile_ensemble(weights, pxtrs):
  # pxtrs_mean = tf.reduce_mean(pxtrs, axis=1, keepdims=True)
  # pxtrs_plus_2 = pxtrs
  # ensemble_score = tf.reduce_prod(tf.pow(weights, pxtrs_plus_2), axis = 1, keepdims=True)
  log_weights = tf.math.log(2+weights)
  ensemble_score = tf.reduce_sum(tf.multiply(pxtrs, log_weights), axis=1, keepdims=True)
  temp_score = ensemble_score
  ensemble_score = simple_dense_network("ensemble_score", ensemble_score, [1], act= tf.nn.sigmoid)
  return ensemble_score, temp_score

def weight_predtict_model_with_pair(user_seq, photo_category_idx0, photo_category_idx1, pxtrs_idx0, pxtrs_idx1):
  user_intent = intent_predictor(user_seq, 16)
  user_intent = simple_dense_network("intent_emb", user_intent, [16], act= tf.nn.sigmoid)
  pxtr_weights_idx0, pxtr_weights_idx1 = ranking_ensemble_model_with_pair(pxtrs_idx0, pxtrs_idx1, user_intent, photo_category_idx0, photo_category_idx1)
  pred_score_idx0, pred_score_idx1 = relative_quantile_ensemble_with_pair(pxtr_weights_idx0, pxtr_weights_idx1, pxtrs_idx0, pxtrs_idx1)
  # pred_score_list = tf.unstack(pred_score, axis=-1)
  # pred_score_squeezed = tf.squeeze(pred_score_list, axis=-1)  # 压缩成形状为 [5, bz]
  # interact_score = simple_dense_network("ensemble_score", pxtrs, [1], act= tf.nn.softmax)
  return pxtr_weights_idx0, pxtr_weights_idx1, pred_score_idx0, pred_score_idx1

def relative_quantile_ensemble_with_pair(weights_idx0, weights_idx1, pxtrs_idx0, pxtrs_idx1):
  # pxtrs_mean = tf.reduce_mean(pxtrs, axis=1, keepdims=True)
  # pxtrs_plus_2 = pxtrs
  # ensemble_score = tf.reduce_prod(tf.pow(weights, pxtrs_plus_2), axis = 1, keepdims=True)
  log_weights_idx0 = tf.math.log(2+weights_idx0)
  ensemble_score_idx0 = tf.reduce_sum(tf.multiply(pxtrs_idx0, log_weights_idx0), axis=1, keepdims=True)
  log_weights_idx1 = tf.math.log(2+weights_idx1)
  ensemble_score_idx1 = tf.reduce_sum(tf.multiply(pxtrs_idx1, log_weights_idx1), axis=1, keepdims=True)
  # temp_score_idx0 = ensemble_score_idx0
  # ensemble_score = simple_dense_network("ensemble_score_idx0", ensemble_score, [1], act= tf.nn.sigmoid)
  return ensemble_score_idx0, ensemble_score_idx1

def ranking_ensemble_model_with_pair(pxtrs_idx0, pxtrs_idx1, user_intent, photo_category_idx0, photo_category_idx1):
  pxtrs_idx0_expand = tf.expand_dims(pxtrs_idx0, 1)
  pxtrs_idx1_expand = tf.expand_dims(pxtrs_idx1, 1)
  user_intent_expand = tf.expand_dims(user_intent, 1)
  photo_category_idx0_expand = tf.expand_dims(photo_category_idx0, 1)
  photo_category_idx1_expand = tf.expand_dims(photo_category_idx1, 1)
  pxtr_idx0_atten = multi_head_attention("pxtr_idx0_self_attention", pxtrs_idx0_expand, pxtrs_idx0_expand, pxtrs_idx0_expand, 8, 2, 16)
  pxtr_idx1_atten = multi_head_attention("pxtr_idx1_self_attention", pxtrs_idx1_expand, pxtrs_idx1_expand, pxtrs_idx1_expand, 8, 2, 16)
  intent_cross_pxtr_idx0 = multi_head_attention("intent_aware_cross_pxtr_idx0_attention", user_intent_expand, pxtr_idx0_atten, pxtr_idx0_atten, 8, 2, 16)
  intent_cross_pxtr_idx1 = multi_head_attention("intent_aware_cross_pxtr_idx1_attention", user_intent_expand, pxtr_idx1_atten, pxtr_idx1_atten, 8, 2, 16)
  intent_cross_category_idx0 = multi_head_attention("intent_aware_cross_category_attention_idx0",
                user_intent_expand, photo_category_idx0_expand, photo_category_idx0_expand, 8, 2, 16)
  intent_cross_category_idx1 = multi_head_attention("intent_aware_cross_category_attention_idx1",
                user_intent_expand, photo_category_idx1_expand, photo_category_idx1_expand, 8, 2, 16)
  intent_cross_pxtr_idx0 = tf.squeeze(intent_cross_pxtr_idx0, axis = [1])
  intent_cross_pxtr_idx1 = tf.squeeze(intent_cross_pxtr_idx1, axis = [1])
  intent_cross_category_idx0 = tf.squeeze(intent_cross_category_idx0, axis = [1])
  intent_cross_category_idx1 = tf.squeeze(intent_cross_category_idx1, axis = [1])
  concat_reslut_idx0 = tf.concat([intent_cross_pxtr_idx0, intent_cross_category_idx0, user_intent], axis = 1)
  concat_reslut_idx1 = tf.concat([intent_cross_pxtr_idx1, intent_cross_category_idx1, user_intent], axis = 1)
  weights_idx0 = simple_dense_network("projection_idx0", concat_reslut_idx0, [8], act= tf.nn.elu)
  weights_idx1 = simple_dense_network("projection_idx1", concat_reslut_idx1, [8], act= tf.nn.elu)
  return weights_idx0, weights_idx1

def ranking_ensemble_model(pxtrs, user_intent, photo_category):
  pxtrs_expand = tf.expand_dims(pxtrs, 1)
  user_intent_expand = tf.expand_dims(user_intent, 1)
  photo_category_expand = tf.expand_dims(photo_category, 1)
  pxtr_atten = multi_head_attention("pxtr_self_attention", pxtrs_expand, pxtrs_expand, pxtrs_expand, 8, 2, 16)
  intent_cross_pxtr = multi_head_attention("intent_aware_cross_pxtr_attention", user_intent_expand, pxtr_atten, pxtr_atten, 8, 2, 16)
  intent_cross_category = multi_head_attention("intent_aware_cross_category_attention", user_intent_expand, photo_category_expand, photo_category_expand, 8, 2, 16)
  intent_cross_pxtr = tf.squeeze(intent_cross_pxtr, axis = [1])
  intent_cross_category = tf.squeeze(intent_cross_category, axis = [1])
  concat_reslut = tf.concat([intent_cross_pxtr, intent_cross_category, user_intent], axis = 1)
  weights= simple_dense_network("projection", concat_reslut, [8], act= tf.nn.elu)
  return weights

def multi_head_attention(name, querys, keys, values, multi_head_dim, head_num, output_dim, mask = False):

    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
      qs = tf.layers.dense(inputs=querys, units=multi_head_dim * head_num,
                                 use_bias=False)  # [batch_size, len_seq, dim]
      ks = tf.layers.dense(inputs=keys, units=multi_head_dim * head_num,
                            use_bias=False)  # [batch_size, len_seq, dim]
      vs = tf.layers.dense(inputs=values, units=multi_head_dim * head_num,
                            use_bias=False)  # [batch_size, len_seq, dim]
      # split head                   
      q_input = tf.concat(tf.split(qs, head_num, axis=-1), axis=0) # [head * batch_size, len_seq, dim/head]
      k_input = tf.concat(tf.split(ks, head_num, axis=-1), axis=0)
      v_input = tf.concat(tf.split(vs, head_num, axis=-1), axis=0)
		  # 公式 : Q*K/sqrt(dim)
      outputs = tf.matmul(q_input,tf.transpose(k_input,[0,2,1]))
      outputs = outputs / (k_input.get_shape().as_list()[-1]**0.5)
      # softmax 前 mask
      if mask: 
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
        
        #使 key_masks 的维度能够和 outputs 匹配
        key_masks = tf.tile(key_masks, [head_num, 1])
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(querys)[1], 1])

        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

      outputs = tf.nn.softmax(outputs)
      result = tf.matmul(outputs, v_input)
      result = tf.concat(tf.split(result, head_num, axis=0), axis=2)
      result = tf.layers.dense(result, output_dim)
    return result

def intent_predictor(user_seq, dim):
  # intent_input= tf.concat([user_seq, context], axis = 1)
  intent_input= user_seq
  output = simple_dense_network("intent_predictor", intent_input, [dim], act= tf.nn.softmax)
  return output

training = args.mode == 'train'
# user feature
compress_kwargs = {}
if args.mode in ["predict", "gsu"]:
  compress_kwargs["compress_group"] = "USER"

uid_emb = config.new_embedding("uid_emb", dim=32, slots=[38, 34], **compress_kwargs)
uid_stat = config.new_embedding("uid_stat", dim=8, slots=[184, 35, 189], **compress_kwargs)
did_stat = config.new_embedding("did_stat", dim=8, slots=[701, 702, 703, 704, 705, 706], **compress_kwargs)
u_mean_stat = config.new_embedding("u_mean_stat", dim = 8, slots = [950, 952, 954, 956, 958, 960], **compress_kwargs)
u_std_stat = config.new_embedding("u_std_stat", dim = 8, slots = [951, 953, 955, 957, 959, 961], **compress_kwargs)

# pair-wise 需要两套 item emb
slot_offset_list=[0,10000]
pid_emb_slots = [26,128]
pid_xtr_slots = [576, 577, 578, 579, 567, 146, 147, 71, 142]
pid_stat_slots = [152, 110, 185, 685, 686, 673, 1118, 141]
pid_gate_slots = [682, 683, 786, 787]
pid_pxtr_slots = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012]
top_bias_slots = [498, 143, 603, 3621]
photo_category_slots = [201, 202]

# item +
# pid_emb_idx0 = config.new_embedding("pid_emb_idx0", dim=32, slots=[i+slot_offset_list[0] for i in pid_emb_slots])
# pid_xtr_idx0 = config.new_embedding("pid_xtr_idx0", dim=8, slots=[i+slot_offset_list[0] for i in pid_xtr_slots])
# pid_stat_idx0 = config.new_embedding("pid_stat_idx0", dim=8, slots=[i+slot_offset_list[0] for i in pid_stat_slots])
# pid_gate_idx0 = config.new_embedding("pid_gate_idx0", dim=8, slots=[i+slot_offset_list[0] for i in pid_gate_slots])
# pid_pxtr_idx0 = config.new_embedding("pid_pxtr_idx0", dim=8, slots=[i+slot_offset_list[0] for i in pid_pxtr_slots])
# top_bias_idx0 = config.new_embedding("top_bias_idx0", dim=8, slots=[i+slot_offset_list[0] for i in top_bias_slots])
photo_category_idx0 = config.new_embedding("photo_category_idx0", dim=8, slots=[i+slot_offset_list[0] for i in photo_category_slots])

# item -
# pid_emb_idx1 = config.new_embedding("pid_emb_idx1", dim=32, slots=[i+slot_offset_list[1] for i in pid_emb_slots])
# pid_xtr_idx1 = config.new_embedding("pid_xtr_idx1", dim=8, slots=[i+slot_offset_list[1] for i in pid_xtr_slots])
# pid_stat_idx1 = config.new_embedding("pid_stat_idx1", dim=8, slots=[i+slot_offset_list[1] for i in pid_stat_slots])
# pid_gate_idx1 = config.new_embedding("pid_gate_idx1", dim=8, slots=[i+slot_offset_list[1] for i in pid_gate_slots])
# pid_pxtr_idx1 = config.new_embedding("pid_pxtr_idx1", dim=8, slots=[i+slot_offset_list[1] for i in pid_pxtr_slots])
# top_bias_idx1 = config.new_embedding("top_bias_idx1", dim=8, slots=[i+slot_offset_list[1] for i in top_bias_slots])
photo_category_idx1 = config.new_embedding("photo_category_idx1", dim=8, slots=[i+slot_offset_list[1] for i in photo_category_slots])

uid_seq_list = []
for i in range(1, 22):
  slot_id = 900 + i
  uid_seq_embed = config.new_embedding(f'uid_action_list_{i}', dim=8, expand=20, slots=[slot_id], **compress_kwargs)
  uid_seq_embed = tf.reshape(uid_seq_embed, [-1, 20, 8])
  uid_seq_list.append(uid_seq_embed)
# [[batch len dim]] -> [batch len dim * 13] -> [batch dim * 13]
uid_seq = tf.concat(uid_seq_list, axis=2) # []
uid_seq = tf.reduce_mean(uid_seq, axis=1)
uid_seq_output = simple_dense_network("seq_encoder", uid_seq, 
                                       [32], act=tf.nn.leaky_relu)

pxtrs_float_list = ["pltr","pwtr", "pftr", "pcmtr", "plvtr", "pctr", "pcmef", "pwtd"]
pxtrs_tensor_idx0_list = [config.get_extra_param(pxtr+"_idx0",size=1) for pxtr in pxtrs_float_list] # len(pxtrs_tensor_list) == len(pxtrs_float_list)
pxtrs_tensor_idx0 = tf.concat(pxtrs_tensor_idx0_list, axis = 1) # shape: batch_size * len(pxtrs_float_list)
pxtrs_tensor_idx1_list = [config.get_extra_param(pxtr+"_idx1",size=1) for pxtr in pxtrs_float_list]
pxtrs_tensor_idx1 = tf.concat(pxtrs_tensor_idx1_list, axis = 1) # shape: batch_size * len(pxtrs_float_list)
uid_seq_output = [uid_seq_output, u_mean_stat, u_std_stat]
uid_seq_output = tf.concat(uid_seq_output, axis=1)

# outputs = []
### ------------------------ weight_predtict_model ------------------------
# pxtr_weights, pred_score, temp_score = weight_predtict_model(uid_seq_output, photo_category, pxtrs_tensor)
pxtr_weights_idx0, pxtr_weights_idx1, pred_score_idx0, pred_score_idx1 = weight_predtict_model_with_pair(uid_seq_output, photo_category_idx0, photo_category_idx1, pxtrs_tensor_idx0, pxtrs_tensor_idx1)

level_reward_idx0 = config.get_label('level_reward_idx0')
level_reward_idx1 = config.get_label('level_reward_idx1')

like_eval_pair_label_idx0 = config.get_label('like_idx0')
like_eval_pair_label_idx1 = config.get_label('like_idx1')
follow_eval_pair_label_idx0 = config.get_label('follow_idx0')
follow_eval_pair_label_idx1 = config.get_label('follow_idx1')
forward_eval_pair_label_idx0 = config.get_label('forward_idx0')
forward_eval_pair_label_idx1 = config.get_label('forward_idx1')
comment_eval_pair_label_idx0 = config.get_label('comment_idx0')
comment_eval_pair_label_idx1 = config.get_label('comment_idx1')
click_comment_eval_pair_label_idx0 = config.get_label('click_comment_button_idx0')
click_comment_eval_pair_label_idx1 = config.get_label('click_comment_button_idx1')
long_view_eval_pair_label_idx0 = config.get_label('long_view_idx0')
long_view_eval_pair_label_idx1 = config.get_label('long_view_idx1')
effective_view_eval_pair_label_idx0 = config.get_label('effective_view_idx0')
effective_view_eval_pair_label_idx1 = config.get_label('effective_view_idx1')

one = tf.ones_like(like_eval_pair_label_idx0, dtype=tf.float32)
zero = tf.zeros_like(like_eval_pair_label_idx0, dtype=tf.float32)
like_eval_pair_label = tf.where(like_eval_pair_label_idx0 > like_eval_pair_label_idx1, one, zero)
follow_eval_pair_label = tf.where(follow_eval_pair_label_idx0 > follow_eval_pair_label_idx1, one, zero)
forward_eval_pair_label = tf.where(forward_eval_pair_label_idx0 > forward_eval_pair_label_idx1, one, zero)
comment_eval_pair_label = tf.where(comment_eval_pair_label_idx0 > comment_eval_pair_label_idx1, one, zero)
click_comment_eval_pair_label = tf.where(click_comment_eval_pair_label_idx0 > click_comment_eval_pair_label_idx1, one, zero)
long_view_eval_pair_label = tf.where(long_view_eval_pair_label_idx0 > long_view_eval_pair_label_idx1, one, zero)
effective_view_eval_pair_label = tf.where(effective_view_eval_pair_label_idx0 > effective_view_eval_pair_label_idx1, one, zero)

if args.mode == 'train':
  pred_logit_sub = tf.math.sigmoid(pred_score_idx0 - pred_score_idx1)
  reward_bias_param = tf.fill(tf.shape(level_reward_idx0), 100.0)
  reward_sub = tf.subtract(tf.log(tf.add(level_reward_idx0, reward_bias_param)), tf.log(tf.add(level_reward_idx1, reward_bias_param)))
  bpr_loss_weight = tf.where(reward_sub > one, reward_sub, one)

  def _BPR_loss_with_weight_by_logits_sub(logits_sub, loss_weights):
      loss = - tf.reduce_mean(tf.multiply(tf.math.log(logits_sub + 1e-7), loss_weights))
      return loss
  loss = _BPR_loss_with_weight_by_logits_sub(pred_logit_sub, bpr_loss_weight)

  pxtr_weights_idx0_pltr_pos = tf.where(tf.slice(pxtr_weights_idx0,[0, 0], [-1, 1]) > tf.slice(pxtr_weights_idx1,[0, 0], [-1, 1]) , one, zero)
  pxtr_weights_idx0_pwtr_pos = tf.where(tf.slice(pxtr_weights_idx0,[0, 1], [-1, 1]) > tf.slice(pxtr_weights_idx1,[0, 1], [-1, 1]) , one, zero)
  pxtr_weights_idx0_pftr_pos = tf.where(tf.slice(pxtr_weights_idx0,[0, 2], [-1, 1]) > tf.slice(pxtr_weights_idx1,[0, 2], [-1, 1]) , one, zero)
  pxtr_weights_idx0_pcmtr_pos = tf.where(tf.slice(pxtr_weights_idx0,[0, 3], [-1, 1]) > tf.slice(pxtr_weights_idx1,[0, 3], [-1, 1]) , one, zero)
  pxtr_weights_idx0_plvtr_pos = tf.where(tf.slice(pxtr_weights_idx0,[0, 4], [-1, 1]) > tf.slice(pxtr_weights_idx1,[0, 4], [-1, 1]) , one, zero)
  pxtr_weights_idx0_pctr_pos = tf.where(tf.slice(pxtr_weights_idx0,[0, 5], [-1, 1]) > tf.slice(pxtr_weights_idx1,[0, 5], [-1, 1]) , one, zero)

  ensemble_score_pos = tf.where(tf.slice(pred_score_idx0,[0, 0], [-1, 1]) > tf.slice(pred_score_idx1,[0, 0], [-1, 1]) , one, zero)
      
  # assert name in all_labels
  #label = config.get_label("interact_label")
  #labels = [config.get_label(label) for label in task_names]
  # weight = config.get_label('interact_weight')
  # one = tf.fill(tf.shape(weight), 1.0) # weight
  # zero = tf.fill(tf.shape(weight), 0.0) # weight
  # pxtr_weights_pltr_pos = tf.where(tf.slice(pxtr_weights, [0, 0], [-1, 1])> 0 , one, zero)
  # pxtr_weights_pwtr_pos = tf.where(tf.slice(pxtr_weights, [0, 1], [-1, 1])> 0 , one, zero)
  # pxtr_weights_pftr_pos = tf.where(tf.slice(pxtr_weights, [0, 2], [-1, 1])> 0 , one, zero)
  # pxtr_weights_pcmtr_pos = tf.where(tf.slice(pxtr_weights, [0, 3], [-1, 1])> 0 , one, zero)
  # pxtr_weights_plvtr_pos = tf.where(tf.slice(pxtr_weights, [0, 4], [-1, 1])> 0 , one, zero)
  # pxtr_weights_pctr_pos = tf.where(tf.slice(pxtr_weights, [0, 5], [-1, 1])> 0 , one, zero)

  # targets = [
  #   # ('good_quality', tf.slice(pred_score, [0, 0], [-1, 1]), good_quality, level_reward, 'auc'),
  #   ('good_quality_pos', pred_score, one, level_reward, 'auc'),
  #   ('good_quality_neg', pred_score, zero, one , 'auc'),
  #   ('like_label',pxtr_weights_pltr_pos, ltr_weight_label, one, 'auc'),
  #   ('wtr_weight_label',pxtr_weights_pwtr_pos, wtr_weight_label, one, 'auc'),
  #   ('ftr_weight_label',pxtr_weights_pftr_pos, ftr_weight_label, one, 'auc'),
  #   ('cmtr_weight_label',pxtr_weights_pcmtr_pos, cmtr_weight_label, one, 'auc'),
  #   ('lvtr_weight_label',pxtr_weights_plvtr_pos, lvtr_weight_label, one, 'auc'),
  #   ('ctr_weight_label',pxtr_weights_pctr_pos, ctr_weight_label, one, 'auc'),
  #   # ('interact_label', tf.slice(pred_score, [0, 0], [-1, 1]), interact_label, weight, 'auc'),
  #   # ('click_comment_label', tf.slice(pred_score, [0, 1], [-1, 1]), slide_click_comment_button, one, 'auc'),
  #   # ('long_view', tf.slice(pred_score, [0, 2], [-1, 1]), long_view, one, 'auc'),
  #   # ('effective_view', tf.slice(pred_score, [0, 3], [-1, 1]), effective_view, one, 'auc'),
  #   # ('follow', tf.slice(pred_score, [0, 4], [-1, 1]), follow, one, 'auc'),
  #   # ('comment_time_pos', tf.slice(pred_score, [0, 5], [-1, 1]), one, reward_comment_time * click_comment_mask, 'auc'),
  #   # ('comment_time_neg', tf.slice(pred_score, [0, 5], [-1, 1]), zero, one * click_comment_mask, 'auc'),
  # ]
  eval_targets = [
    ('like_weight_label',pxtr_weights_idx0_pltr_pos, like_eval_pair_label, one, 'auc'),
    ('wtr_weight_label',pxtr_weights_idx0_pwtr_pos, follow_eval_pair_label, one, 'auc'),
    ('ftr_weight_label',pxtr_weights_idx0_pftr_pos, forward_eval_pair_label, one, 'auc'),
    ('cmtr_weight_label',pxtr_weights_idx0_pcmtr_pos, comment_eval_pair_label, one, 'auc'),
    ('clmt_weight_label',pxtr_weights_idx0_pcmtr_pos, click_comment_eval_pair_label, one, 'auc'),
    ('lvtr_weight_label',pxtr_weights_idx0_plvtr_pos, long_view_eval_pair_label, one, 'auc'),
    ('ctr_weight_label',pxtr_weights_idx0_pctr_pos, effective_view_eval_pair_label, one, 'auc'),

    ('like_ensemble_score_pos',ensemble_score_pos, like_eval_pair_label, one, 'auc'),
    ('wtr_ensemble_score_pos',ensemble_score_pos, follow_eval_pair_label, one, 'auc'),
    ('ftr_ensemble_score_pos',ensemble_score_pos, forward_eval_pair_label, one, 'auc'),
    ('cmtr_ensemble_score_pos',ensemble_score_pos, comment_eval_pair_label, one, 'auc'),
    ('clmt_ensemble_score_pos',ensemble_score_pos, click_comment_eval_pair_label, one, 'auc'),
    ('lvtr_ensemble_score_pos',ensemble_score_pos, long_view_eval_pair_label, one, 'auc'),
    ('ctr_ensemble_score_pos',ensemble_score_pos, effective_view_eval_pair_label, one, 'auc'),
  ]
  
  
  
  
  print_ops = []
  my_step = config.get_step()
  print_op = tf.cond(
    tf.equal(tf.mod(my_step, 10), 0),
    lambda: tf.print(
              "bpr loss:", loss,
              ", pred_score_idx0: ", tf.slice(pred_score_idx0, [0, 0], [10, 1]),
              ", pred_score_idx1: ", tf.slice(pred_score_idx1, [0, 0], [10, 1]),
              ", level_reward_idx0:", tf.slice(level_reward_idx0, [0, 0], [10, 1]),
              ", level_reward_idx1:", tf.slice(level_reward_idx1, [0, 0], [10, 1]),
              ", mse_idx0:", tf.square(tf.slice(pred_score_idx0, [0, 0], [10, 1]) - tf.slice(level_reward_idx0, [0, 0], [10, 1])),
              ", pxtr_weights_pltr_idx0", tf.slice(pxtr_weights_idx0, [0, 0], [10, 1]),
              ", pxtr_weights_pwtr_idx0", tf.slice(pxtr_weights_idx0, [0, 1], [10, 1]),
              ", pxtr_weights_pftr_idx0", tf.slice(pxtr_weights_idx0, [0, 2], [10, 1]),
              ", pxtr_weights_pcmtr_idx0", tf.slice(pxtr_weights_idx0, [0, 3], [10, 1]),
              ", pxtr_weights_plvtr_idx0", tf.slice(pxtr_weights_idx0, [0, 4], [10, 1]),
              ", pxtr_weights_pct_idx0", tf.slice(pxtr_weights_idx0, [0, 5], [10, 1]),
              ", pxtr_weights_pcmef_idx0", tf.slice(pxtr_weights_idx0, [0, 6], [10, 1]),
              ", pxtr_weights_pwtd_idx0", tf.slice(pxtr_weights_idx0, [0, 7], [10, 1]),
              ", pred_logit_sub: ", tf.reduce_mean(pred_logit_sub),
              summarize=-1,
              output_stream=sys.stdout), lambda: tf.no_op()
  )    
  print_ops.append(print_op)

  # optimizer = tf.train.AdagradOptimizer(0.2, initial_accumulator_value=0.1)
  optimizer = tf.train.GradientDescentOptimizer(0.001, name="opt")
  opt = optimizer.minimize(loss)
  if args.with_kai: # 同步
    config.dump_kai_training_config('./training/conf', eval_targets, loss=loss, text=args.text, extra_ops=print_ops, init_params_in_tf=True)
  else: # 异步
    optimizer = tf.train.GradientDescentOptimizer(1, name="opt")
    opt = optimizer.minimize(loss)
    config.dump_training_config('./training/conf', eval_targets, opts=[opt, print_op], text=args.text)

else:
  targets = []
  task_names = ["like_weight", "follow_weight", "forward_weight", "comment_weight", "longview_weight", "effective_view_weight"]
  task_names_idx1 = ["like_weight_idx1", "follow_weight_idx1", "forward_weight_idx1", "comment_weight_idx1", "longview_weight_idx1", "effective_view_weight_idx1"]
  num_weights = 6
  outputs = tf.split(pxtr_weights_idx0, num_weights, axis  = 1)
  outputs_idx1 = tf.split(pxtr_weights_idx1, num_weights, axis  = 1)
  for task_name, pred in zip(task_names, outputs):
    targets.append((task_name, pred))
  for task_name, pred in zip(task_names_idx1, outputs_idx1):
    targets.append((task_name, pred))
  q_names, preds = zip(*targets)

  if args.dryrun:
    config.mock_and_profile(preds, "./predict_log/", batch_sizes=[200], compressed_embedding_size={"USER": 4})
  else:
    config.dump_predict_config("./predict/config", targets, input_type=3, extra_preds=q_names)
