from __future__ import print_function
from cProfile import label

import os
import logging
import argparse
import numpy
import sys
import functools
import numpy as np

from umnn_model import MonotonicNN

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

def unmm_network(name, inputs, units, dropout=0, act=tf.nn.relu, last_act=tf.nn.elu, stop_gradient=False):
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
      output = output + 1
      return output
   
def umnn_integralNN(pxtrs, context_pxtr):
  orl_pxtrs = pxtrs
  _, indices = tf.nn.top_k(-pxtrs[:, 0], k=tf.shape(pxtrs)[0])
  sorted_by_pxtr1_pxtrs = tf.gather(pxtrs, indices, axis=0)
  cumulative_sum = tf.cumsum(context_pxtr, axis=1)

  # 使用 TensorArray 处理动态形状
  sorted_by_pxtr1_pxtrs_array = tf.TensorArray(dtype=tf.float32, size=tf.shape(sorted_by_pxtr1_pxtrs)[0])

  # 使用 tf.while_loop 进行循环
  def body(i, ta):
      ta = ta.write(i, sorted_by_pxtr1_pxtrs[i])
      return i + 1, ta

  def condition(i, ta):
      return i < tf.shape(sorted_by_pxtr1_pxtrs)[0]

  _, sorted_by_pxtr1_pxtrs_array = tf.while_loop(condition, body, [0, sorted_by_pxtr1_pxtrs_array])

  # 将 TensorArray 转换回张量
  sorted_by_pxtr1_pxtrs = sorted_by_pxtr1_pxtrs_array.stack()

  integral_loss = tf.reduce_sum(cumulative_sum)
  return sorted_by_pxtr1_pxtrs, orl_pxtrs, integral_loss

def weight_predtict_model(user_seq, photo_category, pxtrs):
  user_intent = intent_predictor(user_seq, 16)
  user_intent = simple_dense_network("intent_emb", user_intent, [16], act= tf.nn.sigmoid)
  pxtr_weights = ranking_ensemble_model(pxtrs, user_intent, photo_category)
  pred_score, temp_score = relative_quantile_ensemble(pxtr_weights, pxtrs)
  # pred_score_list = tf.unstack(pred_score, axis=-1)
  # pred_score_squeezed = tf.squeeze(pred_score_list, axis=-1)  # 压缩成形状为 [5, bz]
  # interact_score = simple_dense_network("ensemble_score", pxtrs, [1], act= tf.nn.softmax)
  return pxtr_weights, pred_score, temp_score

def weight_predtict_umnn_model(user_seq, photo_category, pxtrs):
  user_intent = intent_predictor(user_seq, 16)
  user_intent = simple_dense_network("intent_emb", user_intent, [16], act= tf.nn.sigmoid)
  orl_pxtrs = pxtrs
  post_pxtr = ranking_ensemble_umnn_model(pxtrs, user_intent, photo_category, hidden_layers = [256, 128, 64, 32] )

  pred_ensemble_score = tf.reduce_prod(post_pxtr, axis = 1, keepdims=True)
  orl_ensemble_score = tf.reduce_prod(orl_pxtrs, axis = 1, keepdims=True)
  return pred_ensemble_score, orl_ensemble_score, integral_loss, sorted_by_pxtr1_pxtrs

def ranking_ensemble_umnn_model(pxtrs, user_intent, photo_category, hidden_layers):
  pxtrs_expand = tf.expand_dims(pxtrs, 1)
  user_intent_expand = tf.expand_dims(user_intent, 1)
  photo_category_expand = tf.expand_dims(photo_category, 1)
  pxtr_atten = multi_head_attention("pxtr_self_attention", pxtrs_expand, pxtrs_expand, pxtrs_expand, 8, 2, 16)
  intent_cross_pxtr = multi_head_attention("intent_aware_cross_pxtr_attention", user_intent_expand, pxtr_atten, pxtr_atten, 8, 2, 16)
  intent_cross_category = multi_head_attention("intent_aware_cross_category_attention",
                user_intent_expand, photo_category_expand, photo_category_expand, 8, 2, 16)
  intent_cross_pxtr = tf.squeeze(intent_cross_pxtr, axis = [1])
  intent_cross_category = tf.squeeze(intent_cross_category, axis = [1])
  concat_reslut = tf.concat([intent_cross_pxtr, intent_cross_category, user_intent], axis = 1)

  # concat_reslut_wiz_pxtr = tf.concat([concat_reslut, tf.expand_dims(pxtrs[:,0], axis=1)], axis=1)

  input_dim = concat_reslut.get_shape().as_list()[1]  
  model_monotonic = MonotonicNN(input_dim, hidden_layers)
  #model_monotonic = MonotonicNN(pxtrs_expand, concat_reslut)
  post_pxtr = model_monotonic.forward(pxtrs_expand[:, 0:1], concat_reslut)
  pxtrs_updated = tf.concat([tf.expand_dims(post_pxtr, axis=1), pxtrs[:, 1:]], axis=1)
  return pxtrs_updated

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

def umnn_ensemble(sorted_by_pxtr1_pxtrs):
  ensemble_score = tf.reduce_prod(sorted_by_pxtr1_pxtrs, axis = 1, keepdims=True)
  ##ensemble_score = simple_dense_network("ensemble_score", ensemble_score, [1], act= tf.nn.sigmoid)
  return ensemble_score

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
  weights= simple_dense_network("projection", concat_reslut, [6], act= tf.nn.elu)
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

pid_emb = config.new_embedding("pid_emb", dim=32, slots=[26, 128])
pid_xtr = config.new_embedding("pid_xtr", dim=8, slots=[576, 577, 578, 579, 567, 146, 147, 71, 142]) # 9
pid_stat = config.new_embedding("pid_stat", dim=8, slots=[152, 110, 185, 685, 686, 673, 1118, 141])
pid_gate = config.new_embedding("pid_gate", dim=8, slots=[682, 683, 786, 787])
pid_pxtr = config.new_embedding("pid_pxtr", dim=8, slots=[1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012])
top_bias = config.new_embedding("top_bias", dim=8, slots=[498, 143, 603, 3621])
photo_category = config.new_embedding("photo_category", dim=8, slots=[201, 202])

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

# pxtrs_float_list = ["pctr", "pltr", "pftr", "pwtr", "plvtr", "psvr", "pvtr", "pptr", "pcmtr",
#                     "empirical_ctr", "empirical_ltr", "empirical_ftr",
#                     "cascade_pctr", "cascade_plvtr", "cascade_psvr", "cascade_pltr"]

pxtrs_float_list = ["pltr","pwtr", "pftr", "pcmtr", "plvtr", "pctr"]
pxtrs_tensor_list = [config.get_extra_param(pxtr, size = 1) for pxtr in pxtrs_float_list] # len(pxtrs_tensor_list) == len(pxtrs_float_list)
pxtrs_tensor = tf.concat(pxtrs_tensor_list, axis = 1) # shape: batch_size * len(pxtrs_float_list)
# epsilon = 1e-7
# pxtrs_tensor_standardization = tf.clip_by_value(pxtrs_tensor, clip_value_min=epsilon, clip_value_max=1.0 - epsilon)  # 使得最小值大于0
# multi_pxtrs_float_tensor = get_multi_trans_pxtr(pxtrs_tensor_standardization)

# feature_input = [uid_emb, uid_stat, did_stat, pid_emb, pid_xtr, pid_stat, pid_gate, top_bias, pid_pxtr, uid_seq_output, multi_pxtrs_float_tensor]
# feature_input = tf.concat(feature_input, axis=1)

uid_seq_output = [uid_seq_output, u_mean_stat, u_std_stat]
uid_seq_output = tf.concat(uid_seq_output, axis=1)

# outputs = []
### ------------------------ weight_predtict_model ------------------------
pred_score, orl_ensemble_score, integral_loss, sorted_by_pxtr1_pxtrs = weight_predtict_umnn_model(uid_seq_output, photo_category, pxtrs_tensor)

task_names = ["good_quality"]


interact_label = config.get_label("interact_label")
slide_click_comment_button = config.get_label("comment_effective_stay")
long_view = config.get_label("long_view")
effective_view = config.get_label("effective_view")
follow = config.get_label("follow")
one = tf.ones_like(long_view, dtype=tf.float32)
zero = tf.zeros_like(long_view, dtype=tf.float32)
click_comment_mask = tf.where(slide_click_comment_button > 0, one, zero)

comment_time = config.get_label('comment_watch_time')
comment_action_coeff = config.get_label('comment_action_coeff')
comment_stay_coeff = config.get_label('comment_stay_coeff')
comment_action = config.get_label('comment_action_weight')
slide_comment = config.get_label('comment')
comment_coeff = config.get_label('comment_coeff')
reward_comment_time = comment_time * comment_stay_coeff + comment_action * comment_action_coeff + slide_comment * comment_coeff 

good_quality = config.get_label('good_quality')
level_reward = config.get_label('level_reward')
## log
ltr_alpha = config.get_label('ltr_alpha')
wtr_alpha = config.get_label('wtr_alpha')
ftr_alpha = config.get_label('ftr_alpha')
cmtr_alpha = config.get_label('cmtr_alpha')
cmef_alpha = config.get_label('cmef_alpha')
ctr_alpha = config.get_label('ctr_alpha')
lvtr_alpha = config.get_label('lvtr_alpha')
like = config.get_label('like')
forward = config.get_label('forward')
click_comment_button = config.get_label('click_comment_button')

level_reward_pos= tf.where(tf.square(pred_score - level_reward) < 0.05, one, zero)

if args.mode == 'train':
      
  # assert name in all_labels
  #label = config.get_label("interact_label")
  #labels = [config.get_label(label) for label in task_names]
  weight = config.get_label('interact_weight')
  one = tf.fill(tf.shape(weight), 1.0) # weight
  zero = tf.fill(tf.shape(weight), 0.0) # weight
  # pxtr_weights_pltr_pos = tf.where(tf.slice(pxtr_weights, [0, 0], [-1, 1])> 0 , one, zero)
  # pxtr_weights_pwtr_pos = tf.where(tf.slice(pxtr_weights, [0, 1], [-1, 1])> 0 , one, zero)
  # pxtr_weights_pftr_pos = tf.where(tf.slice(pxtr_weights, [0, 2], [-1, 1])> 0 , one, zero)
  # pxtr_weights_pcmtr_pos = tf.where(tf.slice(pxtr_weights, [0, 3], [-1, 1])> 0 , one, zero)
  # pxtr_weights_plvtr_pos = tf.where(tf.slice(pxtr_weights, [0, 4], [-1, 1])> 0 , one, zero)
  # pxtr_weights_pctr_pos = tf.where(tf.slice(pxtr_weights, [0, 5], [-1, 1])> 0 , one, zero)

  targets = [
    ('good_quality', pred_score, good_quality, level_reward, 'auc'),
    # ('good_quality_pos', pred_score, one, level_reward, 'auc'),
    # ('good_quality_neg', pred_score, zero, one , 'auc'),
    # ('interact_label', tf.slice(pred_score, [0, 0], [-1, 1]), interact_label, weight, 'auc'),
    # ('click_comment_label', tf.slice(pred_score, [0, 1], [-1, 1]), slide_click_comment_button, one, 'auc'),
    # ('long_view', tf.slice(pred_score, [0, 2], [-1, 1]), long_view, one, 'auc'),
    # ('effective_view', tf.slice(pred_score, [0, 3], [-1, 1]), effective_view, one, 'auc'),
    # ('follow', tf.slice(pred_score, [0, 4], [-1, 1]), follow, one, 'auc'),
    # ('comment_time_pos', tf.slice(pred_score, [0, 5], [-1, 1]), one, reward_comment_time * click_comment_mask, 'auc'),
    # ('comment_time_neg', tf.slice(pred_score, [0, 5], [-1, 1]), zero, one * click_comment_mask, 'auc'),
  ]
  eval_targets = [
    ('good_quality', pred_score, good_quality, one, 'auc'),
    # ('like_label',pxtr_weights_pltr_pos, like, one, 'auc'),
    # ('follow_label',pxtr_weights_pwtr_pos, follow, one, 'auc'),
    # ('forward_label',pxtr_weights_pftr_pos, forward, one, 'auc'),
    # ('click_comment_button_label',pxtr_weights_pcmtr_pos, slide_click_comment_button, one, 'auc'),
    # ('long_view_label',pxtr_weights_plvtr_pos, long_view, one, 'auc'),
    # ('effective_view_label',pxtr_weights_pctr_pos, effective_view, one, 'auc'),
    # ('click_comment_label',tf.slice(pred_score, [0, 1], [-1, 1]), slide_click_comment_button, one, 'auc'),
    # ('long_view', tf.slice(pred_score, [0, 2], [-1, 1]), long_view, one, 'auc'),
    # ('effective_view', tf.slice(pred_score, [0, 3], [-1, 1]), effective_view, one, 'auc'),
    # ('follow', tf.slice(pred_score, [0, 4], [-1, 1]), follow, one, 'auc'),
  ]
  # targets.append(("interact_label", interact_score, label, weight, 'auc'))
  # for i, task_name in enumerate(task_names):
  #   targets.append((task_names[i], pred_score[i], labels[i], weight, 'auc'))
  #   eval_targets.append((task_names[i], pred_score[i], labels[i], one, 'auc'))

  # q_name, preds, labels, weight, auc = zip(*targets)
  # loss = tf.losses.log_loss(labels, preds, weight, reduction="weighted_sum")
  q_name, preds, labels, weight, auc = zip(*targets)
  log_loss = tf.losses.log_loss(labels, preds, weight, reduction="weighted_sum")
  integral_loss = 10000000000 / (integral_loss + 0.01)
  loss = log_loss + integral_loss
  
  
  
  print_ops = []
  my_step = config.get_step()
  print_op = tf.cond(
    tf.equal(tf.mod(my_step, 10), 0),
    lambda: tf.print(
              "loss:", loss,
              "integral_loss:", integral_loss,
              ", pred_score: ", tf.slice(pred_score, [0, 0], [10, 1]),
              ", orl_ensemble_score: ", tf.slice(orl_ensemble_score, [0, 0], [10, 1]),
              ", sorted_by_pxtr1_pxtrs",  tf.slice(sorted_by_pxtr1_pxtrs, [0, 0], [10, 1]),
              ", level_reward:", tf.slice(level_reward, [0, 0], [10, 1]),
              ", mse:", tf.square(tf.slice(pred_score, [0, 0], [10, 1]) - tf.slice(level_reward, [0, 0], [10, 1])),
              # ", ltr_alpha:", tf.slice(ltr_alpha, [0, 0], [10, 1]),
              # ", wtr_alpha:", tf.slice(wtr_alpha, [0, 0], [10, 1]),
              # ", ftr_alpha:", tf.slice(ftr_alpha, [0, 0], [10, 1]),
              # ", cmtr_alpha:", tf.slice(cmtr_alpha, [0, 0], [10, 1]),
              # ", cmef_alpha:", tf.slice(cmef_alpha, [0, 0], [10, 1]),
              # ", ctr_alpha:", tf.slice(ctr_alpha, [0, 0], [10, 1]),
              # ", lvtr_alpha:", tf.slice(lvtr_alpha, [0, 0], [10, 1]),
              # ", effective_view:", tf.slice(effective_view, [0, 0], [10, 1]),
              # ", long_view:", tf.slice(long_view, [0, 0], [10, 1]),
              # ", like:", tf.slice(like, [0, 0], [10, 1]),
              # ", comment:", tf.slice(slide_comment, [0, 0], [10, 1]),
              # ", follow:", tf.slice(follow, [0, 0], [10, 1]),
              # ", forward:", tf.slice(forward, [0, 0], [10, 1]),
              # ", click_comment_button:", tf.slice(click_comment_button, [0, 0], [10, 1]),
              # ", pxtr_weights_pltr", tf.slice(pxtr_weights, [0, 0], [20, 1]),
              # ", pxtr_weights_pwtr", tf.slice(pxtr_weights, [0, 1], [20, 1]),
              # ", pxtr_weights_pftr", tf.slice(pxtr_weights, [0, 2], [20, 1]),
              # ", pxtr_weights_pcmtr", tf.slice(pxtr_weights, [0, 3], [20, 1]),
              # ", pxtr_weights_plvtr", tf.slice(pxtr_weights, [0, 4], [20, 1]),
              # ", pxtr_weights_pctr", tf.slice(pxtr_weights, [0, 5], [20, 1]),
              # "temp_score:", tf.slice(temp_score, [0, 0], [10, 1]),
              summarize=-1,
              output_stream=sys.stdout), lambda: tf.no_op()
  )    
  print_ops.append(print_op)

  # optimizer = tf.train.AdagradOptimizer(0.2, initial_accumulator_value=0.1, name="opt")
  optimizer = tf.train.GradientDescentOptimizer(1, name="opt")
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
  num_weights = 6
  outputs = tf.split(pxtr_weights, num_weights, axis  = 1)
  for task_name, pred in zip(task_names, outputs):
    targets.append((task_name, pred))
  q_names, preds = zip(*targets)

  if args.dryrun:
    config.mock_and_profile(preds, "./predict_log/", batch_sizes=[200], compressed_embedding_size={"USER": 4})
  else:
    config.dump_predict_config("./predict/config", targets, input_type=3, extra_preds=q_names)
