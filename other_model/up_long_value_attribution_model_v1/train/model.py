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

def batch_group_fm_quadratic(fm_input):
  summed_emb = tf.reduce_sum(fm_input, axis=1)
  summed_emb_squared = tf.square(summed_emb)
  squared_emb = tf.square(fm_input)
  squared_sum_emb = tf.reduce_sum(squared_emb, 1)
  fm_out = 0.5 * tf.subtract(summed_emb_squared, squared_sum_emb)
  return fm_out

def exp_and_sigmoid_act(max_value):
  def func(x):
    return tf.minimum(tf.math.exp(x), max_value), tf.math.sigmoid(x)
  return func

def mlp(name, inputs, units, dropouts=None, activation=tf.nn.leaky_relu):
  with tf.variable_scope(f'{name}_mlp', reuse=tf.AUTO_REUSE):
    output = inputs
    for i in range(len(units)):
      output = tf.layers.dense(output, units[i], activation=activation,
                              kernel_initializer=tf.glorot_uniform_initializer())
      if dropouts is not None and dropouts[i] > 0: 
        output = tf.layers.dropout(output, dropouts[i], training=(args.mode == 'train'))
      # output = config.batch_norm(output, name+f"_bn_{i}")
    return output
  
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

def mlp_stop_grad(name, inputs, units, dropout, activation=tf.nn.leaky_relu):
  with tf.variable_scope(f'{name}_tower', reuse=tf.AUTO_REUSE):
    output = tf.stop_gradient(inputs)
    for i in range(len(units)):
      output = tf.layers.dense(output, units[i], activation=activation,
                              kernel_initializer=tf.glorot_uniform_initializer())
      # output = config.batch_norm(output, name+f"_bn_{i}")
    return output

def attention(name, input1, input2):
  # input1 : [None K]
  # input2 : [None K]
  dim = input1.get_shape().as_list()[-1]
  inputs = tf.concat([input1[:, None, :], input2[:, None, :]], axis=1) # [B 2 dim]
  # Q = tf.get_variable(name + "q_trans_matrix", (dim, dim))
  # K = tf.get_variable(name + "k_trans_matrix", (dim, dim))
  # V = tf.get_variable(name + "v_trans_matrix", (dim, dim))
  w1 = tf.get_variable(f'{name}_attention_w1', (dim, dim))
  w2 = tf.get_variable(f'{name}_attention_w2', (dim, dim))
  w3 = tf.get_variable(f'{name}_attention_w3', (dim, dim))
  
  Q = tf.tensordot(inputs, w1, axes=1) # [B 2 dim]
  K = tf.tensordot(inputs, w2, axes=1)
  V = tf.tensordot(inputs, w3, axes=1)
  # (N,L)
  a = tf.reduce_sum(tf.multiply(Q, K), axis=-1) / \
      tf.sqrt(tf.cast(inputs.shape[-1], tf.float32))
  a = tf.nn.softmax(a, axis=1) # [B 2]
  # (N,L,K)
  outputs = tf.multiply(a[:, :, None], V)
  return tf.reduce_sum(outputs, axis=1)  # (N, K)

def mmoe_layer(inputs, num_tasks, num_experts, expert_units, name, expert_act=tf.nn.relu, gate_act=tf.nn.softmax):
  expert_outputs, final_outputs = [], []
  for i in range(num_experts):
    # expert里是否需要 dropout
    expert_layer = mlp(f'expert_{name}_{i}', inputs, expert_units, dropouts=[0]*len(expert_units),
                        activation=expert_act)
    expert_outputs.append(expert_layer)
  expert_outputs = tf.stack(expert_outputs, axis=1)  # [batch_size, num_experts, expert_units[-1])

  for i in range(num_tasks):
    gate_layer = mlp(f'gate_{name}_{i}', inputs, [num_experts], activation=gate_act)
    gate_layer = tf.reshape(gate_layer, [-1, 1, num_experts])
    # gate_layer [batch, 1, num_experts]
    # weighted_expert_output = gate_layer @ expert_outputs
    weighted_expert_output = tf.matmul(gate_layer, expert_outputs)
    final_outputs.append(tf.reshape(weighted_expert_output, [-1, expert_units[-1]]))

  final_outputs = tf.stack(final_outputs, axis=0) # (num_tasks, batch_size, expert_units[-1])

  final_outputs = tf.transpose(final_outputs, perm=[1, 0, 2]) # (batch_size, num_tasks, expert_units[-1])
  return final_outputs

def cross_net_v2(input_layer, layer_num, name):
  embedding_layer_l = input_layer # [bs,d]
  input_size = input_layer.shape[1].value
  for i in range(layer_num):
    index = i + 1
    kernel = tf.get_variable(name="{0}_kernerl_{1}".format(name, index), shape=[input_size, input_size])
    bias = tf.get_variable(name="{0}_bias_{1}".format(name, index), shape=[input_size])
    embedding_layer_l_w = tf.matmul(embedding_layer_l, kernel)
    embedding_layer_l_b = tf.add(input_layer, bias)
    embedding_layer_l = tf.multiply(embedding_layer_l_b, embedding_layer_l_w)
#   output_w = tf.get_variable(name="{0}_output_w".format(name), shape=[input_size, output_size])
#   output_layer = tf.matmul(embedding_layer_l, output_w)
  return embedding_layer_l

def dot_product_attention_layer_v2(photo_list, query, attn_size, scope):
    
  query_dim = query.get_shape().as_list()[1]
  seq_len = photo_list.get_shape().as_list()[1]
  key_masks = 1 - tf.sign(tf.reduce_sum(tf.abs(photo_list), axis=2))  # [b, L]

  # Q = tf.get_variable(name + '_q_trans', (query_dim, attn_size))
  # K = tf.get_variable(name + '_k_trans', (key_dim, attn_size))
  # V = tf.get_variable(name + '_v_trans', (key_dim, query_dim))
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    query_trans = tf.layers.dense(query, attn_size, use_bias=True) # (batch_size attn_size)
    keys = tf.layers.dense(photo_list, attn_size, use_bias=True) # (batch_size seq_len attn_size)
    values = tf.layers.dense(photo_list, query_dim, use_bias=True) # (batch_size seq_len query_dim)

    # query_trans = tf.tensordot(query, Q, axes = 1) # (batch_size attn_size)
    # keys = tf.tensordot(photo_list, K, axes = 1)  # (batch_size seq_len attn_size)
    # values = tf.tensordot(photo_list, V, axes = 1) # (batch_size seq_len query_dim)

    key_trans = tf.transpose(keys, perm = [1, 0, 2]) # (seq_len, batch_size, attn_size)

    qk_dot = tf.multiply(key_trans, query_trans) / tf.sqrt(tf.cast(attn_size, dtype=tf.float32)) # (seq_len, batch_size, attn_size)

    qk_dot_reduce = tf.transpose(tf.reduce_sum(qk_dot, axis = 2), perm = [1, 0]) # (batch_size, seq_len)
    
    padding_num = -2 ** 32 + 1
    qk_dot_reduce = qk_dot_reduce + key_masks * padding_num
    
    qk_dot_softmax = tf.nn.softmax(qk_dot_reduce, axis = 1) # (batch_size, seq_len)
    qk_dot_softmax = tf.reshape(qk_dot_softmax, [-1, seq_len, 1])
    
    attention_val = tf.multiply(values, qk_dot_softmax) # (batch_size, seq_len, query_dim)

    attention_res = tf.reduce_sum(attention_val, axis = 1)

    return attention_res

def dot_product_attention_layer(name, photo_list, key_dim, query, query_dim, attn_size):

  batch_size =  tf.shape(query)[0]
  seq_len = tf.shape(photo_list)[1]

  # 这里也可以用 一层mlp实现 K V 相同
  Q = tf.get_variable(name + '_q_trans', (query_dim, attn_size))
  K = tf.get_variable(name + '_k_trans', (key_dim, attn_size))
  V = tf.get_variable(name + '_v_trans', (key_dim, query_dim))

  query_trans = tf.tensordot(query, Q, axes = 1) # (batch_size attn_size)
  keys = tf.tensordot(photo_list, K, axes = 1)  # (batch_size seq_len attn_size)
  values = tf.tensordot(photo_list, V, axes = 1)

  key_trans = tf.transpose(keys, perm = [1, 0, 2]) # (seq_len, batch_size, attn_size)

  qk_dot = tf.multiply(key_trans, query_trans) / tf.sqrt(tf.cast(attn_size, dtype=tf.float32)) # (seq_len, batch_size, attn_size)

  qk_dot_reduce = tf.transpose(tf.reduce_sum(qk_dot, axis = 2), perm = [1, 0]) # (batch_size, seq_len)
  qk_dot_softmax = tf.nn.softmax(qk_dot_reduce, axis = 1) # (batch_size, seq_len)

  attention_val = tf.multiply(values, tf.reshape(qk_dot_softmax, (batch_size, seq_len, 1))) # (batch_size, seq_len, query_dim)

  attention_res = tf.reduce_sum(attention_val, axis = 1)

  return attention_res

# Model Input
###################################################################

training = args.mode == 'train'
# user feature
compress_kwargs = {}
if args.mode in ["predict", "gsu"]:
    compress_kwargs["compress_group"] = "USER"

uid_emb = config.new_embedding("uid_emb", dim=64, slots=[38, 34], **compress_kwargs)

author_emb = config.new_embedding("author_emb", dim=64, slots=[128, 519, 1142])

#pid_emb = config.new_embedding("pid_emb", dim=64, slots=[26])
pid_stat = config.new_embedding("pid_stat", dim=16, slots=[185, 685, 686, 141])
pid_gate = config.new_embedding("pid_gate", dim=16, slots=[786, 787])
pid_pxtr = config.new_embedding("pid_pxtr", dim=16, slots=[1101, 1102, 1103, 1104, 1105, 1107, 1108, 1109, 1110, 1111, 1112])

author_fea = config.new_embedding("author_fea", dim=16, slots=[1200, 1201, 1202, 1203])
feas = ["author_healthiness"]
author_float_fea = [config.get_extra_param(fea, size=1) for fea in feas]
author_float_fea = tf.concat(author_float_fea, axis=1)
author_fea = tf.concat([author_fea, author_float_fea], axis=1)


uid_click_seq_embed = config.new_embedding(f'uid_action_list_click', dim=64, expand=30, slots=[2201], **compress_kwargs)
uid_click_seq_embed = tf.reshape(uid_click_seq_embed, [-1, 30, 64])
#short_click_interest_emb = dot_product_attention_layer("short_click_interest_att_current", uid_click_seq_embed, 64, pid_emb, 64, 64)
short_click_interest_emb = tf.reduce_mean(uid_click_seq_embed, axis=1)

uid_like_seq_embed = config.new_embedding(f'uid_action_list_like', dim=64, expand=30, slots=[2202], **compress_kwargs)
uid_follow_seq_embed = config.new_embedding(f'uid_action_list_follow', dim=64, expand=30, slots=[2203], **compress_kwargs)
uid_like_seq_embed = tf.reshape(uid_like_seq_embed, [-1, 30, 64])
uid_follow_seq_embed = tf.reshape(uid_follow_seq_embed, [-1, 30, 64])
uid_interact_seq_embed = tf.concat([uid_like_seq_embed, uid_follow_seq_embed], axis=2)
#short_interact_interest_emb = dot_product_attention_layer("short_interact_interest_att_current", uid_interact_seq_embed, 128, pid_emb, 64, 128)
short_interact_interest_emb = tf.reduce_mean(uid_interact_seq_embed, axis=1)

input = [uid_emb, pid_gate, pid_stat, author_emb, author_fea, pid_pxtr, short_click_interest_emb, short_interact_interest_emb]
input = tf.concat(input, axis=1)

author_input = [author_emb, author_fea]
author_input = tf.concat(author_input, axis=1)

user_input = [uid_emb, short_click_interest_emb]
user_input = tf.concat(user_input, axis=1)

print("input:", input.shape)
task_names = ["long_play_value", "effective_play_value", "play_time_min", "profile_stay_time_min", "like_value", "follow_value", "forward_value",
                "comment_value", "profile_enter_value"]


num_tasks = len(task_names)
task_outs = []


with tf.name_scope('model'):
  
  long_play_value = simple_dense_network("long_play_value_layers", input,  [128, 64, 1], act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
  effective_play_value = simple_dense_network("effective_play_value_layers", input, [128, 64, 1], act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
  play_time_min = simple_dense_network("play_time_min_layers", input,  [128, 64, 1], act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
  profile_stay_time_min = simple_dense_network("profile_stay_time_min_layers", input, [128, 64, 1], act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
  like_value = simple_dense_network("like_value_layers", input, [128, 64, 1], act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
  follow_value = simple_dense_network("follow_value_layers", input, [128, 64, 1], act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
  forward_value = simple_dense_network("forward_value_layers", input, [128, 64, 1], act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
  comment_value = simple_dense_network("comment_value_layers", input, [128, 64, 1], act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
  profile_enter_value = simple_dense_network("profile_enter_value_layers", input, [128, 64, 1], act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)



if args.mode == 'train':
  labels = [config.get_label(label) for label in task_names]
  one = tf.fill(tf.shape(labels[0]), 1.0) # weight
  zero = tf.fill(tf.shape(labels[0]), 0.0) # weight
  targets = [
        ('long_play_value', long_play_value, one, labels[0], 'auc'),
        ('long_play_value_neg', long_play_value, zero, one, 'auc'),

        ('effective_play_value', effective_play_value, one, labels[1], 'auc'),
        ('effective_play_value_neg', effective_play_value, zero, one, 'auc'),

        ('play_time_min', play_time_min, one, labels[2], 'auc'),
        ('play_time_min_neg', play_time_min, zero, one, 'auc'),

        ('profile_stay_time_min', profile_stay_time_min, one, labels[3], 'auc'),
        ('profile_stay_time_min_neg', profile_stay_time_min, zero, one, 'auc'),

        ('like_value', like_value, one, labels[4], 'auc'),
        ('like_value_neg', like_value, zero, one, 'auc'),

        ('follow_value', follow_value, one, labels[5], 'auc'),
        ('follow_value_neg', follow_value, zero, one, 'auc'),

        ('forward_value', forward_value, one, labels[6], 'auc'),
        ('forward_value_neg', forward_value, zero, one, 'auc'),

        ('comment_value', comment_value, one, labels[7], 'auc'),
        ('comment_value_neg', comment_value, zero, one, 'auc'),

        ('profile_enter_value', profile_enter_value, one, labels[8], 'auc'),
        ('profile_enter_value_neg', profile_enter_value, zero, one, 'auc'),
    ]

  eval_targets = [
        ('long_play_value', long_play_value, labels[0], one, 'auc'),
        ('effective_play_value', effective_play_value, labels[1], one, 'auc'),
        ('play_time_min', play_time_min, labels[2], one, 'auc'),
        ('profile_stay_time_min', profile_stay_time_min, labels[3], one, 'auc'),
        ('like_value', like_value, labels[4], one, 'auc'),
        ('follow_value', follow_value, labels[5], one, 'auc'),
        ('forward_value', forward_value, labels[6], one, 'auc'),
        ('comment_value', comment_value, labels[7], one, 'auc'),
        ('profile_enter_value', profile_enter_value, labels[8], one, 'auc'),
    ]

  q_name, preds, labels, weights, loss_func = zip(*targets)

  total_loss = tf.losses.log_loss(labels, preds, weights, reduction="weighted_sum")


  print_ops = []
  my_step = config.get_step()
  print_op = tf.cond(
    tf.equal(tf.mod(my_step, 100), 0),
    lambda: tf.print(
              "loss:", total_loss,
              "\n long_play_value:", tf.reduce_mean(long_play_value / (1-long_play_value)),
              "\n effective_play_value:", tf.reduce_mean(effective_play_value / (1-effective_play_value)),
              "\n play_time_min:", tf.reduce_mean(play_time_min / (1-play_time_min)),
              "\n profile_stay_time_min:", tf.reduce_mean(profile_stay_time_min / (1-profile_stay_time_min)),
              "\n like_value:", tf.reduce_mean(like_value / (1-like_value)),
              "\n follow_value:", tf.reduce_mean(follow_value / (1-follow_value)),
              "\n forward_value:", tf.reduce_mean(forward_value / (1-forward_value)),
              "\n comment_value:", tf.reduce_mean(comment_value / (1-comment_value)),
              "\n profile_enter_value:", tf.reduce_mean(profile_enter_value / (1-profile_enter_value)),
              summarize=-1,
              output_stream=sys.stdout), lambda: tf.no_op()
  )    
  print_ops.append(print_op)

  with tf.control_dependencies(print_ops):
        if args.with_kai_v2:
            config.set_slot_param_attr([34, 38, 128], config.nn.ParamAttr(access_method=config.nn.ProbabilityAccess(100.0),
                                                       recycle_method=config.nn.UnseendaysRecycle(delete_after_unseen_days=30, delete_threshold=2.0, allow_dynamic_delete=False)))
            sparse_optimizer = config.optimizer.Adam(0.05)
            dense_optimizer = config.optimizer.Adam(0.001)
            sparse_optimizer.minimize(total_loss, var_list=config.get_collection(config.GraphKeys.EMBEDDING_INPUT))
            dense_optimizer.minimize(total_loss, var_list=config.get_collection(config.GraphKeys.TRAINABLE_VARIABLES))
        else:
            optimizer = tf.train.GradientDescentOptimizer(1, name="opt")
            opt = optimizer.minimize(total_loss)

  if args.dryrun:
      config.mock_and_profile(tf.print(tf.shape(long_play_value)), './training_log/', batch_sizes=[128, 288])
  elif args.with_kai:
      config.dump_kai_training_config('./training/conf', eval_targets, loss=total_loss, text=args.text, extra_ops=print_ops)
  elif args.with_kai_v2:
      config.build_model(optimizer=[sparse_optimizer, dense_optimizer], metrics=eval_targets)
  else:
      config.dump_training_config('./training/conf', eval_targets, opts=[opt, print_ops], text=args.text)

else:
  targets = [
        ('long_play_value', long_play_value),
        ('effective_play_value', effective_play_value),
        ('play_time_min', play_time_min),
        ('profile_stay_time_min', profile_stay_time_min),
        ('like_value', like_value),
        ('follow_value', follow_value),
        ('forward_value', forward_value),
        ('comment_value', comment_value),
        ('profile_enter_value', profile_enter_value),
    ]
  q_names, preds = zip(*targets)

  if args.dryrun:
    config.mock_and_profile(preds, "./predict_log/", batch_sizes=[200], compressed_embedding_size={"USER": 4})
  else:
    config.dump_predict_config("./predict/config", targets, input_type=3, extra_preds=q_names)
