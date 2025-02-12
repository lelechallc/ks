from __future__ import print_function

import os
import sys
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['train', 'predict'])
parser.add_argument('--dryrun', dest="dryrun", const=True, default=False, nargs='?')
parser.add_argument('--with_kai', action="store_true")
parser.add_argument('--text', action="store_true")
args = parser.parse_args()

if not args.dryrun and not args.with_kai:
    # monkey patch
    import mio_tensorflow.patch as mio_tensorflow_patch

    mio_tensorflow_patch.apply()

import tensorflow as tf
from mio_tensorflow.config import MioConfig

logging.basicConfig()

base_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), './base.yaml')

config = MioConfig.from_base_yaml(base_config, clear_embeddings=True, clear_params=True,
                                  dryrun=args.dryrun, label_with_kv=True, grad_no_scale=False,
                                  with_kai=args.with_kai)


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


TRAIN = args.mode == 'train'
# user feature
compress_kwargs = {}
if args.mode in ["predict", "gsu"]:
    compress_kwargs["compress_group"] = "USER"

# embeddings
uid_emb = config.new_embedding("user_emb", dim=4, slots=[101, 102], **compress_kwargs)

c_id_emb = config.new_embedding("comment_emb", dim=64, slots=[201, 202])
c_cnt_emb = config.new_embedding("comment_cnt_emb", dim=32, slots=[203, 204, 205, 209])
c_xtr_emb = config.new_embedding("comment_xtr_emb", dim=32, slots=[206, 207])
c_pos_emb = config.new_embedding("comment_pos_emb", dim=8, slots=[208])
# c_mmu_emb = config.new_embedding("comment_mmu_emb", dim=32, slots=[407, 408, 409])

# mmu_content_emb = config.get_extra_param('mmu_comment_content_emb', size=256)

field_input = tf.concat([uid_emb, c_id_emb, c_cnt_emb, c_xtr_emb, c_pos_emb], axis=-1)

expand_xtr = simple_dense_network("expand_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
like_xtr = simple_dense_network("like_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
reply_xtr = simple_dense_network("reply_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)


if args.mode == 'train':
    expand_label = config.get_label("expandaction")
    like_label = config.get_label("likeaction")
    reply_label = config.get_label("replyaction")
    sample_weight = config.get_label("sample_weight")

    # expand_loss = tf.losses.log_loss(labels=expand_label, predictions=expand_xtr, weights=sample_weight,reduction=tf.losses.Reduction.SUM)
    # like_loss = tf.losses.log_loss(labels=like_label, predictions=like_xtr, weights=sample_weight, reduction=tf.losses.Reduction.SUM)
    # reply_loss = tf.losses.log_loss(labels=reply_label, predictions=reply_xtr, weights=sample_weight, reduction=tf.losses.Reduction.SUM)
    # loss = expand_loss + like_loss + reply_loss

    targets = [
        ("expand_xtr", expand_xtr, expand_label, sample_weight, "auc"),
        ("like_xtr", like_xtr, like_label, sample_weight, "auc"),
        ("reply_xtr", reply_xtr, reply_label, sample_weight, "auc"),
    ]

    q_name, preds, labels, weights, auc = zip(*targets)
    loss = tf.losses.log_loss(labels, preds, weights, reduction="weighted_sum")

    mask_ones = tf.ones_like(sample_weight, dtype=tf.float32)
    mask_zeros = tf.zeros_like(sample_weight, dtype=tf.float32)
    comment_vv_tag = config.get_label("comment_vv_tag")

    eval_targets = [
        ("expand_xtr", expand_xtr, expand_label, sample_weight, "auc"),

        ("expand_xtr_80vv", expand_xtr, expand_label, tf.where(tf.less_equal(comment_vv_tag, 1.1), tf.where(tf.less_equal(0.0, comment_vv_tag), sample_weight, mask_zeros), mask_zeros), "auc"),
        ("expand_xtr_200vv", expand_xtr, expand_label, tf.where(tf.less_equal(comment_vv_tag, 2.1), tf.where(tf.less_equal(1.1, comment_vv_tag), sample_weight, mask_zeros), mask_zeros), "auc"),
        ("expand_xtr_500vv", expand_xtr, expand_label, tf.where(tf.less_equal(comment_vv_tag, 3.1), tf.where(tf.less_equal(2.1, comment_vv_tag), sample_weight, mask_zeros), mask_zeros), "auc"),
        ("expand_xtr_1000vv", expand_xtr, expand_label, tf.where(tf.less_equal(comment_vv_tag, 4.1), tf.where(tf.less_equal(3.0, comment_vv_tag), sample_weight, mask_zeros), mask_zeros), "auc"),
        ("expand_xtr_2000vv", expand_xtr, expand_label, tf.where(tf.less_equal(comment_vv_tag, 5.1), tf.where(tf.less_equal(4.0, comment_vv_tag), sample_weight, mask_zeros), mask_zeros), "auc"),
        ("expand_xtr_2000vv+", expand_xtr, expand_label, tf.where(tf.less_equal(comment_vv_tag, 6.1), tf.where(tf.less_equal(5.0, comment_vv_tag), sample_weight, mask_zeros), mask_zeros), "auc"),

        ("like_xtr", like_xtr, like_label, sample_weight, "auc"),
        ("like_xtr_80vv", like_xtr, like_label, tf.where(tf.less_equal(comment_vv_tag, 1.1), tf.where(tf.less_equal(0.0, comment_vv_tag), sample_weight, mask_zeros), mask_zeros), "auc"),
        ("like_xtr_200vv", like_xtr, like_label, tf.where(tf.less_equal(comment_vv_tag, 2.1), tf.where(tf.less_equal(1.1, comment_vv_tag), sample_weight, mask_zeros), mask_zeros), "auc"),
        ("like_xtr_500vv", like_xtr, like_label, tf.where(tf.less_equal(comment_vv_tag, 3.1), tf.where(tf.less_equal(2.1, comment_vv_tag), sample_weight, mask_zeros), mask_zeros), "auc"),
        ("like_xtr_1000vv", like_xtr, like_label, tf.where(tf.less_equal(comment_vv_tag, 4.1), tf.where(tf.less_equal(3.1, comment_vv_tag), sample_weight, mask_zeros), mask_zeros), "auc"),
        ("like_xtr_2000vv", like_xtr, like_label, tf.where(tf.less_equal(comment_vv_tag, 5.1), tf.where(tf.less_equal(4.1, comment_vv_tag), sample_weight, mask_zeros), mask_zeros), "auc"),
        ("like_xtr_2000vv+", like_xtr, like_label, tf.where(tf.less_equal(comment_vv_tag, 6.1), tf.where(tf.less_equal(5.1, comment_vv_tag), sample_weight, mask_zeros), mask_zeros), "auc"),

        ("reply_xtr", reply_xtr, reply_label, sample_weight, "auc"),
        ("reply_xtr_80vv", reply_xtr, reply_label, tf.where(tf.less_equal(comment_vv_tag, 1.1), tf.where(tf.less_equal(0.0, comment_vv_tag), sample_weight, mask_zeros), mask_zeros), "auc"),
        ("reply_xtr_200vv", reply_xtr, reply_label, tf.where(tf.less_equal(comment_vv_tag, 2.1), tf.where(tf.less_equal(1.1, comment_vv_tag), sample_weight, mask_zeros), mask_zeros), "auc"),
        ("reply_xtr_500vv", reply_xtr, reply_label, tf.where(tf.less_equal(comment_vv_tag, 3.1), tf.where(tf.less_equal(2.1, comment_vv_tag), sample_weight, mask_zeros), mask_zeros), "auc"),
        ("reply_xtr_1000vv", reply_xtr, reply_label, tf.where(tf.less_equal(comment_vv_tag, 4.1), tf.where(tf.less_equal(3.1, comment_vv_tag), sample_weight, mask_zeros), mask_zeros), "auc"),
        ("reply_xtr_2000vv", reply_xtr, reply_label, tf.where(tf.less_equal(comment_vv_tag, 5.1), tf.where(tf.less_equal(4.1, comment_vv_tag), sample_weight, mask_zeros), mask_zeros), "auc"),
        ("reply_xtr_2000vv+", reply_xtr, reply_label, tf.where(tf.less_equal(comment_vv_tag, 6.1), tf.where(tf.less_equal(5.1, comment_vv_tag), sample_weight, mask_zeros), mask_zeros), "auc"),
    ]

    mmu_predict_reply_score = config.get_label("predict_reply_score")
    mmu_quality_v2_score = config.get_label("quality_v2_score")
    mmu_predict_like_score = config.get_label("predict_like_score")

    print_ops = []
    my_step = config.get_step()
    print_op = tf.cond(
        tf.equal(tf.mod(my_step, 100), 0),
        lambda: tf.print(
            # "\n mmu_emb: ", tf.slice(mmu_content_emb, [0, 0], [1, -1]),
            # "\n mmu_score_emb: ", tf.slice(c_mmu_emb, [0, 0], [1, -1]),
            "\n mmu_predict_reply_score[mean min max]: ", [tf.reduce_mean(mmu_predict_reply_score), tf.reduce_min(mmu_predict_reply_score), tf.reduce_max(mmu_predict_reply_score)],
            "\n mmu_quality_v2_score[mean min max]: ", [tf.reduce_mean(mmu_quality_v2_score), tf.reduce_min(mmu_quality_v2_score), tf.reduce_max(mmu_quality_v2_score)],
            "\n mmu_predict_like_score[mean min max]: ", [tf.reduce_mean(mmu_predict_like_score), tf.reduce_min(mmu_predict_like_score), tf.reduce_max(mmu_predict_like_score)],
            summarize=-1,
            output_stream=sys.stdout), lambda: tf.no_op()
    )
    print_ops.append(print_op)


    optimizer = tf.train.GradientDescentOptimizer(0.5, name="opt")
    opt = optimizer.minimize(loss)

    if args.with_kai:  # 同步
        config.dump_kai_training_config('./training/conf', eval_targets, loss=loss, text=args.text, extra_ops=print_ops,
                                        init_params_in_tf=True)
    else:  # 异步
        optimizer = tf.train.GradientDescentOptimizer(1, name="opt")
        opt = optimizer.minimize(loss)
        config.dump_training_config('./training/conf', eval_targets, opts=[opt, print_op], text=args.text)
else:
    targets = [
        ("expand_xtr", expand_xtr),
        ("like_xtr", like_xtr),
        ("reply_xtr", reply_xtr)
    ]
    q_names, preds = zip(*targets)
    config.dump_predict_config('./predict/config', targets, input_type=3, extra_preds=q_names)
