""" 在v2基础上，增加单评论停留时长信号。模型结构采用单层PLE，并且包含曝光位置特征！
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'predict', 'eval'], default='eval')
parser.add_argument('--dryrun', dest="dryrun", const=True, default=False, nargs='?')
parser.add_argument('--with_kai', action="store_true")
args = parser.parse_args()


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


def tower_module(name, inputs, units, last_act='sigmoid'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        x = inputs
        for i in range(len(units)):
            if i == len(units)-1:
                act = last_act
            else:
                act = tf.nn.leaky_relu
            x = tf.layers.dense(x, units[i], activation=act, kernel_initializer=tf.glorot_uniform_initializer())
        return x
    

def expert_module(name, inputs, units, dropout=0):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        x = inputs
        for i in range(len(units)):
            if dropout>0:
                x = tf.layers.dropout(x, dropout, training=True)
            x = tf.layers.dense(x, units[i], activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer())
        return x


def gate_module(name, inputs, task_expert_outputs, shared_expert_outputs):
    """
    args
        inputs: (b, h)
        task_expert_outputs: list of (b, h)
        shared_expert_outputs: list of (b, h)
    return
        gate_outputs: list of (b, h)
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        gate_outputs = []
        for i in range(len(task_expert_outputs)):
            selected_matrix = tf.stack([task_expert_outputs[i]]+shared_expert_outputs, axis=1)     # (b, n, h)
            logits = tf.layers.dense(inputs, selected_matrix.shape[1], activation=None, kernel_initializer=tf.glorot_uniform_initializer(), name=f'gate_{i}')    # (b, n)
            weights = tf.expand_dims(tf.nn.softmax(logits, axis=-1), axis=-1)      # (b,n,1)
            weighted = weights*selected_matrix   # (b,n,h)
            task_output = tf.reduce_sum(weighted, axis=1)   # (b, h)
            gate_outputs.append(task_output)
        return gate_outputs


def main_model(inputs, tower_names=['expand', 'like', 'reply'], tower_units=[64, 1], tower_last_act={}, 
            num_shared_experts=3, expert_units=[256, 128], dropout=0, training=False,
            bias_tower_inputs={}, bias_tower_units=[128, 64, 1], bias_tower_last_act={},
            ):
    """ 实现单层简化版PLE模型(CGC layer) """
    with tf.variable_scope('main_model', reuse=tf.AUTO_REUSE):   
        if not training:
            dropout=0
        # experts
        task_expert_outputs=[]
        shared_expert_outputs=[]
        for task_name in tower_names:
            task_expert_outputs.append(expert_module(f'expert_{task_name}', inputs, expert_units, dropout))
        for i in range(num_shared_experts):
            shared_expert_outputs.append(expert_module(f'expert_shared_{i}', inputs, expert_units, dropout))
        
        # gates
        gate_outputs=gate_module('gate_model', inputs, task_expert_outputs, shared_expert_outputs)  # list of (b,h)

        # towers
        outputs=[]
        if dropout>0:
            tower_inputs = [tf.layers.dropout(x, dropout, training=True) for x in gate_outputs]
        else:
            tower_inputs = gate_outputs
        for i, tower_name in enumerate(tower_names):
            output = tower_module(tower_name, tower_inputs[i], tower_units, last_act=tower_last_act.get(tower_name, 'sigmoid'))
            if tower_name in bias_tower_inputs:
                bias_output = tower_module(f'bias_{tower_name}', bias_tower_inputs[tower_name], bias_tower_units, last_act=bias_tower_last_act.get(tower_name, 'sigmoid'))
                if training:
                    output = output+bias_output
            outputs.append(output)
        return outputs

    
# define model structure
field_input = tf.concat([user_embedding, comment_id_embedding, comment_info_embedding, position_embedding, pid_embedding, aid_embedding, uid_embedding, did_embedding, context_embedding, comment_genre_embedding, comment_length_embedding], -1)
# input_wo_pos = tf.concat([user_embedding, comment_id_embedding, comment_info_embedding, pid_embedding, aid_embedding, uid_embedding, did_embedding, context_embedding, comment_genre_embedding, comment_length_embedding], -1)
input_did_pos = tf.concat([did_embedding, position_embedding, context_embedding], -1)
# task_names = ['expand_xtr', 'like_first_xtr', 'like_second_xtr', 'reply_first_xtr', 'reply_second_xtr', 'copy_xtr', 'share_xtr', 'audience_xtr', 'continuous_expand_xtr', 'hate_xtr', 'sub_at_xtr', 'duration_predict']
task_names = ['expand_xtr', 'like_xtr', 'reply_xtr', 'copy_xtr', 'share_xtr', 'audience_xtr', 'continuous_expand_xtr', 'duration_predict']

expand_xtr, like_xtr, reply_xtr, copy_xtr, share_xtr, audience_xtr, \
    continuous_expand_xtr, duration_predict= main_model(
    field_input, task_names, tower_units=[64, 1], tower_last_act={'duration_predict': 'relu'},
    num_shared_experts=3, expert_units=[256, 128], dropout=0.3, training=args.mode=='train', 
)

if args.mode == 'train' or args.mode == 'eval':

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
    loss_duration = tf.losses.huber_loss(duration_label, duration_predict, weights=1.0, delta=3.0)
    loss = loss + loss_duration
    optimizer = kai.nn.optimizer.Adam(1e-3)
    optimizer.minimize(loss)

    eval_targets = [
        ('expand_predict', expand_xtr, expand_label, ones, "auc"),
        ('like_predict', like_xtr, like_label, ones, "auc"),
        ('reply_predict', reply_xtr, reply_label, ones, "auc"),
        ('copy_predict', copy_xtr, copy_label, ones, "auc"),
        ('share_predict', share_xtr, share_label, ones, "auc"),
        ('audience_predict', audience_xtr, audience_label, ones, "auc"),
        ('continuous_expand_predict', continuous_expand_xtr, continuous_expand_label, ones, "auc"),
        ('duration_predict', duration_predict, duration_label, ones, 'linear_regression'),

        ('expand_hot', expand_xtr, expand_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('like_hot', like_xtr, like_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('reply_hot', reply_xtr, reply_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('copy_hot', copy_xtr, copy_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('share_hot', share_xtr, share_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('audience_hot', audience_xtr, audience_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('continuous_expand_hot', continuous_expand_xtr, continuous_expand_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('duration_hot', duration_predict, duration_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "linear_regression"),

        ('expand_climb', expand_xtr, expand_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('like_climb', like_xtr, like_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('reply_climb', reply_xtr, reply_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('copy_climb', copy_xtr, copy_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('share_climb', share_xtr, share_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('audience_climb', audience_xtr, audience_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('continuous_expand_climb', continuous_expand_xtr, continuous_expand_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('duration_climb', duration_predict, duration_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "linear_regression"),

    ]

    # 6. finish define model structure 
    kai.build_model(optimizer=[optimizer], metrics=eval_targets)
else:
    targets = [
        ("expand_xtr", expand_xtr),
        ("like_xtr", like_xtr),
        # ("like_second_xtr", like_second_xtr),
        ("reply_xtr", reply_xtr),
        # ("reply_second_xtr", reply_second_xtr),
        ("copy_xtr", copy_xtr),
        ("share_xtr", share_xtr),
        ("audience_xtr", audience_xtr),
        ("continuous_expand_xtr", continuous_expand_xtr),
        # ("hate_xtr", hate_xtr),
        # ("like_xtr", 1-(1-like_first_xtr)*(1-like_second_xtr)),
        # ("reply_xtr", 1-(1-reply_first_xtr)*(1-reply_second_xtr)),
        # ("sub_at_xtr", sub_at_xtr),
        ("duration_predict", duration_predict),
    ]
    q_names, preds = zip(*targets)
    config.dump_predict_config('./predict/config', targets, input_type=3, extra_preds=q_names)
