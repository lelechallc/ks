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

    user_embedding = kai.nn.new_embedding("user_embedding", dim=4, slots=[101, 102])
    comment_id_embedding = kai.nn.new_embedding("c_id_embedding", dim=64, slots=[201])
    comment_info_embedding = kai.nn.new_embedding("c_info_embedding", dim=32, slots=[203, 204, 205, 206, 207, 209])
    position_embedding = kai.nn.new_embedding("position_embedding", dim=8, slots=[208])
    pid_emb = kai.nn.new_embedding("pid_emb", dim=64, slots=[103])
    aid_emb = kai.nn.new_embedding("aid_emb", dim=64, slots=[202])
    uid_emb = kai.nn.new_embedding("uid_emb", dim=64, slots=[105])
    photo_aid_emb = kai.nn.new_embedding("photo_aid_emb", dim=64, slots=[104])
    did_emb = kai.nn.new_embedding("did_emb", dim=64, slots=[106])

    # context_embedding1 = kai.nn.new_embedding("context_embedding1", dim=32, slots=[114])    # city
    # context_embedding2 = kai.nn.new_embedding("context_embedding2", dim=8, slots=[115, 116])  # request_hour, request_day

    comment_content_segs0 = kai.nn.new_embedding("comment_content_segs", dim=64, expand=SEQ_LEN, slots=[300])    # [-1, 1280]  
    comment_content_segs = tf.reshape(comment_content_segs0, [-1, SEQ_LEN, 64])     

    seq_pos_embedding0 = kai.nn.new_embedding("seq_pos_embedding", dim=64, expand=SEQ_LEN, slots=[301])   # [-1, 1280] 
    seq_pos_embedding = tf.reshape(seq_pos_embedding0, [-1, SEQ_LEN, 64])

    cmt_token_emb_seq = comment_content_segs + seq_pos_embedding

    final_seg_num = kai.nn.get_dense_fea('final_seg_num', dim=1, dtype=tf.float32)    # [-1, 1]
    padding_input = kai.nn.get_dense_fea("mask_pack", dim=SEQ_LEN, dtype=tf.float32)  # [-1, seq]

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
    comment_id_embedding = config.new_embedding("c_id_embedding", dim=64, slots=[201])
    comment_info_embedding = config.new_embedding("c_info_embedding", dim=32, slots=[203, 204, 205, 206, 207, 209])
    position_embedding = config.new_embedding("position_embedding", dim=8, slots=[208])

    pid_emb = config.new_embedding("pid_emb", dim=64, slots=[103])
    aid_emb = config.new_embedding("aid_emb", dim=64, slots=[202])
    uid_emb = config.new_embedding("uid_emb", dim=64, slots=[105])
    photo_aid_emb = config.new_embedding("photo_aid_emb", dim=64, slots=[104])
    did_emb = config.new_embedding("did_emb", dim=64, slots=[106])

    # context_embedding1 = config.new_embedding("context_embedding1", dim=32, slots=[114])  # city
    # context_embedding2 = config.new_embedding("context_embedding2", dim=8, slots=[115, 116])  # request_hour, request_day

    comment_content_segs0 = config.new_embedding("comment_content_segs", dim=64, expand=SEQ_LEN, slots=[300])    # [-1, 1280]  
    comment_content_segs = tf.reshape(comment_content_segs0, [-1, SEQ_LEN, 64])     

    seq_pos_embedding0 = config.new_embedding("seq_pos_embedding", dim=64, expand=SEQ_LEN, slots=[301])   # [-1, 1280] 
    seq_pos_embedding = tf.reshape(seq_pos_embedding0, [-1, SEQ_LEN, 64])

    cmt_token_emb_seq = comment_content_segs + seq_pos_embedding

    final_seg_num = config.get_dense_fea('final_seg_num', dim=1, dtype=tf.float32)    # [-1, 1]
    padding_input = config.get_dense_fea("mask_pack", dim=SEQ_LEN, dtype=tf.float32)  # [-1, seq]



def layer_norm(name, x, eps=1e-5):
    """layer normalization

    Args:
        name (str): _description_
        x (Tensor): [-1, k_seq, num_units]
        eps (float, optional): Defaults to 1e-5.

    Returns:
        output (Tensor): same shape as x
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        dim = x.shape.as_list()[-1]
        mean, var = tf.nn.moments(x, axes=[-1], keep_dims=True)
        x = (x - mean) * tf.math.rsqrt(var + eps)
        gamma = tf.get_variable(f'gamma_{name}', shape=[dim], initializer=tf.initializers.ones())
        beta = tf.get_variable(f'beta_{name}', shape=[dim], initializer=tf.initializers.ones())
        output = gamma * x + beta
        return output


def multi_head_attention(name, queries, keys, padding_mask=None, num_units=None, num_heads=8, add_embedding=None):
    """ 
    queries: [-1, q_seq, h] q_seq=k_seq
    keys:    [-1, k_seq, h]
    values:  [-1, k_seq, h]
    padding_mask: [-1, k_seq]. 是由 0 / 1 组成的 mask 的矩阵, padding位值为0, 非padding位值为1
    num_units: int. 等于 num_heads * depth
    num_heads: int
    add_embedding: Tensor, 残差连接的input_tensor

    return 
        attention: [-1, k_seq, num_units]
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]
        q_seq = tf.shape(queries)[1]
        k_seq = tf.shape(keys)[1]
        
        # 可学习参数1
        Q = tf.layers.dense(queries, num_units) # [-1, q_seq, dim]
        K = tf.layers.dense(keys, num_units)    # [-1, k_seq, dim]
        V = tf.layers.dense(keys, num_units)    # [-1, k_seq, dim]

        assert num_units % num_heads == 0
        depth = num_units // num_heads
        Q_ = tf.transpose(tf.reshape(Q, [-1, q_seq, num_heads, depth]), perm=[0, 2, 1, 3])   # [-1, num_heads, q_seq, depth]
        K_ = tf.transpose(tf.reshape(K, [-1, k_seq, num_heads, depth]), perm=[0, 2, 1, 3])   # [-1, num_heads, k_seq, depth]
        V_ = tf.transpose(tf.reshape(V, [-1, k_seq, num_heads, depth]), perm=[0, 2, 1, 3])   # [-1, num_heads, k_seq, depth]

        # [-1, num_heads, k_seq, k_seq]
        outputs = tf.matmul(Q_, K_, transpose_b=True) / tf.math.sqrt(tf.cast(depth, tf.float32))    # [-1, num_heads, q_seq, k_seq]
        
        # padding mask
        if padding_mask is not None:
            # [-1, k_seq] -> [-1, 1, k_seq, 1] -> [-1, num_heads, k_seq, k_seq]
            padding_mask = (1 - tf.tile(padding_mask[:, None, :, None], [1, num_heads, 1, k_seq])) * -1e9 # [-1, num_heads, k_seq, k_seq] 原值1转换为0，原值0转换为-1e9
            outputs = outputs + padding_mask

        outputs = tf.nn.softmax(outputs, axis=-1)   # [-1, num_heads, k_seq, k_seq]
        attention = tf.transpose(tf.matmul(outputs, V_), perm=[0, 2, 1, 3])  # [-1, k_seq, num_heads, depth]
        attention = tf.reshape(attention, [-1, k_seq, num_units])  # [-1, k_seq, num_units]

        # attention = tf.layers.dense(attention, attention.shape[-1]) # [-1, k_seq, num_units]

        # add and norm
        # 可学习参数2
        if add_embedding is not None:
            attention = layer_norm(name=f"{name}_norm", x=attention + tf.layers.dense(add_embedding, num_units))
        else:
            attention = layer_norm(name=f"{name}_norm", x=attention + K)
        return attention
    

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


attention_output = multi_head_attention("multi_head_attention", cmt_token_emb_seq, cmt_token_emb_seq, padding_input, num_units=128)   # [-1, seq, dim]
cls_output = attention_output[:, 0, :]  # [-1, dim]
cls_output = tf.layers.dense(cls_output, 64, activation=tf.nn.relu) # [-1, 64]

hadamard_uid = tf.multiply(cls_output, uid_emb)
hadamard_pid = tf.multiply(cls_output, pid_emb)
hadamard_cid = tf.multiply(cls_output, comment_id_embedding)
hadamard_aid = tf.multiply(cls_output, aid_emb)
inner_uid = tf.expand_dims(tf.reduce_sum(hadamard_uid, axis=1), 1)
inner_pid = tf.expand_dims(tf.reduce_sum(hadamard_pid, axis=1), 1)
inner_cid = tf.expand_dims(tf.reduce_sum(hadamard_cid, axis=1), 1)
inner_aid = tf.expand_dims(tf.reduce_sum(hadamard_aid, axis=1), 1)
    
# define model structure
# total size=916 (不包括context_embedding1， context_embedding2)
used_feats = [user_embedding, comment_id_embedding, comment_info_embedding, position_embedding,
                         pid_emb, aid_emb, uid_emb, photo_aid_emb, did_emb,
                         cls_output, hadamard_uid, hadamard_pid, hadamard_cid, hadamard_aid, inner_uid, inner_pid, inner_cid, inner_aid,
                        #  context_embedding1, context_embedding2
                         ]
field_input = tf.concat(used_feats, -1)
expand_xtr = simple_dense_network("expand_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
like_xtr = simple_dense_network("like_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
reply_xtr = simple_dense_network("reply_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
copy_xtr = simple_dense_network("copy_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
share_xtr = simple_dense_network("share_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
audience_xtr = simple_dense_network("audience_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
continuous_expand_xtr = simple_dense_network("continuous_expand_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
# hate_xtr = simple_dense_network("hate_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
# sub_at_xtr = simple_dense_network("sub_at_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)


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
    # like_label = tf.where(like_first_label > 0, ones, zeros)

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
        # ('hate_predict', hate_xtr, hate_label, ones, "auc"),
        # ('sub_at_predict', sub_at_xtr, sub_at_label, ones, "auc"),

    ]

    metric_name, preds, labels, weights, metric_type = zip(*targets)

    # 5. define optimizer
    loss = tf.losses.log_loss(labels, preds, weights, reduction="weighted_sum")
    optimizer = kai.nn.optimizer.Adam(1e-3)
    optimizer.minimize(loss)

    recall_type = kai.nn.get_dense_fea("recall_type", dim=1, dtype=tf.float32)

    comment_genre = kai.nn.get_dense_fea("comment_genre", dim=1, dtype=tf.float32)
    pic_comment = tf.where(comment_genre > 0, ones, zeros)
    

    eval_targets = [
        ('expand_predict', expand_xtr, expand_label, ones, "auc"),
        ('like_predict', like_xtr, like_label, ones, "auc"),
        ('reply_predict', reply_xtr, reply_label, ones, "auc"),
        ('copy_predict', copy_xtr, copy_label, ones, "auc"),
        ('share_predict', share_xtr, share_label, ones, "auc"),
        ('audience_predict', audience_xtr, audience_label, ones, "auc"),
        ('continuous_expand_predict', continuous_expand_xtr, continuous_expand_label, ones, "auc"),
        # ('hate_predict', hate_xtr, hate_label, ones, "auc"),
        # ('sub_at_predict', sub_at_xtr, sub_at_label, ones, "auc"),

        ('expand_hot', expand_xtr, expand_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('like_hot', like_xtr, like_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('reply_hot', reply_xtr, reply_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('copy_hot', copy_xtr, copy_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('share_hot', share_xtr, share_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('audience_hot', audience_xtr, audience_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('continuous_expand_hot', continuous_expand_xtr, continuous_expand_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        # ('hate_hot', hate_xtr, hate_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        # ('sub_at_hot', sub_at_xtr, sub_at_xtr, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),

        ('expand_climb', expand_xtr, expand_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('like_climb', like_xtr, like_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('reply_climb', reply_xtr, reply_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('copy_climb', copy_xtr, copy_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('share_climb', share_xtr, share_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('audience_climb', audience_xtr, audience_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('continuous_expand_climb', continuous_expand_xtr, continuous_expand_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        # ('hate_climb', hate_xtr, hate_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        # ('sub_at_climb', sub_at_xtr, sub_at_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),

        ('pic_expand_predict', expand_xtr, expand_label, pic_comment, "auc"),
        ('pic_like_predict', like_xtr, like_label, pic_comment, "auc"),
        ('pic_reply_predict', reply_xtr, reply_label, pic_comment, "auc"),
        ('pic_copy_predict', copy_xtr, copy_label, pic_comment, "auc"),
        ('pic_share_predict', share_xtr, share_label, pic_comment, "auc"),
        ('pic_audience_predict', audience_xtr, audience_label, pic_comment, "auc"),
        ('pic_continuous_expand_predict', continuous_expand_xtr, continuous_expand_label, pic_comment, "auc"),
        # ('pic_hate_predict', hate_xtr, hate_label, pic_comment, "auc"),
        # ('pic_sub_at_predict', sub_at_xtr, sub_at_label, pic_comment, "auc"),

        ('text_expand_predict', expand_xtr, expand_label,  1 - pic_comment, "auc"),
        ('text_like_predict', like_xtr, like_label, 1 - pic_comment, "auc"),
        ('text_reply_predict', reply_xtr, reply_label, 1 - pic_comment, "auc"),
        ('text_copy_predict', copy_xtr, copy_label, 1-pic_comment, "auc"),
        ('text_share_predict', share_xtr, share_label, 1-pic_comment, "auc"),
        ('text_audience_predict', audience_xtr, audience_label, 1-pic_comment, "auc"),
        ('text_continuous_expand_predict', continuous_expand_xtr, continuous_expand_label, 1-pic_comment, "auc"),
        # ('text_hate_predict', hate_xtr, hate_label, 1-pic_comment, "auc"),
        # ('text_sub_at_predict', sub_at_xtr, sub_at_label, 1-pic_comment, "auc"),
    ]

    class TensorPrintHook(kai.training.RunHookBase):
        def __init__(self, debug_tensor_map):
            self.has_print = False
            self.debug_tensor_map = debug_tensor_map

        def begin(self, stream_context):
            pass

        def before_pass_run(self, pass_run_context):
            """
            每个 pass 只会 print 一次
            """
            self.has_print = False

            total=0
            for i, feat in enumerate(self.debug_tensor_map['used_features']):
                print(f'feat_{i}: dim={feat.shape[-1]}')
                total += feat.shape[-1]
            print(f'total size={total}')

            for i, feat in enumerate(self.debug_tensor_map['debug_features']):
                print(f'debug_features_{i}: dim={feat.shape}')

        def before_step_run(self, step_run_context):
            return kai.training.StepRunArgs(fetches=self.debug_tensor_map)

        def after_step_run(self, step_run_context, step_run_values):
            if not self.has_print:
                for k, v in step_run_values.result.items():
                    if k == 'print_features':
                        print(v)
                self.has_print = True
            
    debug_tensor = {
        'used_features': used_feats,
        'debug_features': [comment_content_segs0, comment_content_segs, seq_pos_embedding0, seq_pos_embedding, final_seg_num, padding_input],
        'print_features': [comment_content_segs[:3, :6, :2], seq_pos_embedding[:3,:6,:2]]
    }
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
        # ("hate_xtr", hate_xtr),
        # ("sub_at_xtr", sub_at_xtr),
    ]
    q_names, preds = zip(*targets)
    config.dump_predict_config('./predict/config', targets, input_type=3, extra_preds=q_names)

