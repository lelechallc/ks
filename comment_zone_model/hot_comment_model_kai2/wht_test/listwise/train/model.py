""" listwise建模
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'predict'], default='train')
parser.add_argument('--dryrun', dest="dryrun", const=True, default=False, nargs='?')
parser.add_argument('--with_kai', action="store_true")
args = parser.parse_args()

SEQ_SIZE = 180    # 平均infer长度为180

if args.mode == 'train':
    import tensorflow.compat.v1 as tf
    import kai.tensorflow as kai

    print(f'tf_version: {tf.__version__}')

    # common attrs
    user_profile_emb = kai.nn.new_embedding("user_profile_emb", dim=4, slots=[101, 102])  # gender, age
    personalized_id_emb = kai.nn.new_embedding("personalized_id_emb", dim=64, slots=[103, 104, 105, 106])  # userid, photo_author_id, photo_id, device_id

    comment_id_list = []
    comment_info_list = []
    comment_content_list = []
    for slot in [201, 202]:
        comment_id_list.append(tf.reshape(kai.nn.new_embedding(f"comment_id_{slot}", dim=64, expand=SEQ_SIZE, slots=[slot]), [-1, SEQ_SIZE, 64]))

    for slot in [303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313]:
        comment_info_list.append(tf.reshape(kai.nn.new_embedding(f"comment_info_{slot}", dim=16, expand=SEQ_SIZE, slots=[slot]), [-1, SEQ_SIZE, 16]))

    for slot in [250, 251, 252, 253, 254, 255]:
        comment_content_list.append(tf.reshape(kai.nn.new_embedding(f"comment_content_{slot}", dim=8, expand=SEQ_SIZE, slots=[slot]), [-1, SEQ_SIZE, 8]))
    
    comment_id_embedding = tf.concat(comment_id_list, axis=-1)
    comment_info_embedding = tf.concat(comment_info_list, axis=-1)
    comment_content_embedding = tf.concat(comment_content_list, axis=-1)

    padding_mask_input = kai.nn.get_dense_fea("mask_pack", dim=SEQ_SIZE, dtype=tf.float32)  # [-1, seq]
    
else:
    import tensorflow as tf
    from mio_tensorflow.config import MioConfig

    print(f'tf_version: {tf.__version__}')

    if not args.dryrun and not args.with_kai:
        # monkey patch
        import mio_tensorflow.patch as mio_tensorflow_patch
        mio_tensorflow_patch.apply()
    base_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), './base.yaml')
    config = MioConfig.from_base_yaml(base_config, clear_embeddings=True, clear_params=True,
                                    dryrun=args.dryrun, label_with_kv=True, grad_no_scale=False,
                                    with_kai=args.with_kai)
    compress_kwargs = dict(compress_group="USER")

    # common attrs
    user_profile_emb = config.new_embedding("user_profile_emb", dim=4, slots=[101, 102])  # gender, age
    personalized_id_emb = config.new_embedding("personalized_id_emb", dim=64, slots=[103, 104, 105, 106])  # userid, photo_author_id, photo_id, device_id

    comment_id_embedding = config.new_embedding("c_id_embedding", dim=64, slots=[201, 202])   # [-1, dim]
    comment_info_embedding = config.new_embedding("c_info_embedding", dim=16, slots=[303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313])
    comment_content_embedding = config.new_embedding("c_content_embedding", dim=8, slots=[250, 251, 252, 253, 254, 255])
    padding_mask_input = tf.reshape(config.get_extra_param(name="mask_pack", size=1), [1, -1])  # [1, n]


def tower_module(name, inputs, units):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        x = inputs
        for i in range(len(units)):
            if i == len(units)-1:
                act = 'sigmoid'
            else:
                act = tf.nn.leaky_relu
            x = tf.layers.dense(x, units[i], activation=act, kernel_initializer=tf.glorot_uniform_initializer())
        return x
    

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
    

"""
user_profile_emb: [-1, 8]
personalized_id_emb: [-1, 256]
comment_id_embedding: [-1, max_len, 128]
comment_info_embedding: [-1, max_len, 176]
comment_content_embedding: [-1, max_len, 48]
total size=616
"""
    
# forward
if args.mode == 'train':
    attention_input = tf.concat([comment_id_embedding, comment_info_embedding, comment_content_embedding], axis=-1)  # [-1, seq, dim]
    user_input = tf.tile(tf.expand_dims(tf.concat([user_profile_emb, personalized_id_emb], axis=-1), 1), [1, SEQ_SIZE, 1])   # [-1, seq, dim]
else:
    attention_input = tf.expand_dims(tf.concat([comment_id_embedding, comment_info_embedding, comment_content_embedding], axis=-1), 0)  # [1, seq, dim]
    user_input = tf.expand_dims(tf.concat([user_profile_emb, personalized_id_emb], axis=-1), 0)  # [1, seq, dim]


attention_output = multi_head_attention("multi_head_attention", attention_input, attention_input, padding_mask_input, num_units=128)   # [-1, seq, dim]
field_input = tf.concat([user_input, attention_output], axis=-1)

expand_seq_xtr = tower_module("expand", field_input, [128, 64, 32, 1])   #[-1, seq, 1]
like_seq_xtr = tower_module("like", field_input, [128, 64, 32, 1])
reply_seq_xtr = tower_module("reply", field_input, [128, 64, 32, 1])
continuous_expand_seq_xtr = tower_module("continuous_expand", field_input, [128, 64, 32, 1])
copy_seq_xtr = tower_module("copy", field_input, [128, 64, 32, 1])
share_seq_xtr = tower_module("share", field_input, [128, 64, 32, 1])
audience_seq_xtr = tower_module("audience", field_input, [128, 64, 32, 1])

expand_xtr = tf.squeeze(expand_seq_xtr, [-1])   # [-1, seq]
like_xtr = tf.squeeze(like_seq_xtr, [-1])
reply_xtr = tf.squeeze(reply_seq_xtr, [-1])
continuous_expand_xtr = tf.squeeze(continuous_expand_seq_xtr, [-1])
copy_xtr = tf.squeeze(copy_seq_xtr, [-1])
share_xtr = tf.squeeze(share_seq_xtr, [-1])
audience_xtr = tf.squeeze(audience_seq_xtr, [-1])



if args.mode == 'train':
    # # define label input and define metrics
    ## 注意：使用kafka数据流时，不需要使用tf.cast，直接读取tf.float32数据。例如 sample_weight = kai.nn.get_dense_fea("sample_weight", dim=1, dtype=tf.float32)

    expand_label = kai.nn.get_dense_fea("expand_label_list", dim=SEQ_SIZE, dtype=tf.float32)   # [-1, seq]
    like_label = kai.nn.get_dense_fea("like_label_list", dim=SEQ_SIZE, dtype=tf.float32)
    reply_label = kai.nn.get_dense_fea("reply_label_list", dim=SEQ_SIZE, dtype=tf.float32)
    continuous_expand_label = kai.nn.get_dense_fea("continuous_expand_label_list", dim=SEQ_SIZE, dtype=tf.float32)
    copy_label = kai.nn.get_dense_fea("copy_label_list", dim=SEQ_SIZE, dtype=tf.float32)
    share_label = kai.nn.get_dense_fea("share_label_list", dim=SEQ_SIZE, dtype=tf.float32)
    audience_label = kai.nn.get_dense_fea("audience_label_list", dim=SEQ_SIZE, dtype=tf.float32)

    ones = tf.ones_like(expand_label, dtype=tf.float32)
    zeros = tf.zeros_like(expand_label, dtype=tf.float32)

    # recall_type = kai.nn.get_dense_fea("recall_type", dim=1, dtype=tf.float32)

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

    # eval_targets = [
    #     ('expand_predict', expand_xtr, expand_label, ones, "auc"),
    #     ('like_predict', like_xtr, like_label, ones, "auc"),
    #     ('reply_predict', reply_xtr, reply_label, ones, "auc"),
    #     ('copy_predict', copy_xtr, copy_label, ones, "auc"),
    #     ('share_predict', share_xtr, share_label, ones, "auc"),
    #     ('audience_predict', audience_xtr, audience_label, ones, "auc"),
    #     ('continuous_expand_predict', continuous_expand_xtr, continuous_expand_label, ones, "auc"),

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

    #     ('pic_expand_predict', expand_xtr, expand_label, pic_comment, "auc"),
    #     ('pic_like_predict', like_xtr, like_label, pic_comment, "auc"),
    #     ('pic_reply_predict', reply_xtr, reply_label, pic_comment, "auc"),
    #     ('pic_copy_predict', copy_xtr, copy_label, pic_comment, "auc"),
    #     ('pic_share_predict', share_xtr, share_label, pic_comment, "auc"),
    #     ('pic_audience_predict', audience_xtr, audience_label, pic_comment, "auc"),
    #     ('pic_continuous_expand_predict', continuous_expand_xtr, continuous_expand_label, pic_comment, "auc"),

    #     ('text_expand_predict', expand_xtr, expand_label,  1 - pic_comment, "auc"),
    #     ('text_like_predict', like_xtr, like_label, 1 - pic_comment, "auc"),
    #     ('text_reply_predict', reply_xtr, reply_label, 1 - pic_comment, "auc"),
    #     ('text_copy_predict', copy_xtr, copy_label, 1-pic_comment, "auc"),
    #     ('text_share_predict', share_xtr, share_label, 1-pic_comment, "auc"),
    #     ('text_audience_predict', audience_xtr, audience_label, 1-pic_comment, "auc"),
    #     ('text_continuous_expand_predict', continuous_expand_xtr, continuous_expand_label, 1-pic_comment, "auc"),
    # ]


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
                print(f'feat_{i}: dim={feat.shape[-1]}, shape={feat.shape}')
                total += feat.shape[-1]
            print(f'total size={total}')

        def before_step_run(self, step_run_context):
            return kai.training.StepRunArgs(fetches=self.debug_tensor_map)

        def after_step_run(self, step_run_context, step_run_values):
            if not self.has_print:
                for k, v in step_run_values.result.items():
                    if k in ['cids', 'pids', 'expand_label']:
                        print(f"{k} = {v}")
                self.has_print = True
    

    used_features = [user_profile_emb, personalized_id_emb, comment_id_embedding, comment_info_embedding, comment_content_embedding,
                     expand_label, like_label, reply_label, expand_xtr, like_xtr, reply_xtr]
    cids = tf.cast(kai.nn.get_dense_fea("comment_id_list", dtype=tf.float32, dim=SEQ_SIZE), tf.int64)
    pids = tf.cast(kai.nn.get_dense_fea("photo_id", dtype=tf.float32, dim=SEQ_SIZE), tf.int64)
    debug_tensor = {
        'used_features': used_features,
        # 'cids': tf.slice(cids, [0, 0], [3, -1]),
        'cids': tf.slice(comment_id_list, [0, 0, 0], [3, 3, 1]),
        'pids': tf.slice(pids, [0, 0], [3, -1]),
        'expand_label': tf.slice(expand_label, [0, 0], [3, -1]),
    }
    kai.add_run_hook(TensorPrintHook(debug_tensor), "debug_tensor_hook")


    # 6. finish define model structure 
    kai.build_model(optimizer=[optimizer], metrics=targets)
else:
    targets = [
        ("expand_xtr", tf.reshape(expand_xtr, [-1, 1])),
        ("like_xtr", tf.reshape(like_xtr, [-1, 1])),
        ("reply_xtr", tf.reshape(reply_xtr, [-1, 1])),
        ("copy_xtr", tf.reshape(copy_xtr, [-1, 1])),
        ("share_xtr", tf.reshape(share_xtr, [-1, 1])),
        ("audience_xtr", tf.reshape(audience_xtr, [-1, 1])),
        ("continuous_expand_xtr", tf.reshape(continuous_expand_xtr, [-1, 1])),
    ]
    q_names, preds = zip(*targets)
    config.dump_predict_config('./predict/config', targets, input_type=3, extra_preds=q_names)
