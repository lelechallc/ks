"""
个性化 v2 基础上增加 fusion 层
"""

import os
import argparse
import logging


parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'predict'], default='train')
parser.add_argument('--dryrun', dest="dryrun", const=True, default=False, nargs='?')
parser.add_argument('--with_kai', action="store_true")
args = parser.parse_args()


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def default_load_dense_func(
    warmup_weight: dict, warmup_extra: dict, ps_weight: dict, ps_extra: dict, tf_weight: dict,
        load_option):
    """
    该函数的功能是聚合从各处加载的dense模型，生成最终的weight与extra。

    该函数遵循以下基本逻辑:
    1、若有warmup_weight/extra, 优先使用
    2、若无warmup，weight优先使用tf_weight, extra优先使用ps_extra
    3、若warmup中dense参数非全量，则会使用tf_weight进行补全, extra使用ps_extra补全

    Arguments:
        warmup_weight {[dict]} -- [从保存的model加载得到的weight]
        warmup_extra {[dict]} -- [从保存的model加载得到的extra]
        ps_weight {[dict]} -- [从参数服务器上拉取的weight]
        ps_extra {[dict]} -- [从参数服务器上拉取的extra]
        tf_weight {[dict]} -- [tensorflow本地通过初始化op生成的weight]
        load_option {dict} -- [控制当次 load 行为的所有配置]

    Raises:
        path: [description]

    Returns:
        weight [dcit] -- [最终聚合得到的weight]
        extra  [dict] -- [最终聚合得到的extra]
    """
    # 加载其他路径模型，宽松检查；加载当前路径模型，严格检查。
    strict = not (load_option.load_from_other_model_path)

    if strict:
        tf_weight_keys = set(tf_weight.keys())
        warmup_weight_keys = set(warmup_weight.keys())
        warmup_extra_keys = set(warmup_extra.keys())
        ps_extra_keys = set(ps_extra.keys())
        # assert len(tf_weight) == len(warmup_weight), "参数数量不匹配：{} vs {}，模型多定义了[{}]，weight 多加载了[{}]".format(
        #     len(tf_weight), len(warmup_weight),
        #     tf_weight_keys.difference(warmup_weight_keys), warmup_weight_keys.difference(tf_weight_keys))
        if len(tf_weight) != len(warmup_weight):
            print("参数数量不匹配：{} vs {}，模型多定义了[{}]，weight 多加载了[{}]".format(
                len(tf_weight), len(warmup_weight),
                tf_weight_keys.difference(warmup_weight_keys), warmup_weight_keys.difference(tf_weight_keys)))

        assert len(tf_weight) == len(warmup_extra), "参数数量不匹配：{} vs {}，模型多定义了[{}]，extra 多加载了[{}]".format(
            len(tf_weight), len(warmup_extra),
            tf_weight_keys.difference(warmup_extra_keys), warmup_extra_keys.difference(tf_weight_keys))
        assert len(tf_weight) == len(ps_extra), "参数数量不匹配：{} vs {}，模型多定义了[{}]，ps extra 多拉取了[{}]".format(
            len(tf_weight), len(ps_extra),
            tf_weight_keys.difference(ps_extra_keys), ps_extra_keys.difference(tf_weight_keys))
        for name in tf_weight:
            assert name in warmup_weight, "加载的权重中没有 {}".format(name)
            assert tf_weight[name].size == warmup_weight[name].size, "{} 的 weight size 不匹配：模型定义了 {} vs 加载了 {}".format(
                name, tf_weight[name].size, warmup_weight[name].size)
            assert name in warmup_extra, "加载的优化器参数中没有 {}".format(name)
            assert ps_extra[name].size == warmup_extra[name].size, "{} 的 extra size 不匹配：模型定义了 {} vs 加载了 {}".format(
                name, ps_extra[name].size, warmup_extra[name].size)

    weight = None
    extra = None
    dense_variable_nums = len(tf_weight)

    for var_name in list(warmup_weight):
        if var_name not in tf_weight:
            print("加载的 dense variable({}) 在运行时不存在，其值被忽略。".format(var_name))  # noqa
            del warmup_weight[var_name]
            del warmup_extra[var_name]
        elif warmup_weight[var_name].size != tf_weight[var_name].size:
            print("加载的 dense variable({}) size ({} vs {}) 不匹配，其值被忽略".format(var_name, warmup_weight[var_name].size, tf_weight[var_name].size))  # noqa
            del warmup_weight[var_name]
            del warmup_extra[var_name]
    weight = warmup_weight

    for var_name in list(warmup_extra):
        if var_name not in ps_extra:
            print("加载的 dense variable extra({}) 在运行时不存在，其值被忽略。".format(var_name))  # noqa
            del warmup_extra[var_name]
        elif warmup_extra[var_name].size != ps_extra[var_name].size:
            print("加载的 dense variable extra({}) size ({} vs {}) 不匹配，其值被忽略".format(var_name, warmup_extra[var_name].size, ps_extra[var_name].size))  # noqa
            del warmup_extra[var_name]
    extra = warmup_extra

    if len(weight) < dense_variable_nums:
        for var_name, var in tf_weight.items():
            if var_name not in weight:
                print(
                    "加载的权重中未找到 {}, 使用 tensorflow 初始化的值".format(var_name))
                weight[var_name] = var

    if len(extra) < dense_variable_nums:
        for var_name, var in ps_extra.items():
            if var_name not in extra:
                print(
                    "加载的优化器参数中未找到 {}, 使用 ps 初始化的值".format(var_name))
                extra[var_name] = var

    assert len(weight) == dense_variable_nums
    assert len(extra) == dense_variable_nums
    print("加载的参数：weight={}, extra={}".format(weight.keys(), extra.keys()))
    return weight, extra


# 1. define sparse input
if args.mode == 'train':
    import tensorflow.compat.v1 as tf
    import kai.tensorflow as kai
    kai.set_load_dense_func(default_load_dense_func)

    user_embedding = kai.nn.new_embedding("user_embedding", dim=4, slots=[101, 102])
    comment_udp_id_embedding = kai.nn.new_embedding("c_udp_id_embedding", dim=64, slots=[103, 104, 105, 106])

    comment_id_embedding = kai.nn.new_embedding("c_id_embedding", dim=64, slots=[201, 202])
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

    user_embedding = config.new_embedding("user_embedding", dim=4, slots=[101, 102], **compress_kwargs)
    comment_udp_id_embedding = config.new_embedding("c_udp_id_embedding", dim=64, slots=[103, 104, 105, 106], **compress_kwargs)

    comment_id_embedding = config.new_embedding("c_id_embedding", dim=64, slots=[201, 202])
    comment_info_embedding = config.new_embedding("c_info_embedding", dim=32, slots=[203, 204, 205, 206, 207, 209])
    position_embedding = config.new_embedding("position_embedding", dim=8, slots=[208])

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


def multi_head_attention(name, inputs, padding_mask=None, num_units=None, num_heads=8, add_embedding=None):
    """  self-att
    inputs: [-1, seq_len, h]
    padding_mask: [-1, seq_len]. 是由 0 / 1 组成的 mask 的矩阵, padding位值为0, 非padding位值为1. 如果为None，则不使用mask.
    num_units: int. 等于 num_heads * depth
    num_heads: int
    add_embedding: Tensor, 残差连接的input_tensor

    return 
        attention: [-1, seq_len, num_units]
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if num_units is None:
            num_units = inputs.shape[-1]

        if inputs.shape[-1] != num_units:
            inputs = tf.layers.dense(inputs, num_units)
        
        seq_len = inputs.shape[1]
        
        # 可学习参数1
        Q = tf.layers.dense(inputs, num_units) # [-1, q_seq, dim]
        K = tf.layers.dense(inputs, num_units)    # [-1, k_seq, dim]
        V = tf.layers.dense(inputs, num_units)    # [-1, k_seq, dim]

        assert num_units % num_heads == 0
        depth = num_units // num_heads
        Q_ = tf.transpose(tf.reshape(Q, [-1, seq_len, num_heads, depth]), perm=[0, 2, 1, 3])   # [-1, num_heads, q_seq, depth]
        K_ = tf.transpose(tf.reshape(K, [-1, seq_len, num_heads, depth]), perm=[0, 2, 1, 3])   # [-1, num_heads, k_seq, depth]
        V_ = tf.transpose(tf.reshape(V, [-1, seq_len, num_heads, depth]), perm=[0, 2, 1, 3])   # [-1, num_heads, k_seq, depth]

        # [-1, num_heads, k_seq, k_seq]
        outputs = tf.matmul(Q_, K_, transpose_b=True) / tf.math.sqrt(tf.cast(depth, tf.float32))    # [-1, num_heads, q_seq, k_seq]
        
        # padding mask
        if padding_mask is not None:
            # [-1, k_seq] -> [-1, 1, k_seq, 1] -> [-1, num_heads, k_seq, k_seq]
            padding_mask = (1 - tf.tile(padding_mask[:, None, :, None], [1, num_heads, 1, seq_len])) * -1e9 # [-1, num_heads, k_seq, k_seq] 原值1转换为0，原值0转换为-1e9
            outputs = outputs + padding_mask

        outputs = tf.nn.softmax(outputs, axis=-1)   # [-1, num_heads, k_seq, k_seq]
        attention = tf.transpose(tf.matmul(outputs, V_), perm=[0, 2, 1, 3])  # [-1, k_seq, num_heads, depth]
        attention = tf.reshape(attention, [-1, seq_len, num_units])  # [-1, k_seq, num_units]

        # attention = tf.layers.dense(attention, attention.shape[-1]) # [-1, k_seq, num_units]

        # add and norm
        # 可学习参数2
        if add_embedding is not None:
            attention = layer_norm(name=f"{name}_norm", x=attention + tf.layers.dense(add_embedding, num_units))
        else:
            attention = layer_norm(name=f"{name}_norm", x=attention + inputs)
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
            if i == len(units) - 1:
                act = last_act
                hidden = output
            output = tf.layers.dense(output, unit, activation=act,
                                  kernel_initializer=tf.glorot_uniform_initializer())
        return output, hidden
    
# define model structure
used_features=[user_embedding, comment_id_embedding, comment_info_embedding, position_embedding, comment_udp_id_embedding]
field_input = tf.concat(used_features, -1)
# last_act = tf.nn.sigmoid
last_act = None
expand_xtr, expand_hidden = simple_dense_network("expand_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=last_act)
like_xtr, like_hidden = simple_dense_network("like_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=last_act)
reply_xtr, reply_hidden = simple_dense_network("reply_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=last_act)
copy_xtr, copy_hidden = simple_dense_network("copy_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=last_act)
share_xtr, share_hidden = simple_dense_network("share_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=last_act)
audience_xtr, audience_hidden = simple_dense_network("audience_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=last_act)
continuous_expand_xtr, continuous_expand_hidden = simple_dense_network("continuous_expand_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=last_act)

# attention_input = tf.stack([expand_xtr, like_xtr, reply_xtr, copy_xtr, share_xtr, audience_xtr, continuous_expand_xtr], axis=1)   # [-1, 7, 1]
attention_input = tf.stack([expand_hidden, like_hidden, reply_hidden, copy_hidden, share_hidden, audience_hidden, continuous_expand_hidden], axis=1)
# attention_input = tf.concat([pxtrs, hiddens], axis=-1)  # [-1, 7, 65]
attention_output = multi_head_attention('multi_head_attention', attention_input, num_units=48, num_heads=6) # [-1, 7, 48]
final_xtrs = tf.squeeze(tf.layers.dense(attention_output, 1, activation=None), axis=-1)    # [-1, 7]
split_xtrs = tf.split(final_xtrs, num_or_size_splits=final_xtrs.shape[-1], axis=-1)
expand_xtr1, like_xtr1, reply_xtr1, copy_xtr1, share_xtr1, audience_xtr1, continuous_expand_xtr1 = split_xtrs
expand_xtr = tf.sigmoid(expand_xtr + expand_xtr1)
like_xtr = tf.sigmoid(like_xtr + like_xtr1)
reply_xtr = tf.sigmoid(reply_xtr + reply_xtr1)
copy_xtr = tf.sigmoid(copy_xtr + copy_xtr1)
share_xtr = tf.sigmoid(share_xtr + share_xtr1)
audience_xtr = tf.sigmoid(audience_xtr + audience_xtr1)
continuous_expand_xtr = tf.sigmoid(continuous_expand_xtr + continuous_expand_xtr1)


if args.mode == 'train':
    # define label input and define metrics
    sample_weight = kai.nn.get_dense_fea("new_sample_weight", dim=1, dtype=tf.float32)
    ones = tf.ones_like(sample_weight, dtype=tf.float32)
    zeros = tf.zeros_like(sample_weight, dtype=tf.float32)

    expandAction_first = tf.cast(kai.nn.get_dense_fea("expandAction_first", dim=1, dtype=tf.int64), tf.float32)
    expand_label = tf.where(expandAction_first > 0, ones, zeros)
    continuous_expand_label = tf.where(expandAction_first > 1, ones, zeros)
    
    like_first_label = tf.cast(kai.nn.get_dense_fea("likeAction_first", dim=1, dtype=tf.int64), tf.float32)
    like_second_label = tf.cast(kai.nn.get_dense_fea("likeAction_second", dim=1, dtype=tf.int64), tf.float32)
    like_label = tf.where((like_first_label > 0) | (like_second_label > 0), ones, zeros)

    reply_first_label = tf.cast(kai.nn.get_dense_fea("replyAction_first", dim=1, dtype=tf.int64), tf.float32)
    reply_second_label = tf.cast(kai.nn.get_dense_fea("replyAction_second", dim=1, dtype=tf.int64), tf.float32)
    reply_label = tf.where((reply_first_label > 0) | (reply_second_label > 0), ones, zeros)

    copy_first_label = tf.cast(kai.nn.get_dense_fea("copyAction_first", dim=1, dtype=tf.int64), tf.float32)
    copy_second_label = tf.cast(kai.nn.get_dense_fea("copyAction_second", dim=1, dtype=tf.int64), tf.float32)
    copy_label = tf.where((copy_first_label > 0) | (copy_second_label > 0), ones, zeros)

    share_first_label = tf.cast(kai.nn.get_dense_fea("shareAction_first", dim=1, dtype=tf.int64), tf.float32)
    share_second_label = tf.cast(kai.nn.get_dense_fea("shareAction_second", dim=1, dtype=tf.int64), tf.float32)
    share_label = tf.where((share_first_label > 0) | (share_second_label > 0), ones, zeros)

    audience_first_label = tf.cast(kai.nn.get_dense_fea("audienceAction_first", dim=1, dtype=tf.int64), tf.float32)
    audience_second_label = tf.cast(kai.nn.get_dense_fea("audienceAction_second", dim=1, dtype=tf.int64), tf.float32)
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
    optimizer = kai.nn.optimizer.AdamW(1e-3)
    optimizer.minimize(loss)

    recall_type = tf.cast(kai.nn.get_dense_fea("recall_type", dim=1, dtype=tf.int64), tf.float32)

    eval_targets = [
        ('expand_predict', expand_xtr, expand_label, ones, "auc"),
        ('like_predict', like_xtr, like_label, ones, "auc"),
        ('reply_predict', reply_xtr, reply_label, ones, "auc"),
        ('copy_predict', copy_xtr, copy_label, ones, "auc"),
        ('share_predict', share_xtr, share_label, ones, "auc"),
        ('audience_predict', audience_xtr, audience_label, ones, "auc"),
        ('continuous_expand_predict', continuous_expand_xtr, continuous_expand_label, ones, "auc"),
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
        'used_features': [],
        'debug_features': used_features + [expand_xtr, like_xtr, reply_xtr],
        # 'print_features': [comment_content_segs[:3, :6, :2], seq_pos_embedding[:3,:6,:2]]
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
    ]
    q_names, preds = zip(*targets)
    config.dump_predict_config('./predict/config', targets, input_type=3, extra_preds=q_names)

