"""EPNet -> shared bottom (带PPNet) -> CGC layer -> towers
    去掉了ltr、rtr、曝光位置特征
"""
import os
import argparse
import contextlib

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'predict'], default='train')
parser.add_argument('--dryrun', dest="dryrun", const=True, default=False, nargs='?')
parser.add_argument('--with_kai', action="store_true")
args = parser.parse_args()

# 1. define sparse input
if args.mode == 'train':
    import tensorflow.compat.v1 as tf
    import kai.tensorflow as kai

    user_embedding_id = kai.nn.new_embedding("user_embedding_id", dim=64, slots=[103, 104, 105, 106])
    user_embedding_info = kai.nn.new_embedding("user_embedding_info", dim=8, slots=[107, 108, 109, 110, 111, 112, 113])
    comment_embedding_id = kai.nn.new_embedding("comment_embedding_id", dim=64, slots=[201, 202])
    comment_embedding_info = kai.nn.new_embedding("comment_embedding_info", dim=32, slots=[203, 204, 205, 209, 230, 231, 232])
    comment_embedding_tag = kai.nn.new_embedding("comment_embedding_tag", dim=8, slots=[213, 214, 215, 216, 217, 218, 220])
    comment_embedding_is = kai.nn.new_embedding("comment_embedding_is", dim=4, slots=[233, 234, 235, 236, 237, 238])
    # comment_embedding_list = kai.nn.new_embedding("comment_embedding_list", dim=32, slots=[212, 219])

    cmt_id_param_attr = kai.nn.ParamAttr(
        access_method=kai.nn.ProbabilityAccess(10.0),
        recycle_method=kai.nn.UnseendaysRecycle(
            delete_after_unseen_days=30,
            allow_dynamic_delete=True
        )
    )
    kai.nn.set_slot_param_attr([201], cmt_id_param_attr)

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
                                    with_kai=args.with_kai, predict=True)
    compress_kwargs = dict(compress_group="USER")

    user_embedding_id = config.new_embedding("user_embedding_id", dim=64, slots=[103, 104, 105, 106], **compress_kwargs)
    user_embedding_info = config.new_embedding("user_embedding_info", dim=8, slots=[107, 108, 109, 110, 111, 112, 113], **compress_kwargs)
    comment_embedding_id = config.new_embedding("comment_embedding_id", dim=64, slots=[201, 202])
    comment_embedding_info = config.new_embedding("comment_embedding_info", dim=32, slots=[203, 204, 205, 209, 230, 231, 232])
    comment_embedding_tag = config.new_embedding("comment_embedding_tag", dim=8, slots=[213, 214, 215, 216, 217, 218, 220])
    comment_embedding_is = config.new_embedding("comment_embedding_is", dim=4, slots=[233, 234, 235, 236, 237, 238])
    # comment_embedding_list = config.nn.new_embedding("comment_embedding_list", dim=32, slots=[212, 219])


def slice_inputs(inputs):
    if isinstance(inputs, list):
        return inputs[0], inputs[1:]
    else:
        return inputs, []


def tf_name_scope(scope_name):
    if args.mode == "predict":
        return tf.name_scope(None)
    else:
        return tf.name_scope(scope_name)


def tf_name_scope_tower(scope_name):
    if args.mode == "predict":
        return tf.name_scope(scope_name)
    else:
        return tf.name_scope(scope_name)

def matmul_ex_wrapper(a, b, transpose_a=False, transpose_b=False):
    return tf.matmul_float_to_bf16(a, b, transpose_a=transpose_a, transpose_b=transpose_b)


def matmul_wrapper(*args, **kwargs):
    return tf.matmul(*args, **kwargs)


def swish(x, beta=1.0):
    return x * tf.nn.sigmoid(beta * x)


def new_xla_jit_context():
    if use_xla:
        return tf.xla.experimental.jit_scope()
    else:
        return contextlib.suppress()


use_matmul_ex = True if args.with_kai else False
matmul_fun = matmul_ex_wrapper if use_matmul_ex else matmul_wrapper
use_xla = False if args.mode == "predict" else True
dense_norm_type = 'batch_norm'
gate_norm_type = 'batch_norm'
emb_norm_type = 'batch_norm'
bottom_norm_type = 'batch_norm'
trans_norm_type = 'layer_norm'
sparse_norm_type = None
# dense_activation = tf.nn.leaky_relu
dense_activation = swish
predict_tower_units = [128, 64, 1]


def norm(x, begin_axis=-1, eps=1e-5, norm_type=dense_norm_type, scope_name=''):
    shape = x.shape.as_list()
    axes = list(range(len(shape)))[begin_axis:]

    if norm_type == 'layer_norm':
        mean, var = tf.nn.moments(x, axes, keepdims=True, name=f"{scope_name}_moments")
        x = (x - mean) * tf.math.rsqrt(var + eps)
        gamma = tf.get_variable(f'layer_norm_gamma_{scope_name}', shape=x.shape.as_list()[
                                begin_axis:], initializer=tf.initializers.ones())
        beta = tf.get_variable(f'layer_norm_beta_{scope_name}', shape=x.shape.as_list()[
                               begin_axis:], initializer=tf.initializers.zeros())
        output = gamma * x + beta
        return output
    elif norm_type == 'batch_norm':
        if args.mode == "predict":
            output = config.batch_norm(input=x, name=f'batch_norm_{scope_name}')
        else:
            output = kai.batch_norm(input=x, name=f'batch_norm_{scope_name}',
                                        forward_with_moving_val=False, 
                                        use_tf_cond=False,
                                        bn_decay=0.0)
        return output
    elif norm_type is None:
        return x

def norm_tower(x, begin_axis=-1, eps=1e-5, norm_type=dense_norm_type, scope_name='', name=''):
    shape = x.shape.as_list()
    axes = list(range(len(shape)))[begin_axis:]

    if norm_type == 'layer_norm':
        mean, var = tf.nn.moments(x, axes, keepdims=True, name=f"{scope_name}_moments")
        x = (x - mean) * tf.math.rsqrt(var + eps)
        gamma = tf.get_variable(f'layer_norm_gamma_{scope_name}', shape=x.shape.as_list()[
                                begin_axis:], initializer=tf.initializers.ones())
        beta = tf.get_variable(f'layer_norm_beta_{scope_name}', shape=x.shape.as_list()[
                               begin_axis:], initializer=tf.initializers.zeros())
        output = gamma * x + beta
        return output
    elif norm_type == 'batch_norm':
        if args.mode == "predict":
            output = config.batch_norm(input=x, name=f'{name}/batch_norm_{scope_name}')
        else:
            output = kai.batch_norm(input=x, name=f'batch_norm_{scope_name}',
                                        forward_with_moving_val=False, 
                                        use_tf_cond=False,
                                        bn_decay=0.0)
        return output
    elif norm_type is None:
        return x

def emb_norm(inputs, scope_name):
    return norm(inputs, norm_type=emb_norm_type, scope_name=scope_name)


def mio_dense_layer(inputs, units, activation, name, weight_name, norm_type=None):
    print("mio_dense_layer: inputs: {}; units: {}; activation: {}; name: {}; weight_name: {}; norm_type: {}; ".
          format(inputs, units, activation, name, weight_name, norm_type))
    i, extra_i = slice_inputs(inputs)

    with tf_name_scope(name):
        o = matmul_fun(i, tf.get_variable(
            weight_name, (i.get_shape()[-1], units)))
        for idx, extra_i in enumerate(extra_i):
            o += matmul_fun(extra_i, tf.get_variable(
                f"{weight_name}_extra_{idx}", (extra_i.get_shape()[-1], units)))

        bias = tf.get_variable(f"{weight_name}_bias", (units))
        o = tf.nn.bias_add(o, bias)

        if norm_type is not None:
            # print("mio_dense_layer: name: {0}; norm_type: {1}".format(name, norm_type))
            o = norm(o, norm_type=norm_type, scope_name=weight_name)

        if activation is not None:
            o = activation(o)

        return o

def mio_dense_layer_tower(inputs, units, activation, name, weight_name, norm_type=None, scope_name=''):
    print("mio_dense_layer: inputs: {}; units: {}; activation: {}; name: {}; weight_name: {}; norm_type: {}; ".
          format(inputs, units, activation, name, weight_name, norm_type))
    i, extra_i = slice_inputs(inputs)

    with tf_name_scope(name):
        o = matmul_fun(i, tf.get_variable(
            weight_name, (i.get_shape()[-1], units)))
        for idx, extra_i in enumerate(extra_i):
            o += matmul_fun(extra_i, tf.get_variable(
                f"{weight_name}_extra_{idx}", (extra_i.get_shape()[-1], units)))

        bias = tf.get_variable(f"{weight_name}_bias", (units))
        o = tf.nn.bias_add(o, bias)

        if norm_type is not None:
            # print("mio_dense_layer: name: {0}; norm_type: {1}".format(name, norm_type))
            # print("hzh:", scope_name)
            o = norm_tower(o, norm_type=norm_type, scope_name=weight_name, name=scope_name)

    if activation is not None:
        o = activation(o)

    return o

def mio_dense_layer_batch(inputs, units, activation, name, weight_name, norm_type=None):
    """
    input is (a * b * c)
    tf.matmul(b * a * c, b * c * d) = b * a * d. 
    """
    b = inputs.get_shape()[1]
    c = inputs.get_shape()[2]
    d = units
    inputs = tf.transpose(inputs, perm=[1, 0, 2])
    o = matmul_fun(inputs, tf.reshape(tf.get_variable(weight_name, (b, c * d)), (b, c, d)))
    bias = tf.get_variable(f"{weight_name}_bias", (b, d))
    result = tf.transpose(o, perm=[1, 0, 2])
    result = result + bias
    if norm_type is not None:
        print("mio_dense_layer: name: {0}; norm_type: {1}".format(name, norm_type))
        result = norm(result, norm_type=norm_type, scope_name=weight_name)

    if activation is not None:
        result = activation(result)

    return result


# def simple_tower_network(name, inputs, units, dropout=0, act=dense_activation, last_act=tf.nn.sigmoid, norm_type=dense_norm_type, stop_gradient=False):
#     if stop_gradient:
#         output = tf.stop_gradient(inputs)
#     else:
#         output = inputs
#     with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
#         if dropout > 0:
#             output = tf.layers.dropout(output, dropout, training=(args.mode == 'train'))
#         for i, unit in enumerate(units):
#             # output = tf.layers.Dense(unit, act, name='dense_{}_{}'.format(name, i))(output)
#             if i == len(units) - 1:
#                 act = last_act
#                 norm_type = None            
#             output = mio_dense_layer_tower(output, unit, act, name=f"dense_{name}_{i}",
#                                     weight_name=f"{name}_tower_h{i+1}_param",
#                                     norm_type=norm_type, scope_name=name)
#         return output
    
def simple_tower_network(name, inputs, units, dropout=0, act=dense_activation, last_act=tf.nn.sigmoid, norm_type=dense_norm_type, stop_gradient=False):
    if stop_gradient:
        output = tf.stop_gradient(inputs)
    else:
        output = inputs
    with tf_name_scope(name):
        if dropout > 0:
            output = tf.layers.dropout(output, dropout, training=(args.mode == 'train'))
        for i, unit in enumerate(units):
            # output = tf.layers.Dense(unit, act, name='dense_{}_{}'.format(name, i))(output)
            if i == len(units) - 1:
                act = last_act
                norm_type = None            
            output = mio_dense_layer(output, unit, act, name=f"dense_{name}_{i}",
                                    weight_name=f"{name}_tower_h{i+1}_param",
                                    norm_type=norm_type)
        return output


def simple_dense_network(inputs, units, name, weight_name_template, act=dense_activation, hidden_layer_extra_inputs=[],
                         norm_type=dense_norm_type):
    output = inputs
    for i, unit in enumerate(units):
        output = mio_dense_layer(output, unit, act, name=f"dense_{name}_{i}",
                                 weight_name=weight_name_template.format(i + 1),
                                 norm_type=norm_type)
        output = [output, *hidden_layer_extra_inputs]
    return output[0]


def simple_lhuc_network(inputs, unit1, unit2, name, weight_name):
    with tf_name_scope(f"{name}_lhuc"):
        output = tf.concat(inputs, 1)
        with tf_name_scope(f"{name}_lhuc_layer_0"):
            output = mio_dense_layer(output, unit1, dense_activation, name=f"dense_{name}_0",
                                     weight_name=f"{weight_name}_layer1_param", norm_type=gate_norm_type)
        with tf_name_scope(f"{name}_lhuc_layer_1"):
            output = 2.0 * mio_dense_layer(output, unit2, tf.nn.sigmoid, name=f"dense_{name}_1",
                                           weight_name=f"{weight_name}_layer2_param",
                                           norm_type=gate_norm_type)
        return output


def build_expert_component(inputs, expert_units, num_experts, name, norm_type=bottom_norm_type):
    print("build_expert_component inputs: {}; expert_units: {}; num_experts: {}; name: {}".format(inputs, expert_units, num_experts, name))
    expert_outputs = []
    with tf_name_scope("experts_network"):
        for i in range(num_experts):
            expert_layer = simple_dense_network(inputs, expert_units, f"{name}_experts",
                                                f"{name}_expert{i}_h{{}}_param", act=dense_activation,
                                                norm_type=norm_type)
            expert_outputs.append(expert_layer)
        # concat_expert = tf.concat(expert_outputs, 1) # (b, num_experts * expert_units[-1])
        stack_expert = tf.stack(expert_outputs, 1)  # (b, num_experts, expert_units[-1])

    return stack_expert


def build_expert_gate_component(inputs, num_experts, expert_outputs, name, gate_units=[128, 32]):
    print("build_expert_gate_component inputs: {}; num_experts: {}; expert_outputs: {}; name: {}".format(inputs, num_experts, expert_outputs, name))
    with tf_name_scope("gates_network"):
        gate_input = simple_dense_network(inputs, gate_units, f"{name}_gates_mlp",
                                        f"{name}_gates_mlp_h{{}}_param", act=swish, norm_type=bottom_norm_type)
        gate_act = tf.nn.sigmoid if num_experts == 1 else tf.nn.softmax
        gate_layer = mio_dense_layer(gate_input, num_experts, gate_act, f"{name}_gates",
                                        f"{name}_gates_param")
        gate_layer_reshape = tf.reshape(gate_layer, [-1, 1, num_experts])
        weighted_expert_output = matmul_fun(gate_layer_reshape, expert_outputs)
        output_last_shape = weighted_expert_output.get_shape()[-1]
        output = tf.reshape(weighted_expert_output, [-1, output_last_shape])

    return output, gate_layer  # (b, output_last_shape)


def build_senet_unit(inputs, num_blocks=3, task_nn_hidden=512, name='senet'):
    print("build_senet_unit: inputs: {}; num_blocks: {}; task_nn_hidden: {}; name: {}"
          .format(inputs, num_blocks, task_nn_hidden, name))

    senet_outputs = []
    is_debug = True
    for i in range(num_blocks):
        if task_nn_hidden:
            gate_input = simple_dense_network(inputs, [task_nn_hidden], f"{name}_senet_block_{i}_mlp",
                                    f"{name}_senet_block_{i}_mlp_h{{}}_param", act=swish, norm_type=bottom_norm_type)
        else:
            gate_input = inputs
        gate_output = mio_dense_layer(gate_input, inputs.get_shape()[-1], tf.nn.sigmoid, name=f"{name}_senet_block_{i}",
                                 weight_name=f"{name}_senet_block_{i}_param", norm_type=bottom_norm_type)
        gate_output = 2.0 * gate_output
        senet_outputs.append(gate_output)

    if num_blocks > 1:
        gate_output, gate_layer = build_expert_gate_component(inputs, num_blocks, tf.stack(senet_outputs, 1), f"{name}_senet", gate_units=[128])
        if is_debug:
            for i in range(num_blocks):
                tf.summary.scalar(f"{name}_senet_block_gate_{i}", tf.reduce_mean(gate_layer, 0)[i])
    
    return gate_output * inputs


def glu_unit(inputs, output_d=None, unit=None, name='glu', direct_output=False):
    matmul_fun = matmul_ex_wrapper if use_matmul_ex else tf.matmul
    dim = len(inputs.get_shape())
    if dim !=2 and dim != 3:
        raise ValueError("glu only takes inputs of 2/3 dimensions.")
    d = inputs.get_shape()[-1]
    col = None
    if dim == 3:
        col = inputs.get_shape()[1]
    inputs = tf.reshape(inputs, (-1, d))
    if unit is None:
        unit = 2 * d
    u_1 = matmul_fun(inputs, tf.get_variable(f"{name}_weight_u", (d, unit)))
    u = u_1 + tf.get_variable(f"{name}_bias_u", (unit))

    v_1 = matmul_fun(inputs, tf.get_variable(f"{name}_weight_v", (d, unit)))
    v = v_1 + tf.get_variable(f"{name}_bias_v", (unit))

    if direct_output:
        u = norm(u, norm_type=bottom_norm_type, scope_name=f"{name}_weight_u")
        v = norm(v, norm_type=bottom_norm_type, scope_name=f"{name}_weight_v")
        v = 2.0 * tf.nn.sigmoid(v)
        o = tf.multiply(u, v)
        return o

    v = tf.nn.sigmoid(v)
    o = tf.multiply(u, v)

    if not output_d:
        result_1 = matmul_fun(o, tf.get_variable(
            f"{name}_weight_result", (unit, d)))
        result = result_1 + tf.get_variable(f"{name}_bias_result", (d))
        return tf.reshape(result, (-1, col, d)) if col else result
    else:
        result_1 = matmul_fun(o, tf.get_variable(
            f"{name}_weight_result", (unit, output_d)))
        result = result_1 + tf.get_variable(f"{name}_bias_result", (output_d))
        return tf.reshape(result, (-1, col, output_d)) if col else result

    
def cgc_layer(inputs, shared_expert_units, shared_num_experts, idp_expert_units, task_list, use_senet_input, name):
    is_debug = True
    num_blocks = 1
    nn_hidden = None if num_blocks == 1 else inputs.get_shape()[1].value // (num_blocks * 2)
    idp_num_experts = 1

    if use_senet_input:
        senet_inputs = build_senet_unit(inputs, num_blocks=num_blocks, task_nn_hidden=nn_hidden, name=f"{name}_shared")
    else:
        senet_inputs = inputs
    shared_expert_output = build_expert_component(senet_inputs, shared_expert_units, shared_num_experts, f"{name}_shared")

    final_outputs = {}
    for task_name in task_list:
        if use_senet_input:
            senet_inputs = build_senet_unit(inputs, num_blocks=num_blocks, task_nn_hidden=nn_hidden, name=f"{name}_{task_name}")
        else:
            senet_inputs = inputs

        idp_expert_output = build_expert_component(senet_inputs, idp_expert_units, idp_num_experts, f"{name}_{task_name}")
        total_expert_outputs = tf.concat([shared_expert_output, idp_expert_output], 1)

        output, gate_layer = build_expert_gate_component(inputs, shared_num_experts + idp_num_experts, total_expert_outputs, f"{name}_{task_name}")
        if is_debug:
            for i in range(shared_num_experts + idp_num_experts):
                tf.summary.scalar(f"{task_name}_expert_gate_{i}", tf.reduce_mean(gate_layer, 0)[i])
        final_outputs[task_name] = output

    return final_outputs


############# define model structure #############
with tf_name_scope("slot_gate"), new_xla_jit_context():

    user_embedding_id = emb_norm(user_embedding_id, "user_embedding_id")
    user_embedding_info = emb_norm(user_embedding_info, "user_embedding_info")
    comment_embedding_id = emb_norm(comment_embedding_id, "comment_embedding_id")
    comment_embedding_info = emb_norm(comment_embedding_info, "comment_embedding_info")
    # comment_embedding_list = emb_norm(comment_embedding_list, "comment_embedding_list")
    comment_embedding_tag = emb_norm(comment_embedding_tag, "comment_embedding_tag")
    comment_embedding_is = emb_norm(comment_embedding_is, "comment_embedding_is")

    def get_col(x): return x.get_shape()[1].value
    update_inputs_dim64 = [user_embedding_id, comment_embedding_id]
    num_dim64_slots = sum(map(get_col, update_inputs_dim64)) // 64
    update_inputs_dim32 = [comment_embedding_info]
    num_dim32_slots = sum(map(get_col, update_inputs_dim32)) // 32
    update_inputs_dim8 = [user_embedding_info, comment_embedding_tag]
    num_dim8_slots = sum(map(get_col, update_inputs_dim8)) // 8
    update_inputs_dim4 = [comment_embedding_is]
    num_dim4_slots = sum(map(get_col, update_inputs_dim4)) // 4
    num_dimn_slots_list = [num_dim64_slots, num_dim32_slots, num_dim8_slots, num_dim4_slots]
    print("num_slots", num_dimn_slots_list)

    slot_gate_input = [tf.stop_gradient(embedding) for embedding in 
                       (update_inputs_dim64 + update_inputs_dim32 + update_inputs_dim8 + update_inputs_dim4)]
    slot_gate = simple_lhuc_network(slot_gate_input, 256, num_dim64_slots + num_dim32_slots + num_dim8_slots + num_dim4_slots, 
                                    "slot_gate", "slot_gate")
    update_inputs_dim64_concat = tf.reshape(
        tf.concat(update_inputs_dim64, 1), (-1, num_dim64_slots, 64))
    update_inputs_dim32_concat = tf.reshape(
        tf.concat(update_inputs_dim32, 1), (-1, num_dim32_slots, 32))
    update_inputs_dim8_concat = tf.reshape(
        tf.concat(update_inputs_dim8, 1), (-1, num_dim8_slots, 8))
    update_inputs_dim4_concat = tf.reshape(
        tf.concat(update_inputs_dim4, 1), (-1, num_dim4_slots, 4))
    slot_gate_split = tf.split(slot_gate, num_dimn_slots_list, 1)

    base_features_dim64 = tf.expand_dims(slot_gate_split[0], 2) * update_inputs_dim64_concat
    base_features_dim32 = tf.expand_dims(slot_gate_split[1], 2) * update_inputs_dim32_concat
    base_features_dim8 = tf.expand_dims(slot_gate_split[2], 2) * update_inputs_dim8_concat
    base_features_dim4 = tf.expand_dims(slot_gate_split[3], 2) * update_inputs_dim4_concat


with tf_name_scope("base_transformer"), new_xla_jit_context():
    new_dim64_trans = tf.reshape(base_features_dim64, (-1, num_dim64_slots * 64))
    new_dim32_trans = tf.reshape(base_features_dim32, (-1, num_dim32_slots * 32))
    new_dim8_trans = tf.reshape(base_features_dim8, (-1, num_dim8_slots * 8))
    new_dim4_trans = tf.reshape(base_features_dim4, (-1, num_dim4_slots * 4))

    new_trans_features = tf.concat([new_dim64_trans, new_dim32_trans, new_dim8_trans, new_dim4_trans], -1)
    print("new features", new_trans_features)


with tf_name_scope("cgc_network"), new_xla_jit_context():
    shared_bottom_layer = glu_unit(new_trans_features, output_d=new_trans_features.get_shape()[-1], unit=new_trans_features.get_shape()[-1], 
                                   name="shared_bottom_layer", direct_output=True)
    cgc_output = {}
    task_list = ["expand_xtr", "like_xtr", "reply_xtr", "copy_xtr", 
                 "share_xtr", "audience_xtr", "continuous_expand_xtr"]
    cgc_output = cgc_layer(inputs=shared_bottom_layer, shared_expert_units=[256, 256], shared_num_experts=3, 
                            idp_expert_units=[256, 256], task_list=task_list, use_senet_input=True, name="cgc")


with tf_name_scope("tower_network"), new_xla_jit_context():
    expand_xtr = simple_tower_network("expand_xtr", cgc_output["expand_xtr"], predict_tower_units)
    like_xtr = simple_tower_network("like_xtr", cgc_output["like_xtr"], predict_tower_units)
    reply_xtr = simple_tower_network("reply_xtr", cgc_output["reply_xtr"], predict_tower_units)
    copy_xtr = simple_tower_network("copy_xtr", cgc_output["copy_xtr"], predict_tower_units)
    share_xtr = simple_tower_network("share_xtr", cgc_output["share_xtr"], predict_tower_units)
    audience_xtr = simple_tower_network("audience_xtr", cgc_output["audience_xtr"], predict_tower_units)
    continuous_expand_xtr = simple_tower_network("continuous_expand_xtr", cgc_output["continuous_expand_xtr"], predict_tower_units)


if args.mode == 'train':
    # # define label input and define metrics

    # define label input and define metrics
    sample_weight = kai.nn.get_dense_fea("expandAction_v", dim=1, dtype=tf.float32)
    ones = tf.ones_like(sample_weight, dtype=tf.float32)
    zeros = tf.zeros_like(sample_weight, dtype=tf.float32)

    expandAction_first = kai.nn.get_dense_fea("expandAction_v", dim=1, dtype=tf.float32)
    expand_label = tf.where(expandAction_first > 0, ones, zeros)
    continuous_expand_label = tf.where(expandAction_first > 1, ones, zeros)
    
    like_first_label = kai.nn.get_dense_fea("likeAction_v", dim=1, dtype=tf.float32)
    # like_second_label = kai.nn.get_dense_fea("likeAction_second", dim=1, dtype=tf.float32)
    like_label = tf.where(like_first_label > 0, ones, zeros)

    reply_first_label = kai.nn.get_dense_fea("replyAction_v", dim=1, dtype=tf.float32)
    # reply_second_label = kai.nn.get_dense_fea("replyAction_second", dim=1, dtype=tf.float32)
    reply_label = tf.where(reply_first_label > 0, ones, zeros)

    copy_first_label = kai.nn.get_dense_fea("copyAction_v", dim=1, dtype=tf.float32)
    # copy_second_label = kai.nn.get_dense_fea("copyAction_second", dim=1, dtype=tf.float32)
    copy_label = tf.where(copy_first_label > 0, ones, zeros)

    share_first_label = kai.nn.get_dense_fea("shareAction_v", dim=1, dtype=tf.float32)
    # share_second_label = kai.nn.get_dense_fea("shareAction_second", dim=1, dtype=tf.float32)
    share_label = tf.where(share_first_label > 0, ones, zeros)

    audience_first_label = kai.nn.get_dense_fea("audienceAction_v", dim=1, dtype=tf.float32)
    # audience_second_label = kai.nn.get_dense_fea("audienceAction_second", dim=1, dtype=tf.float32)
    audience_label = tf.where(audience_first_label > 0, ones, zeros)

    # slide_label = kai.nn.get_dense_fea("slideAction", dim=1, dtype=tf.float32)
    # slide_copy_label = kai.nn.get_dense_fea("slideAction_copy", dim=1, dtype=tf.float32)
    # randomAction = kai.nn.get_dense_fea("randomAction", dim=1, dtype=tf.float32)
    # complementAction = kai.nn.get_dense_fea("complementAction", dim=1, dtype=tf.float32)
    # depthAction = kai.nn.get_dense_fea("depthAction", dim=1, dtype=tf.float32)
    # seqAction = kai.nn.get_dense_fea("seqAction", dim=1, dtype=tf.float32)

    targets = [
        ('expand_predict', expand_xtr, expand_label, ones, "auc"),
        ('like_predict', like_xtr, like_label, ones, "auc"),
        ('reply_predict', reply_xtr, reply_label, ones, "auc"),
        ('copy_predict', copy_xtr, copy_label, ones, "auc"),
        ('share_predict', share_xtr, share_label, ones, "auc"),
        ('audience_predict', audience_xtr, audience_label, ones, "auc"),
        ('continuous_expand_predict', continuous_expand_xtr, continuous_expand_label, ones, "auc")
    ]

    metric_name, preds, labels, weights, metric_type = zip(*targets)

    # 5. define optimizer
    loss = tf.losses.log_loss(labels, preds, weights, reduction="weighted_sum")
    optimizer = kai.nn.optimizer.Adam(1e-3)
    optimizer.minimize(loss)

    
    eval_targets = [
        ('expand_predict', expand_xtr, expand_label, ones, "auc"),
        ('like_predict', like_xtr, like_label, ones, "auc"),
        ('reply_predict', reply_xtr, reply_label, ones, "auc"),
        ('copy_predict', copy_xtr, copy_label, ones, "auc"),
        ('share_predict', share_xtr, share_label, ones, "auc"),
        ('audience_predict', audience_xtr, audience_label, ones, "auc"),
        ('continuous_expand_predict', continuous_expand_xtr, continuous_expand_label, ones, "auc")
    ]

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

    # debug_tensor = {
    #     "slide_xtr": tf.reshape(tf.slice(slide_xtr, [0, 0], [10, -1]), [-1, 1]),
    #     "slide_label": tf.reshape(tf.slice(slide_label, [0, 0], [10, -1]), [-1, 1]),
    #     "slide_copy_label": tf.reshape(tf.slice(slide_copy_label, [0, 0], [10, -1]), [-1, 1]),
    # }
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
      ("continuous_expand_xtr", continuous_expand_xtr)
    ]
    q_names, preds = zip(*targets)
    config.dump_predict_config('./predict/opt_slim_fix', targets, input_type=3, extra_preds=q_names)

