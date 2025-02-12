""" 增加 地理位置、请求时间、评论发布时间 特征
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'predict'], default='train')
parser.add_argument('--dryrun', dest="dryrun", const=True, default=False, nargs='?')
parser.add_argument('--with_kai', action="store_true")
args = parser.parse_args()


if args.mode == 'train':
    import tensorflow.compat.v1 as tf
    import kai.tensorflow as kai

    print(f'tf_version: {tf.__version__}')

    # 输入特征
    user_profile_emb = kai.nn.new_embedding("user_profile_emb", dim=4, slots=[101, 102])  # gender, age
    personalized_id_emb = kai.nn.new_embedding("personalized_id_emb", dim=64, slots=[103, 104, 105, 106])  # userid, photo_author_id, photo_id, device_id
    context_emb = kai.nn.new_embedding("context_emb", dim=64, slots=[110, 111, 114])     # mod, page_type, city
    video_profile_emb = kai.nn.new_embedding("video_profile_emb", dim=4, slots=[112])    # is_political
    context_emb2 = kai.nn.new_embedding("context_emb2", dim=8, slots=[115, 116])    # request_hour, request_day

    # user_continuous_cid_fea = kai.nn.new_embedding("user_continuous_cid_fea", dim=64, slots=[300])
    # user_continuous_cid_weights_fea = kai.nn.new_embedding("user_continuous_cid_weights_fea", dim=8, slots=[301])
    # user_continuous_cid_mmu_categories_fea = kai.nn.new_embedding("user_continuous_cid_mmu_categories_fea", dim=8, slots=[302])

    c_id_emb = kai.nn.new_embedding("c_id_emb", dim=64, slots=[201, 202])   # cid, author_id
    c_info_emb = kai.nn.new_embedding("c_info_emb", dim=32, slots=[203, 204, 205, 206, 207, 209])
    c_position_emb = kai.nn.new_embedding("c_position_emb", dim=8, slots=[208])

    c_tag_emb = kai.nn.new_embedding("c_tag_emb", dim=8, slots=[250, 251, 252, 253, 254, 255])
    c_cnt_emb = kai.nn.new_embedding("c_cnt_emb", dim=12, slots=[271, 272, 273, 274, 275, 276, 277, 286, 287, 288, 289, 290, 291])
    c_mmu_score_emb = kai.nn.new_embedding("c_mmu_score_emb", dim=8, slots=[278, 279, 280])
    c_xtr_emb = kai.nn.new_embedding("c_xtr_emb", dim=8, slots=[281, 282, 283, 284, 285])
    c_binary_tag_emb = kai.nn.new_embedding("c_binary_tag_emb", dim=4, slots=[270, 292, 293, 294, 295, 296, 297, 298])
    
    # new features


    # mmu_hetu_content_emb = kai.nn.get_dense_fea('mmu_hetu_content_emb', dim=128, dtype=tf.float32)
    # mmu_clip_content_emb = kai.nn.get_dense_fea('mmu_clip_content_emb', dim=1024, dtype=tf.float32)
    # mmu_bert_content_emb = kai.nn.get_dense_fea('mmu_bert_content_emb', dim=256, dtype=tf.float32)
    
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

    # 输入特征
    user_profile_emb = config.new_embedding("user_profile_emb", dim=4, slots=[101, 102])  # gender, age
    personalized_id_emb = config.new_embedding("personalized_id_emb", dim=64, slots=[103, 104, 105, 106])  # userid, photo_author_id, photo_id, device_id
    context_emb = config.new_embedding("context_emb", dim=64, slots=[110, 111, 114])     # mod, page_type
    video_profile_emb = config.new_embedding("video_profile_emb", dim=4, slots=[112])    # is_political
    context_emb2 = config.new_embedding("context_emb2", dim=8, slots=[115, 116])    # request_hour, request_day

    # user_continuous_cid_fea = config.new_embedding("user_continuous_cid_fea", dim=64, slots=[300])
    # user_continuous_cid_weights_fea = config.new_embedding("user_continuous_cid_weights_fea", dim=8, slots=[301])
    # user_continuous_cid_mmu_categories_fea = config.new_embedding("user_continuous_cid_mmu_categories_fea", dim=8, slots=[302])

    c_id_emb = config.new_embedding("c_id_emb", dim=64, slots=[201, 202])   # cid, author_id
    c_info_emb = config.new_embedding("c_info_emb", dim=32, slots=[203, 204, 205, 206, 207, 209])
    c_position_emb = config.new_embedding("c_position_emb", dim=8, slots=[208])

    # new features
    c_tag_emb = config.new_embedding("c_tag_emb", dim=8, slots=[250, 251, 252, 253, 254, 255])
    c_cnt_emb = config.new_embedding("c_cnt_emb", dim=12, slots=[271, 272, 273, 274, 275, 276, 277, 286, 287, 288, 289, 290, 291])
    c_mmu_score_emb = config.new_embedding("c_mmu_score_emb", dim=8, slots=[278, 279, 280])
    c_xtr_emb = config.new_embedding("c_xtr_emb", dim=8, slots=[281, 282, 283, 284, 285])
    c_binary_tag_emb = config.new_embedding("c_binary_tag_emb", dim=4, slots=[270, 292, 293, 294, 295, 296, 297, 298])

    # mmu_hetu_content_emb = config.get_dense_fea('mmu_hetu_content_emb', dim=128, dtype=tf.float32)
    # mmu_clip_content_emb = config.get_dense_fea('mmu_clip_content_emb', dim=1024, dtype=tf.float32)
    # mmu_bert_content_emb = config.get_dense_fea('mmu_bert_content_emb', dim=256, dtype=tf.float32)



def implicit_cross_layer(name, inputs, units, enable_resnet=False):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        output = inputs
        for i, unit in enumerate(units):
            if enable_resnet:
                last_output = output
            output = tf.layers.dense(output, unit, activation='relu',
                                kernel_initializer=tf.glorot_uniform_initializer())
            if enable_resnet and last_output.shape[-1] == output.shape[-1]:
                output = output + last_output
        return output

def explicit_cross_layer(name, inputs, raw_inputs, projection_dim=None):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if projection_dim is None:
            prod_output = tf.layers.dense(inputs, inputs.shape[-1], activation='relu')
        else:
            prod_output = tf.layers.dense(inputs, projection_dim, use_bias=False)  
            prod_output = tf.layers.dense(prod_output, inputs.shape[-1], use_bias=True)
        return raw_inputs*prod_output + inputs


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
    

def main_model(inputs, extra_inputs=None, tower_names=['expand', 'like', 'reply'], tower_units=[256, 128, 64, 1], 
            explicit_cross_layer_num=None, explicit_cross_projection_dim=None, explicit_cross_final_proj_dim=None,
            implicit_cross_layer_units=[], implicit_cross_enable_resnet=False, projection_units_for_extra_input=[],
            explicit_cross_enable_resnet=False):
    with tf.variable_scope('main_model', reuse=tf.AUTO_REUSE):                    
        bottom_outputs = []
        if implicit_cross_layer_units:
            implicit_cross_output = implicit_cross_layer('implicit_cross_layer', inputs, implicit_cross_layer_units, implicit_cross_enable_resnet)
            bottom_outputs.append(implicit_cross_output)

        if explicit_cross_layer_num is not None:
            explicit_cross_output = inputs
            for i in range(explicit_cross_layer_num):
                explicit_cross_output = explicit_cross_layer(f'explicit_cross_layer_{i}', explicit_cross_output, inputs, explicit_cross_projection_dim)
            if explicit_cross_enable_resnet:
                explicit_cross_output = explicit_cross_output + inputs
            if explicit_cross_final_proj_dim is not None:
                explicit_cross_output = tf.layers.dense(explicit_cross_output, explicit_cross_final_proj_dim, activation='relu')
            bottom_outputs.append(explicit_cross_output)
        
        if extra_inputs is not None:
            if projection_units_for_extra_input:
                for i, unit in enumerate(projection_units_for_extra_input):
                    extra_inputs = tf.layers.dense(extra_inputs, unit, activation=tf.nn.leaky_relu,
                                        kernel_initializer=tf.glorot_uniform_initializer())
            bottom_outputs.append(extra_inputs)
        
        if not bottom_outputs:
            bottom_outputs = inputs
        else:
            bottom_outputs = tf.concat(bottom_outputs, -1)
        
        # towers
        outputs=[]
        tower_inputs = bottom_outputs
        for i, tower_name in enumerate(tower_names):
            output = tower_module(tower_name, tower_inputs, tower_units)
            outputs.append(output)
        return outputs
        
    
# forward
used_features = [user_profile_emb, personalized_id_emb, context_emb, video_profile_emb,
                c_id_emb, c_info_emb, c_position_emb, c_tag_emb, c_cnt_emb, c_mmu_score_emb, c_xtr_emb, 
                c_binary_tag_emb, context_emb2
                # user_continuous_cid_fea, user_continuous_cid_weights_fea, user_continuous_cid_mmu_categories_fea
                ]
inputs = tf.concat(used_features, -1)
# extra_inputs = tf.concat([mmu_hetu_content_emb, mmu_clip_content_emb, mmu_bert_content_emb], -1)
expand_xtr, like_xtr, reply_xtr, copy_xtr, share_xtr, audience_xtr, continuous_expand_xtr = main_model(
    inputs, extra_inputs=None, tower_names=['expand', 'like', 'reply', 'copy', 'share', 'audience', 'continuous_expand'],
    tower_units=[256, 128, 64, 1], explicit_cross_layer_num=2, explicit_cross_projection_dim=128,
    explicit_cross_final_proj_dim=256, implicit_cross_layer_units=[], implicit_cross_enable_resnet=True,
    projection_units_for_extra_input=[512, 256], explicit_cross_enable_resnet=True
)



if args.mode == 'train':
    # # define label input and define metrics
    ## 注意：使用kafka数据流时，不需要使用tf.cast，直接读取tf.float32数据。例如 sample_weight = kai.nn.get_dense_fea("sample_weight", dim=1, dtype=tf.float32)

    # define label input and define metrics
    sample_weight = kai.nn.get_dense_fea("sample_weight", dim=1, dtype=tf.float32)
    ones = tf.ones_like(sample_weight, dtype=tf.float32)
    zeros = tf.zeros_like(sample_weight, dtype=tf.float32)

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
    ]

    metric_name, preds, labels, weights, metric_type = zip(*targets)

    # 5. define optimizer
    loss = tf.losses.log_loss(labels, preds, weights, reduction="weighted_sum")
    optimizer = kai.nn.optimizer.Adam(1e-3)
    optimizer.minimize(loss)

    recall_type = kai.nn.get_dense_fea("recall_type", dim=1, dtype=tf.float32)
    # ones = tf.ones_like(expand_label, dtype=tf.float32)
    # zeros = tf.zeros_like(expand_label, dtype=tf.float32)


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

        ('expand_hot', expand_xtr, expand_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('like_hot', like_xtr, like_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('reply_hot', reply_xtr, reply_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('copy_hot', copy_xtr, copy_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('share_hot', share_xtr, share_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('audience_hot', audience_xtr, audience_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),
        ('continuous_expand_hot', continuous_expand_xtr, continuous_expand_label, tf.where(tf.less_equal(recall_type, 1.1), tf.where(tf.less_equal(0.9, recall_type), ones, zeros), zeros), "auc"),

        ('expand_climb', expand_xtr, expand_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('like_climb', like_xtr, like_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('reply_climb', reply_xtr, reply_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('copy_climb', copy_xtr, copy_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('share_climb', share_xtr, share_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('audience_climb', audience_xtr, audience_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),
        ('continuous_expand_climb', continuous_expand_xtr, continuous_expand_label, tf.where(tf.less_equal(recall_type, 4.1), tf.where(tf.less_equal(1.1, recall_type), ones, zeros), zeros), "auc"),

        ('pic_expand_predict', expand_xtr, expand_label, pic_comment, "auc"),
        ('pic_like_predict', like_xtr, like_label, pic_comment, "auc"),
        ('pic_reply_predict', reply_xtr, reply_label, pic_comment, "auc"),
        ('pic_copy_predict', copy_xtr, copy_label, pic_comment, "auc"),
        ('pic_share_predict', share_xtr, share_label, pic_comment, "auc"),
        ('pic_audience_predict', audience_xtr, audience_label, pic_comment, "auc"),
        ('pic_continuous_expand_predict', continuous_expand_xtr, continuous_expand_label, pic_comment, "auc"),

        ('text_expand_predict', expand_xtr, expand_label,  1 - pic_comment, "auc"),
        ('text_like_predict', like_xtr, like_label, 1 - pic_comment, "auc"),
        ('text_reply_predict', reply_xtr, reply_label, 1 - pic_comment, "auc"),
        ('text_copy_predict', copy_xtr, copy_label, 1-pic_comment, "auc"),
        ('text_share_predict', share_xtr, share_label, 1-pic_comment, "auc"),
        ('text_audience_predict', audience_xtr, audience_label, 1-pic_comment, "auc"),
        ('text_continuous_expand_predict', continuous_expand_xtr, continuous_expand_label, 1-pic_comment, "auc"),
    ]


    class TensorPrintHook(kai.training.RunHookBase):
        def __init__(self, debug_tensor_map):
            self.has_print = False
            self.debug_tensor_map = debug_tensor_map

        def begin(self, stream_context):
            pass
            # print("TensorFlow version:", self.debug_tensor_map['tf_version'])

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

        def before_step_run(self, step_run_context):
            return kai.training.StepRunArgs(fetches=self.debug_tensor_map)

        def after_step_run(self, step_run_context, step_run_values):
            # if not self.has_print:
            #     for k, v in step_run_values.result.items():
            #         print(f"{k} = {v}")
            #     self.has_print = True
            pass

    debug_tensor = {
        # "mmu_hetu_content_emb": tf.slice(mmu_hetu_content_emb, [0, 0], [1, -1]),
        # "mmu_clip_content_emb": tf.slice(mmu_clip_content_emb, [0, 0], [1, -1]),
        # "mmu_bert_content_emb": tf.slice(mmu_bert_content_emb, [0, 0], [1, -1]),
        'used_features': used_features,
    }
    kai.add_run_hook(TensorPrintHook(debug_tensor), "debug_tensor_hook")


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

