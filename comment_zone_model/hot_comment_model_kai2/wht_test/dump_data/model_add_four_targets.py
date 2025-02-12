""" 个性化v2. 已上线
在个性化模型udp model基础上增加copy、share、audience、连续展开目标; 增加了photo_author_id特征；去掉了时序区样本（因为还没有这三个目标）；
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'predict'], default='train')
parser.add_argument('--dryrun', dest="dryrun", const=True, default=False, nargs='?')
parser.add_argument('--with_kai', action="store_true")
args = parser.parse_args()

# 1. define sparse input
if args.mode == 'train':
    import tensorflow.compat.v1 as tf
    import kai.tensorflow as kai

    user_embedding = kai.nn.new_embedding("user_embedding", dim=4, slots=[101, 102])
    comment_id_embedding = kai.nn.new_embedding("c_id_embedding", dim=64, slots=[201])
    author_id_embedding = kai.nn.new_embedding("a_id_embedding", dim=64, slots=[202])
    comment_info_embedding = kai.nn.new_embedding("c_info_embedding", dim=32, slots=[203, 204, 205, 206, 207, 209])
    position_embedding = kai.nn.new_embedding("position_embedding", dim=8, slots=[208])
    comment_udp_id_embedding = kai.nn.new_embedding("c_udp_id_embedding", dim=64, slots=[103, 104, 105, 106])

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
    comment_id_embedding = config.new_embedding("c_id_embedding", dim=64, slots=[201, 202])
    comment_info_embedding = config.new_embedding("c_info_embedding", dim=32, slots=[203, 204, 205, 206, 207, 209])
    position_embedding = config.new_embedding("position_embedding", dim=8, slots=[208])
    comment_udp_id_embedding = config.new_embedding("c_udp_id_embedding", dim=64, slots=[103, 104, 105, 106], **compress_kwargs)


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
    
# define model structure
field_input = tf.concat([user_embedding, comment_id_embedding, author_id_embedding, comment_info_embedding, position_embedding, comment_udp_id_embedding], -1)
expand_xtr = simple_dense_network("expand_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
like_xtr = simple_dense_network("like_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
reply_xtr = simple_dense_network("reply_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
copy_xtr = simple_dense_network("copy_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
share_xtr = simple_dense_network("share_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
audience_xtr = simple_dense_network("audience_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)
continuous_expand_xtr = simple_dense_network("continuous_expand_xtr", field_input, [256, 128, 64, 1], 0.0, act=tf.nn.leaky_relu, last_act=tf.nn.sigmoid)


if args.mode == 'train':

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

    # recall_type = tf.cast(kai.nn.get_dense_fea("recall_type", dim=1, dtype=tf.int64), tf.float32)

    # comment_genre = tf.cast(kai.nn.get_dense_fea("comment_genre", dim=1, dtype=tf.int64), tf.float32)
    # pic_comment = tf.where(comment_genre > 0, ones, zeros)
    

    eval_targets = [
        ('expand_predict', expand_xtr, expand_label, ones, "auc"),
        ('like_predict', like_xtr, like_label, ones, "auc"),
        ('reply_predict', reply_xtr, reply_label, ones, "auc"),
        ('copy_predict', copy_xtr, copy_label, ones, "auc"),
        ('share_predict', share_xtr, share_label, ones, "auc"),
        ('audience_predict', audience_xtr, audience_label, ones, "auc"),
        ('continuous_expand_predict', continuous_expand_xtr, continuous_expand_label, ones, "auc"),
    ]
    

    from kai.tensorflow.utils import data_table
    class DumpTensorHook(kai.training.RunHookBase):
        def __init__(self, table_name, dump_tensors_dict):
            """
                本Hook用于获取tf图中dump_tensors_dict对应的tensor数据，导出到HDFS上
            Args:
                table_name (string): 表名
                dump_tensors_dict (dict): 需要导出的tensor数据，dict(tensor_name, tensor_op)
            """
            assert isinstance(dump_tensors_dict, dict)
            worker_id = kai.current_rank()
            model_path = kai.Config().save_option.model_path
            # 新建一个表
            self._dump_table = data_table.DataTable(
                table_name=table_name, worker_id=worker_id, model_path=model_path)
            self._dump_tensors_dict = dump_tensors_dict

            self.cid_cnt = {}
            self.cid_pos_cnt = {}
            self.cid_first_appear_idx = {}
            self.batch_cnt = 0
            self.sample_cnt = 0

        def before_step_run(self, step_run_context):
            """
                将 self._dump_tensors_dict 中的tensor注入fetches中
                后续step run图时会自动跑出来对应Tensor的数值

            Args:
                step_run_context (_type_): _description_

            Returns:
                _type_: _description_
            """
            return kai.training.StepRunArgs(fetches=self._dump_tensors_dict)

        def after_step_run(self, step_run_context, step_run_values):
            """
                获取run图的结果，将结果写入表中

            Args:
                step_run_context (_type_): _description_
                step_run_values (_type_): _description_
            """
            # sink_data = {}
            # for name, op in self._dump_tensors_dict.items():
            #     value = step_run_values.result[name]
            #     batch_size = value.shape[0]
            #     sink_data[name] = value.reshape(batch_size, -1)
            
            # cids = step_run_values.result['cids']
            # is_select_cid=step_run_values.result['is_select_cid']
            # cid_emb=step_run_values.result['cid_emb']
            # binary_label=step_run_values.result['binary_label']
            # batch_size=cids.shape[0]

            # is_select_cid=tf.squeeze(tf.cast(is_select_cid, tf.bool))
            # cid_selected=tf.boolean_mask(cids, is_select_cid)
            # emb_selected=tf.boolean_mask(cid_emb, is_select_cid)
            # emb_norm_selected = tf.norm(emb_selected, axis=1)
            # binary_label_selected = tf.boolean_mask(binary_label, is_select_cid)

            cid_selected = step_run_values.result['cid_selected']
            dummy_cid_selected = step_run_values.result['dummy_cid_selected']
            emb_norm_selected = step_run_values.result['emb_norm_selected']
            binary_label_selected = step_run_values.result['binary_label_selected']
            batch_size = step_run_values.result['cids'].shape[0]

            binary_label_selected = binary_label_selected.tolist()

            sink_cid = cid_selected.tolist()
            sink_dummy_cid = dummy_cid_selected.tolist()
            sink_emb_norm = emb_norm_selected.tolist()
            sink_cid_sample_num = [self.cid_cnt[cid] if cid in self.cid_cnt else 0 for cid in sink_cid]
            sink_cid_pos_sample_num = [self.cid_pos_cnt[cid] if cid in self.cid_pos_cnt else 0 for cid in sink_cid]
            sink_cid_first_idx = [self.cid_first_appear_idx[cid] if cid in self.cid_first_appear_idx else self.sample_cnt for cid in sink_cid]

            assert len(sink_cid) == len(sink_emb_norm) == len(sink_cid_sample_num) == len(sink_cid_pos_sample_num) == len(sink_cid_first_idx)
            sink_data={
                'cid': sink_cid,
                'dummy_cid': sink_dummy_cid,
                'emb_norm': sink_emb_norm,
                'cid_sample_num': sink_cid_sample_num,
                'cid_pos_sample_num': sink_cid_pos_sample_num,
                'first_index': sink_cid_first_idx,
                'cum_sample_num': [self.sample_cnt] * len(sink_cid),
                'cum_batch_num': [self.batch_cnt] * len(sink_cid),
            }

            if len(sink_cid)>0:
                print('dump-data...')
                print(sink_data)
                self._dump_table.append_batch(sink_data)
                print('dump-data done!')

            for i, (cid, label) in enumerate(zip(sink_cid, binary_label_selected)):
                if cid not in self.cid_cnt:
                    self.cid_cnt[cid] = 0
                self.cid_cnt[cid]+=1
                if cid not in self.cid_pos_cnt:
                    self.cid_pos_cnt[cid] = 0
                if label > 0:
                    self.cid_pos_cnt[cid]+=1
                if cid not in self.cid_first_appear_idx:
                    self.cid_first_appear_idx[cid] = self.sample_cnt + i
                
                
            self.batch_cnt+=1
            self.sample_cnt+=batch_size

            # step_id = step_run_context.descr_list.step
            # pass_id = step_run_context.descr_list.pass_id
            # sink_data["step_id"] = [step_id] * batch_size
            # sink_data["pass_id"] = [pass_id] * batch_size
            # self._dump_table.append_batch(sink_data)

    cids = tf.cast(kai.nn.get_dense_fea("comment_id", dim=1, dtype=tf.float32), tf.int64)
    dummy_cids = tf.cast(kai.nn.get_dense_fea("dummy_cid", dim=1, dtype=tf.float32), tf.int64)

    is_select_cid = tf.cast(kai.nn.get_dense_fea("is_select_cid", dim=1, dtype=tf.float32), tf.int64)
    binary_label = tf.where((expand_label > 0) | (like_label > 0) | (reply_label>0) | (copy_label>0) | (share_label>0) | (audience_label>0) | (continuous_expand_label>0), ones, zeros)


    cid_emb=comment_id_embedding

    is_select_cid=tf.reshape(tf.cast(is_select_cid, tf.bool), [-1])
    
    cid_selected=tf.boolean_mask(cids, is_select_cid)
    dummy_cid_selected = tf.boolean_mask(dummy_cids, is_select_cid)
    emb_selected=tf.boolean_mask(cid_emb, is_select_cid)
    emb_norm_selected = tf.norm(emb_selected, axis=1)
    binary_label_selected = tf.boolean_mask(binary_label, is_select_cid)

    cid_selected=tf.reshape(cid_selected, [-1])
    dummy_cid_selected=tf.reshape(dummy_cid_selected, [-1])
    binary_label_selected=tf.reshape(binary_label_selected, [-1])

    kai.add_run_hook(DumpTensorHook('dump_tensors', {
        'cid_selected': cid_selected,
        'dummy_cid_selected': dummy_cid_selected,
        # 'emb_selected': emb_selected,
        'emb_norm_selected': emb_norm_selected,
        'binary_label_selected': binary_label_selected,
        'cids': cids

    }), 'custom_dump_tensor_hook')

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

