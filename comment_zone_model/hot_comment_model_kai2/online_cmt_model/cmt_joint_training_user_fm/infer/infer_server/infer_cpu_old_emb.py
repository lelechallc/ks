""" 用于model_add_duration.py
"""
import os
import sys
import json
import yaml
import argparse
import base64
import collections
import uuid
import socket
from datetime import datetime

current_folder = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(current_dir, '../../../../ks/common_reco/leaf/tools/pypi/'))

from dragonfly.common_leaf_dsl import LeafService, LeafFlow
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.gsu.gsu_api_mixin import GsuApiMixin
from dragonfly.ext.uni_predict.uni_predict_api_mixin import UniPredictApiMixin
from dragonfly.ext.kgnn.node_attr_schema import NodeAttrSchema
from dragonfly.ext.kgnn.kgnn_api_mixin import KgnnApiMixin
# from dragonfly.ext.se_reco.se_reco_api_mixin import SeRecoApiMixin
# from dragonfly.ext.embedding.embedding_api_mixin import EmbeddingApiMixin

parser = argparse.ArgumentParser()
parser.add_argument('--run', action='store_true')
args = parser.parse_args()


# 模型预估目标
PXTRS = ["expand_xtr", "like_xtr", "reply_xtr", "copy_xtr", "share_xtr", "audience_xtr", "continuous_expand_xtr", 'duration_predict', 'hate_xtr', 'report_xtr']
# 最终krp部署时会覆盖这个kess_name
infer_kess_name = "grpc_HotCommentXtrJointBertInferSlot"

# 请求带过来的attrs
item_attrs_from_req = [
  "like_cnt", "reply_cnt", "realshow_cnt", "author_id", "minute_diff", "dislike_cnt", "comment_genre", 'content_length','comment_content'
]
common_attrs_from_req = ["photo_id", "photo_author_id", "gender", "age_segment", "mod", "page_type_str",]
model_btq_prefix = "cmt_rank_joint_bert_emb"


embed_server_shards = 2
# krp 上 embedding_server的kess_name
embed_kess_name = "grpc_HotCommentXtrJointBertSlotEmb"
# colossusdb_embd_service_name="hot_comment_rank_staytime_gauss_colossusdb"
# colossusdb_embd_table_name="cmt-rank-bert-fetch-remote-emb-table"



# std::max(std::min(numerator / (denominator + smooth), max_val), min_value) * buckets
kuiba_discrete_converter = lambda denominator, smooth, max_val, buckets, min_val: {
  "converter": "discrete",
  "converter_args": f"{denominator},{smooth},{max_val},{buckets},{min_val}"
}

discreate_config = lambda attr_name, slot, bucket: {
  attr_name: {"attrs": [{"mio_slot_key_type": slot, "key_type": slot, "attr": [attr_name], **kuiba_discrete_converter(*bucket)}]},
}

id_config = lambda attr_name, slot: {
  attr_name: {"attrs": [{"mio_slot_key_type": slot, "key_type": slot, "attr": [attr_name], "converter": "id"}]},
}

id_config_slot = lambda attr_name, mio_slot_key_type, key_type: {
    attr_name: {"attrs": [
        {"mio_slot_key_type": mio_slot_key_type, "key_type": key_type, "attr": [attr_name], "converter": "id"}]},
}


class PredictServerFlow(LeafFlow, KuibaApiMixin, MioApiMixin, OfflineApiMixin, GsuApiMixin, UniPredictApiMixin, KgnnApiMixin):

  def prepare(self):
    return self \
      .deduplicate() \
      .count_reco_result(
        save_count_to="request_item_num"
      ) \
      .if_("request_item_num == 0") \
        .return_(0) \
      .end_() \
      .copy_user_meta_info(
        save_user_id_to_attr="user_id",
        save_device_id_to_attr="device_id"
      ) \
      .copy_item_meta_info(
        save_item_key_to_attr="comment_id",
      ) \
      .set_attr_value(
        item_attrs=[
          {
            "name": "showAction",
            "type": "int",
            "value": 1
          }
        ]
      ) \
      .enrich_attr_by_lua(
        import_item_attr=["like_cnt", "reply_cnt", "realshow_cnt"],
        export_item_attr=["ltr", "rtr"],
        function_for_item="cal_xtr",
        lua_script="""
            function cal_xtr()
                local vv = realshow_cnt or 0.0
                local ltr = like_cnt / (vv + 1.0)
                local rtr = reply_cnt / (vv + 1.0)
                return ltr, rtr
            end
        """
      ) \


  def predict_with_mio_model(self, **kwargs):
    model_config = kwargs.pop('model_config')
    embedding_kess_name = kwargs.pop('embedding_kess_name')
    # colossusdb_embd_service_name = kwargs.pop('colossusdb_embd_service_name')
    # colossusdb_embd_table_name = kwargs.pop('colossusdb_embd_table_name')
    queue_prefix = kwargs.pop('queue_prefix')
    key = kwargs.pop('key', queue_prefix)
    receive_dnn_model_as_macro_block = kwargs.pop('receive_dnn_model_as_macro_block', False)
    extra_inputs = kwargs.pop('extra_inputs', [])
    embedding_protocol = kwargs.pop('embedding_protocol', 0)
    shards = kwargs.pop('shards', 4)
    extra_signs = kwargs.pop('extra_signs', [])
    extra_slots = kwargs.pop('extra_slots', [])
    use_scale_int8 = kwargs.pop('use_scale_int8', False)
    
    fix_inputs = []
    for c in model_config.slots_config:
      print(c)
      attr = dict(
        attr_name = c['input_name'],
        tensor_name = c['input_name'],
        dim = len(str(c['slots']).split(' ')) * c['dim'] * c.get('expand', 1) + (1 if c.get('sized', False) else 0)
      )
      if c.get('compress_group', None) and c.get('compress_group') == "USER":
        attr['compress_group'] = "USER"
      fix_inputs.append(attr)
    batch_sizes = kwargs.pop('batch_sizes', [])
    implicit_batch = kwargs.pop('implicit_batch', True)
    use_scale_int8 = kwargs.pop('use_scale_int8', False)
    
    return self.enrich_attr_by_lua(
        import_item_attr=["comment_id"],
        export_item_attr=["bert_id"],
        function_for_item="copy",
        lua_script="""
            function copy()
                return comment_id
            end
        """
    ) \
    .extract_kuiba_parameter(
        config={
            **id_config("gender", 101),
            **id_config("age_segment", 102),

            **id_config("photo_id", 103),
            **id_config_slot("photo_author_id", 104, 202),
            **id_config_slot("user_id", 105, 202),
            **id_config("device_id", 106),

            # ## new feature
            **id_config("mod", 110),        # dim=32
            **id_config("page_type_str", 111),  
        },
        is_common_attr=True,
        slots_output="comment_common_slots",
        parameters_output="comment_common_signs",
      ) \
      .extract_kuiba_parameter(
        config={
            **id_config("comment_id", 201),
            **id_config("author_id", 202),
            **discreate_config("like_cnt", 203, [5, 0, 100000, 1, 0]),
            **discreate_config("reply_cnt", 204, [5, 0, 100000, 1, 0]),
            **discreate_config("minute_diff", 205, [36, 0, 336, 1, 0]),
            **discreate_config("ltr", 206, [0.001, 0, 1000, 1, 0]),
            **discreate_config("rtr", 207, [0.001, 0, 1000, 1, 0]),
            **id_config("showAction", 208),
            **discreate_config("dislike_cnt", 209, [3, 0, 100000, 1, 0]),

            # ## new feature
            **id_config("comment_genre", 250),      # dim=8
            **discreate_config("content_length", 251, [5, 0, 1000, 1, 0]),      # dim=32
            **id_config("bert_id", 252), 
        },
        is_common_attr=False,
        slots_output="comment_item_slots",
        parameters_output="comment_item_signs",
      ) \
      .uni_predict_fused(
        embedding_fetchers = [
          dict(
            fetcher_type="BtEmbeddingServerFetcher", 
            kess_service=embedding_kess_name,
            shards=shards,
            # colossusdb_embd_service_name=colossusdb_embd_service_name,
            # colossusdb_embd_table_name=colossusdb_embd_table_name,
            client_side_shard=True,
            slots_inputs=["comment_item_slots"],
            parameters_inputs=["comment_item_signs"],
            common_slots_inputs=["comment_common_slots"],
            common_parameters_inputs=["comment_common_signs"],
            slots_config=[dict(dtype='scale_int8' if use_scale_int8 else 'mio_int16', **sc) for sc in model_config.slots_config],
            max_signs_per_request=500,
            timeout_ms=50,
          )
        ],
        graph=model_config.graph,													# 原图，不需要任何操作
        model_loader_config=dict(							 						## 模型加载相关的配置
            rowmajor=True,								 						# 注意模型是否为 rowmajor
            type='MioTFExecutedByTensorFlowModelLoader',	# 使用 TFModelLoader
            executor_batchsizes=batch_sizes,								# 加载模型的 batch size 设置
            implicit_batch=implicit_batch,								# 是否为 implicit batch，implicit batch 的情况下不支持 XLA 和 Step2 中的 compress_group
            receive_dnn_model_as_macro_block=receive_dnn_model_as_macro_block
        ),
        batching_config = dict(
          batch_timeout_micros=0,
          max_batch_size=max(batch_sizes),
          max_enqueued_batches=1,
          batch_task_type="BatchTensorflowTask",
        ),
        executor_config=dict(
          intra_op_parallelism_threads_num=32,
          inter_op_parallelism_threads_num=32,
        ),
        inputs=fix_inputs + [dict(
                attr_name=attr_name,
                tensor_name=tensor_name,
                common=common,
                dim=dim,
              ) for attr_name, tensor_name, common, dim in extra_inputs],
        queue_prefix=queue_prefix,
        key=key,
        outputs = [dict(
                     attr_name=attr_name,
                     tensor_name=tensor_name,
                     common=False
                   ) for attr_name, tensor_name in model_config.outputs if attr_name in PXTRS],
        param = model_config.param,
        debug_tensor=False   # disable it online, to save time consume
      )

# load Resources
ModelConfig = collections.namedtuple('ModelConfig', ['graph', 'outputs', 'slots_config', 'param',
                                                     'common_slots', 'non_common_slots', 'feature_list'])
def load_mio_tf_model(model_dir):
  with open(os.path.join(model_dir, 'dnn_model.yaml')) as f:
    dnn_model = yaml.load(f, Loader=yaml.SafeLoader)

  with open(os.path.join(model_dir, 'graph.pb'), 'rb') as f:
    base64_graph = base64.b64encode(f.read()).decode('ascii')
    graph = 'base64://' + base64_graph

  feature_list = []

  graph_tensor_mapping = dnn_model['graph_tensor_mapping']
  extra_preds = dnn_model['extra_preds'].split(' ')
  q_names = dnn_model['q_names'].split(' ')
  assert len(extra_preds) == len(q_names)
  outputs = [(extra_pred, graph_tensor_mapping[q_name]) for extra_pred, q_name in zip(extra_preds, q_names)]
  param = [param for param in dnn_model['param'] if param.get('send_to_online', True)]
  print("param=", param)

  slots_config = dnn_model['embedding']['slots_config']
  for sc in slots_config:
    if 'dtype' in sc:
      sc['tensor_dtype'] = sc['dtype']
      del sc['dtype']
  common_slots = set()
  non_common_slots = set()
  for c in slots_config:
    slots = map(int, str(c['slots']).split(' '))
    if c.get('common', False):
      common_slots.update(slots)
    else:
      non_common_slots.update(slots)
  print("common_slots: ", common_slots)
  print("non_common_slots: ", non_common_slots)
  return ModelConfig(graph, outputs, slots_config, param,
                     common_slots, non_common_slots, feature_list)

service = LeafService(kess_name=infer_kess_name,
                      item_attrs_from_request=item_attrs_from_req,
                      common_attrs_from_request=common_attrs_from_req)

predict_graph_file_path = "./predict/config"
interact_open_predict_config = load_mio_tf_model(predict_graph_file_path)

comment_action_predict_flow = PredictServerFlow(name="comment_new_label") \
  .prepare() \
  .predict_with_mio_model(
    model_config=interact_open_predict_config,
    embedding_kess_name=embed_kess_name,
    # colossusdb_embd_service_name=colossusdb_embd_service_name,
    # colossusdb_embd_table_name=colossusdb_embd_table_name,
    shards=embed_server_shards,
    queue_prefix=model_btq_prefix,
    receive_dnn_model_as_macro_block=True,
    embedding_protocol=0,
    batch_sizes=[512, 1024, 2048, 4096],
    implicit_batch=True,
  )

# Infer服务最终返回的值
service.return_item_attrs(["expand_xtr", "like_xtr", "reply_xtr", "copy_xtr", "share_xtr", "audience_xtr", "continuous_expand_xtr", 'duration_predict', 'hate_xtr', 'report_xtr'])
service.add_leaf_flows(leaf_flows = [comment_action_predict_flow], request_type="hot_comment_default")
service.build(output_file=os.path.join(current_folder, "infer_cpu_old_emb_config.json"))

