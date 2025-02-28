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

import lua_script

current_folder = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(current_dir, '../../../../ks/common_reco/leaf/tools/pypi/'))

from dragonfly.common_leaf_dsl import LeafService, LeafFlow
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.gsu.gsu_api_mixin import GsuApiMixin
from dragonfly.ext.uni_predict.uni_predict_api_mixin import UniPredictApiMixin
from dragonfly.ext.kgnn.kgnn_api_mixin import KgnnApiMixin

parser = argparse.ArgumentParser()
parser.add_argument('--run', action='store_true')
args = parser.parse_args()

PXTRS = ["like_weight", "follow_weight", "forward_weight", "comment_weight", "collect_weight", "download_weight"]
# 最终krp部署时会覆盖这个kess_name
infer_kess_name = "grpc_slideRecoInteractWeightsInfer"
# 请求带过来的attrs
pxtr_list = ["pltr","pwtr", "pftr", "pcmtr", "pcltr", "pdtr"]
pxtrs_list = [xtr+'s' for xtr in pxtr_list]
cascade_pxtr_list = ["cascade_pctr", "cascade_pltr", "cascade_plvtr", "cascade_psvr"]

item_attrs_from_req = ["reason", "photo_info_str"] + pxtr_list + cascade_pxtr_list
common_attrs_from_req = ["tab_id", "user_info_str"]
model_btq_prefix = "topic_slide_reco_interact_weights"
embed_server_shards = 2
# krp 上 embedding_server的kess_name
embed_kess_name = "grpc_slideRecoInteractWeightsEmbed"


def load_feature_list_sign(filename):
  ret = set()
  with open(filename) as f:
    for line in f:
      if line.startswith('#') or not line.strip():
        continue
      for field in line.strip().split(','):
        parts = field.strip().split('=')
        assert len(parts) == 2, "Unsupported format: " + line.strip()

        if parts[0].strip() != 'class':
          # ignore unknown field
          continue

        ret.add(parts[1].strip())
  return list(sorted(ret))

class PredictServerFlow(LeafFlow, KuibaApiMixin, MioApiMixin, OfflineApiMixin, GsuApiMixin, UniPredictApiMixin, KgnnApiMixin):

  def prepare(self):
    return self \
      .deduplicate() \
      .parse_protobuf_from_string(
        is_common_attr=True,
        input_attr="user_info_str",
        output_attr="user_info",
        class_name="ks::reco::UserInfo") \
      .parse_protobuf_from_string(
          is_common_attr=False,
          input_attr="photo_info_str",
          output_attr="photo_info",
          class_name="ks::reco::PhotoInfo") \
      .filter_by_attr(attr_name="photo_info", remove_if_attr_missing=True) \
      .enrich_attr_by_lua(
        import_item_attr = ["reason"],
        function_for_item = "change",
        export_item_attr = ["reason_string"],
        lua_script = """
          function change()
            local reason = reason or 0
            return tostring(reason)
          end
        """
      ) \
      .build_protobuf( # 为了extract_with_ks_sign_feature抽特征
        class_name="ks::reco::ContextInfo",
        inputs=[
          { "item_attr": "cascade_pctr", "path": "cascade_pctr" },
          { "item_attr": "cascade_pltr", "path": "cascade_pltr" },
          { "item_attr": "cascade_psvr", "path": "cascade_psvr" },
          { "item_attr": "cascade_plvtr", "path": "cascade_plvtr" },
        ],
        output_item_attr="context_info",
      )

  def extract_feature(self):
    return self \
      .enrich_with_protobuf(
        from_extra_var="user_info",
        is_common_attr=True,
        attrs=[
          dict(name="click_photo_ids", path="user_profile_v1.click_list.photo_id"),
          dict(name="click_hetu_one", path="user_profile_v1.click_list.hetu_tag_level_info.hetu_level_one",
                repeat_limit={"user_profile_v1.click_list.hetu_tag_level_info.hetu_level_one": 1},
                repeat_align=True),
          dict(name="click_hetu_two", path="user_profile_v1.click_list.hetu_tag_level_info.hetu_level_two",
                repeat_limit={"user_profile_v1.click_list.hetu_tag_level_info.hetu_level_two": 1},
                repeat_align=True),

          dict(name="like_photo_ids", path="user_profile_v1.like_list.photo_id"),
          dict(name="like_hetu_one", path="user_profile_v1.like_list.hetu_tag_level_info.hetu_level_one",
                repeat_limit={"user_profile_v1.like_list.hetu_tag_level_info.hetu_level_one": 1},
                repeat_align=True),
          dict(name="like_hetu_two", path="user_profile_v1.like_list.hetu_tag_level_info.hetu_level_two",
                repeat_limit={"user_profile_v1.like_list.hetu_tag_level_info.hetu_level_two": 1},
                repeat_align=True),
          
          dict(name="follow_photo_ids", path="user_profile_v1.follow_list.photo_id"),
          dict(name="follow_hetu_one", path="user_profile_v1.follow_list.hetu_tag_level_info.hetu_level_one",
                repeat_limit={"user_profile_v1.follow_list.hetu_tag_level_info.hetu_level_one": 1},
                repeat_align=True),
          dict(name="follow_hetu_two", path="user_profile_v1.follow_list.hetu_tag_level_info.hetu_level_two",
                repeat_limit={"user_profile_v1.follow_list.hetu_tag_level_info.hetu_level_two": 1},
                repeat_align=True),

          dict(name="forward_photo_ids", path="user_profile_v1.forward_list.photo_id"),
          dict(name="forward_hetu_one", path="user_profile_v1.forward_list.hetu_tag_level_info.hetu_level_one",
                repeat_limit={"user_profile_v1.forward_list.hetu_tag_level_info.hetu_level_one": 1},
                repeat_align=True),
          dict(name="forward_hetu_two", path="user_profile_v1.forward_list.hetu_tag_level_info.hetu_level_two",
                repeat_limit={"user_profile_v1.forward_list.hetu_tag_level_info.hetu_level_two": 1},
                repeat_align=True),

          dict(name="comment_photo_ids", path="user_profile_v1.comment_list.photo_id"),
          dict(name="comment_hetu_one", path="user_profile_v1.comment_list.hetu_tag_level_info.hetu_level_one",
                repeat_limit={"user_profile_v1.comment_list.hetu_tag_level_info.hetu_level_one": 1},
                repeat_align=True),
          dict(name="comment_hetu_two", path="user_profile_v1.comment_list.hetu_tag_level_info.hetu_level_two",
                repeat_limit={"user_profile_v1.comment_list.hetu_tag_level_info.hetu_level_two": 1},
                repeat_align=True),
          
          dict(name="collect_photo_ids", path="user_profile_v1.collect_list.photo_id"),
          dict(name="collect_hetu_one", path="user_profile_v1.collect_list.hetu_tag_level_info.hetu_level_one",
                repeat_limit={"user_profile_v1.collect_list.hetu_tag_level_info.hetu_level_one": 1},
                repeat_align=True),
          dict(name="collect_hetu_two", path="user_profile_v1.collect_list.hetu_tag_level_info.hetu_level_two",
                repeat_limit={"user_profile_v1.collect_list.hetu_tag_level_info.hetu_level_two": 1},
                repeat_align=True),
          
          dict(name="download_photo_ids", path="user_profile_v1.download_video_list.photo_id"),
          dict(name="download_hetu_one", path="user_profile_v1.download_video_list.hetu_tag_level_info.hetu_level_one",
                repeat_limit={"user_profile_v1.download_video_list.hetu_tag_level_info.hetu_level_one": 1},
                repeat_align=True),
          dict(name="download_hetu_two", path="user_profile_v1.download_video_list.hetu_tag_level_info.hetu_level_two",
                repeat_limit={"user_profile_v1.download_video_list.hetu_tag_level_info.hetu_level_two": 1},
                repeat_align=True),
          # dict(name="human_action", path="device_stat.human_action"),
          # dict(name="device_status_flags", path="device_stat.device_status_flags"),
          # dict(name="FollowList", path="user_profile_v1.follow_list.author_id"),
          # dict(name="BidFollowList", path="friend_info_v2.bid_follow_list.friend_id"),
      ]) \
      .enrich_with_protobuf(
        from_extra_var="photo_info",
        is_common_attr=False,
        attrs=[
          dict(name = "photo_hetu_one", path="hetu_tag_level_info.hetu_level_one",
                repeat_limit={"hetu_tag_level_info.hetu_level_one": 1},
                repeat_align=True),
          dict(name = "photo_hetu_two", path="hetu_tag_level_info.hetu_level_two",
                repeat_limit={"hetu_tag_level_info.hetu_level_two": 1},
                repeat_align=True),
        ]
      ) \
  
  def predict_with_mio_model(self, **kwargs):
    model_config = kwargs.pop('model_config')
    embedding_kess_name = kwargs.pop('embedding_kess_name')
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
    
    # attr_name, tensor_name, common, dim 
    extra_inputs = []
    for xtr, xtrs in zip(pxtr_list, pxtrs_list):
      extra_inputs.append((xtrs, xtr, False, 1))
    class PredictSubFlow(LeafFlow, KuibaApiMixin, MioApiMixin, OfflineApiMixin, GsuApiMixin, UniPredictApiMixin, KgnnApiMixin):
       pass

    predict_subflow = PredictSubFlow(name="predict_sub_flow") \
      .extract_with_ks_sign_feature(
        feature_list=model_config.feature_list,
        user_info_attr="user_info",
        photo_info_attr="photo_info",
        context_info_attr="context_info",
        tab_id_attr="tab_id",
        reason_attr="reason_string",
        cascade_pctr_attr="cascade_pctr",
        cascade_plvtr_attr="cascade_plvtr",
        cascade_psvr_attr="cascade_psvr",
        cascade_pltr_attr="cascade_pltr",
        common_slots_output="common_slots",
        common_parameters_output="common_signs",
        item_slots_output="item_slots",
        item_parameters_output="item_signs",
      ) \
      .extract_kuiba_parameter(
        slots_output="kuiba_common_slots",
        parameters_output="kuiba_common_signs",
        is_common_attr=True,
        config=lua_script.COMMON_KUIBA_CONFIG,
        slot_as_attr_name=False
      ) \
      .extract_kuiba_parameter(
        slots_output="kuiba_item_slots",
        parameters_output="kuiba_item_signs",
        is_common_attr=False,
        config=lua_script.ITEM_KUIBA_CONFIG,
        slot_as_attr_name=False
      ) \
      .enrich_attr_by_lua(
        import_item_attr=pxtr_list,
        function_for_item="tolist",
        export_item_attr=pxtrs_list,
        lua_script="""
          function tolist()
            local pltr = pltr or 0.0
            local pwtr = pwtr or 0.0
            local pftr = pftr or 0.0
            local pcmtr = pcmtr or 0.0
            local pcltr = pcltr or 0.0
            local pdtr = pdtr or 0.0

            return {pltr}, {pwtr}, {pftr},  {pcmtr}, {pcltr}, {pdtr}
          end
        """) \
      .uni_predict_fused(
        embedding_fetchers = [
          dict(
            fetcher_type="BtEmbeddingServerFetcher", 
            kess_service=embedding_kess_name,
            shards=shards,
            client_side_shard=True,
            slots_inputs=["item_slots", "kuiba_item_slots"] + extra_slots,
            parameters_inputs=["item_signs", "kuiba_item_signs"] + extra_signs,
            common_slots_inputs=["common_slots", "kuiba_common_slots"],
            common_parameters_inputs=["common_signs", "kuiba_common_signs"],
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
                   ) for attr_name, tensor_name in model_config.outputs if attr_name in PXTRS],
        param = model_config.param,
      ) \
      .get_kconf_params( #kconf 获取 bound 来控制 sigmoid 后的输出
        kconf_configs=[{
            "kconf_key": "reco.GamoraInteract.weightModelBound",
            "export_common_attr": "like_weight_bound",
            "value_type": "double",
            "json_path": "like_weight_bound"
        },{
            "kconf_key": "reco.GamoraInteract.weightModelBound",
            "export_common_attr": "follow_weight_bound",
            "value_type": "double",
            "json_path": "follow_weight_bound"
        },{
            "kconf_key": "reco.GamoraInteract.weightModelBound",
            "export_common_attr": "forward_weight_bound",
            "value_type": "double",
            "json_path": "forward_weight_bound"
        },{
            "kconf_key": "reco.GamoraInteract.weightModelBound",
            "export_common_attr": "comment_weight_bound",
            "value_type": "double",
            "json_path": "comment_weight_bound"
        },{
            "kconf_key": "reco.GamoraInteract.weightModelBound",
            "export_common_attr": "collect_weight_bound",
            "value_type": "double",
            "json_path": "collect_weight_bound"
        },{
            "kconf_key": "reco.GamoraInteract.weightModelBound",
            "export_common_attr": "download_weight_bound",
            "value_type": "double",
            "json_path": "download_weight_bound"
        }]
    ) \
    .pack_item_attr(
        item_source = { "reco_results": True},
        mappings = [{
          "from_item_attr": "like_weight",
          "to_common_attr": "like_weight_list",
      },{
          "from_item_attr": "follow_weight",
          "to_common_attr": "follow_weight_list",
      },{
          "from_item_attr": "forward_weight",
          "to_common_attr": "forward_weight_list",
      },{
          "from_item_attr": "comment_weight",
          "to_common_attr": "comment_weight_list",
      },{
          "from_item_attr": "collect_weight",
          "to_common_attr": "collect_weight_list",
      },{
          "from_item_attr": "download_weight",
          "to_common_attr": "download_weight_list",
      }]
    ) \
    .enrich_attr_by_lua(
      import_common_attr=["like_weight_bound","follow_weight_bound","forward_weight_bound","comment_weight_bound","collect_weight_bound","download_weight_bound",
                        "like_weight_list","follow_weight_list","forward_weight_list","comment_weight_list","collect_weight_list","download_weight_list"],
      function_for_common="handle",
      export_common_attr = ["like_weight_list","follow_weight_list","forward_weight_list","comment_weight_list","collect_weight_list","download_weight_list"],
      lua_script = """
        function calc(weight_list, bound)
            for i=1,#weight_list do
                weight_list[i] = weight_list[i] - bound
            end
            return weight_list
        end
        function handle()
            like_weight_list = calc(like_weight_list, like_weight_bound)
            follow_weight_list = calc(follow_weight_list, follow_weight_bound)
            forward_weight_list = calc(forward_weight_list, forward_weight_bound)
            comment_weight_list = calc(comment_weight_list, comment_weight_bound)
            collect_weight_list = calc(collect_weight_list, collect_weight_bound)
            download_weight_list = calc(download_weight_list, download_weight_bound)
            return like_weight_list, follow_weight_list, forward_weight_list, comment_weight_list, collect_weight_list, download_weight_list
        end
      """
    ) \
    .dispatch_common_attr(
      dispatch_config = [
          {
          "from_common_attr" : "like_weight_list",
          "to_item_attr" : "like_weight"
          },
          {
          "from_common_attr" : "follow_weight_list",
          "to_item_attr" : "follow_weight"
          },
          {
          "from_common_attr" : "forward_weight_list",
          "to_item_attr" : "forward_weight"
          },
          {
          "from_common_attr" : "comment_weight_list",
          "to_item_attr" : "comment_weight"
          },
          {
          "from_common_attr" : "collect_weight_list",
          "to_item_attr" : "collect_weight"
          },
          {
          "from_common_attr" : "download_weight_list",
          "to_item_attr" : "download_weight"
          }
      ]
    )
    return self \
      .arrange_by_sub_flow(sub_flow=predict_subflow, expected_partition_size=256)

# load Resources
ModelConfig = collections.namedtuple('ModelConfig', ['graph', 'outputs', 'slots_config', 'param',
                                                     'common_slots', 'non_common_slots', 'feature_list'])
def load_mio_tf_model(model_dir):
  with open(os.path.join(model_dir, 'dnn_model.yaml')) as f:
    dnn_model = yaml.load(f, Loader=yaml.SafeLoader)

  with open(os.path.join(model_dir, 'graph.pb'), 'rb') as f:
    base64_graph = base64.b64encode(f.read()).decode('ascii')
    graph = 'base64://' + base64_graph

  feature_list = load_feature_list_sign(os.path.join(model_dir, 'feature_list_sign.txt'))

  graph_tensor_mapping = dnn_model['graph_tensor_mapping']
  extra_preds = dnn_model['extra_preds'].split(' ')
  q_names = dnn_model['q_names'].split(' ')
  assert len(extra_preds) == len(q_names)
  outputs = [(extra_pred, graph_tensor_mapping[q_name]) for extra_pred, q_name in zip(extra_preds, q_names)]
  param = [param for param in dnn_model['param'] if param.get('send_to_online', True)]

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

  return ModelConfig(graph, outputs, slots_config, param,
                     common_slots, non_common_slots, feature_list)

service = LeafService(kess_name=infer_kess_name,
                      item_attrs_from_request=item_attrs_from_req,
                      common_attrs_from_request=common_attrs_from_req)


service.AUTO_INJECT_ITEM_ATTR = False
service.AUTO_INJECT_SAMPLE_LIST_USER_ATTR = False
# Infer服务最终返回的值
service.return_item_attrs(PXTRS)

print(current_folder)
current_model_config = load_mio_tf_model(current_folder)

current_model_flow = PredictServerFlow(name = model_btq_prefix) \
  .prepare() \
  .extract_feature() \
  .predict_with_mio_model(
    model_config=current_model_config,
    embedding_kess_name=embed_kess_name,
    shards=embed_server_shards,
    queue_prefix=model_btq_prefix,
    receive_dnn_model_as_macro_block=True,
    embedding_protocol=0,
    batch_sizes=[512, 1024, 2048],
    implicit_batch=True,
  ) 

service.add_leaf_flows(leaf_flows = [current_model_flow],
                       request_type = model_btq_prefix)

if __name__ == "__main__":
  if args.run:
    test_service = LeafService(kess_name="grpc_testLeafService", item_attrs_from_request=[], common_attrs_from_request=[])
    test_service.add_leaf_flows(leaf_flows=[], request_type="test dryrun")
    test_service.build(output_file=os.path.join(current_folder, "test.json"))

    exe = test_service.executor()
    for i in range(10):
      exe.reset()
      exe.run("test_pipeline")
      for item in exe.items:
        print("item_key: ", item.item_key)
        for key in PXTRS:
          print(f"{key}: {item[key]}")
  else:
    service.build(output_file=os.path.join(current_folder, "infer_config.json"))

