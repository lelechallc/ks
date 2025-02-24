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

parser = argparse.ArgumentParser()
parser.add_argument('--run', action='store_true')
args = parser.parse_args()
# 模型预估目标
PXTRS = ["long_play_value", "effective_play_value", "play_time_min", "profile_stay_time_min", "like_value", 
                "follow_value", "forward_value", "comment_value", "profile_enter_value"]
# 最终krp部署时会覆盖这个kess_name
infer_kess_name = "grpc_FrLongtermAttributionV1Infer"
# 请求带过来的attrs
item_attrs_from_req = ["reason", "photo_info_str", 
  "pctr", "pltr", "pftr", "pwtr", "plvtr", "psvr", "pvtr", "pptr", "pcpr", 
  "emp_ctr", "emp_ltr", "emp_ftr", "emp_wtr",
  "cascade_pctr", "cascade_plvtr", "cascade_psvr", "cascade_pltr"]
common_attrs_from_req = ["tab_id", "user_info_str"]
embed_service_name = "fr_up_longterm_attribution_model_v1"
embed_table_name = "up_attribution_emb"
btq_queue_prefix = "fr_up_longterm_attribution_model_v1"

kuiba_list_converter_config_list_limit = lambda limit_n:  {
  "converter": "list",
  "type":5,
  "expire_second": 2592000,
  "use_common_attr_only": True,
  "converter_args": {
    "reversed": False,
    "enable_filter": False,
    "limit": limit_n,
  },
}
kuiba_id_converter = {
  "converter": "id"
}

kuiba_list_converter_config_list_limit = lambda limit_n:  {
  "converter": "list",
  "type":5,
  "expire_second": 2592000,
  "converter_args": {
    "reversed": False,
    "enable_filter": False,
    "limit": limit_n,
  },
}

id_config = lambda attr_name, slot: {
  attr_name: {"attrs": [{"mio_slot_key_type": slot, "key_type": slot, "attr": [attr_name], **kuiba_id_converter}]},
}

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

class PredictServerFlow(LeafFlow, KuibaApiMixin, MioApiMixin, OfflineApiMixin, GsuApiMixin, UniPredictApiMixin):

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
      )\
      .enrich_with_protobuf(
        from_extra_var="user_info",
        is_common_attr=True,
        attrs=[
          dict(name="like_pid_list", path="user_profile_v1.like_list.photo_id"),
          dict(name="follow_pid_list", path="user_profile_v1.follow_list.photo_id"),
          dict(name="click_pid_list", path="user_profile_v1.click_list.photo_id"),
        ]) \
      .enrich_with_protobuf(
        from_extra_var="photo_info",
        is_common_attr=False,
        attrs=[
          dict(name="author_upload_count",path="author.upload_count"),
          dict(name="author_dnn_cluster_id",path="author.user_profile.exp_stat.duration"),
          dict(name="author_category_type", path="live_photo_info.author_category_type"),
          dict(name="author_user_good_count",path="author.user_good_count"),
          dict(name="author_healthiness",path="author.healthiness"),
        ]
      ) \


  def predict_with_mio_model(self, **kwargs):
    model_config = kwargs.pop('model_config')
    
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
      # if c.get('input_name') in ['uid_emb', 'uid_action_list_click', 'uid_action_list_like', 'uid_action_list_follow']:
      #   attr['common'] = True
      if c.get('compress_group', None) and c.get('compress_group') == "USER":
        attr['compress_group'] = "USER"
      fix_inputs.append(attr)
    
    batch_sizes = kwargs.pop('batch_sizes', [])
    implicit_batch = kwargs.pop('implicit_batch', True)
    use_scale_int8 = kwargs.pop('use_scale_int8', False)
    print(fix_inputs)
   
    for attr_name, tensor_name in model_config.outputs:
        print("attr_name:", attr_name)
        print("tensor_name:", tensor_name)

    extra_inputs = [
      ("pctrs", "pctr", False, 1),
      ("pltrs", "pltr", False, 1),
      ("pftrs", "pftr", False, 1),
      ("pwtrs", "pwtr", False, 1),
      ("plvtrs", "plvtr", False, 1),
      ("pvtrs", "pvtr", False, 1),
      ("pptrs", "pptr", False, 1),
      ("empirical_ctrs", "empirical_ctr", False, 1),
      ("empirical_ltrs", "empirical_ltr", False, 1),
      ("empirical_ftrs", "empirical_ftr", False, 1),
      ("empirical_wtrs", "empirical_wtr", False, 1),
      ("author_healthiness_list", "author_healthiness", False, 1),
    ]


    class PredictSubFlow(LeafFlow, KuibaApiMixin, MioApiMixin, OfflineApiMixin, GsuApiMixin, UniPredictApiMixin):
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
        config={
           "pctr": {"attrs": [{"mio_slot_key_type": 1101, "key_type": 1101, "attr": ["pctr"], "converter": "discrete", "converter_args": "1,0,1,400,-1",}]},
            "pltr": {"attrs": [{"mio_slot_key_type": 1102, "key_type": 1102, "attr": ["pltr"], "converter": "discrete", "converter_args": "0.2,0,1,400,-1",}]},
            "pftr": {"attrs": [{"mio_slot_key_type": 1103, "key_type": 1103, "attr": ["pftr"], "converter": "discrete", "converter_args": "0.2,0,1,400,-1",}]},
            "pwtr": {"attrs": [{"mio_slot_key_type": 1104, "key_type": 1104, "attr": ["pwtr"], "converter": "discrete", "converter_args": "0.2,0,1,400,-1",}]},
            "plvtr": {"attrs": [{"mio_slot_key_type": 1105, "key_type": 1105, "attr": ["plvtr"], "converter": "discrete", "converter_args": "1,0,1,400,-1",}]},
            "pvtr": {"attrs": [{"mio_slot_key_type": 1107, "key_type": 1107, "attr": ["pvtr"], "converter": "discrete", "converter_args": "1,0,1,400,-1",}]},
            "pptr": {"attrs": [{"mio_slot_key_type": 1108, "key_type": 1108, "attr": ["pptr"], "converter": "discrete", "converter_args": "0.2,0,1,400,-1",}]},
            "empirical_ctr": {"attrs": [{"mio_slot_key_type": 1109, "key_type": 1109, "attr": ["emp_ctr"], "converter": "discrete", "converter_args": "1,0,1,200,-1",}]},
            "empirical_ltr": {"attrs": [{"mio_slot_key_type": 1110, "key_type": 1110, "attr": ["emp_ltr"], "converter": "discrete", "converter_args": "0.2,0,1,200,-1",}]},
            "empirical_ftr": {"attrs": [{"mio_slot_key_type": 1111, "key_type": 1111, "attr": ["emp_ftr"], "converter": "discrete", "converter_args": "0.2,0,1,200,-1",}]},
            "empirical_wtr": {"attrs": [{"mio_slot_key_type": 1112, "key_type": 1112, "attr": ["emp_wtr"], "converter": "discrete", "converter_args": "0.2,0,1,200,-1",}]},
            "author_upload_count": {"attrs":[{"key_type": 1200, "attr": ["author_upload_count"], "converter": "discrete", "converter_args": "1,0,100000,10,0"}]},
            **id_config("author_dnn_cluster_id",1201), 
            "author_category_type": {"attrs": [{"key_type": 1202, "attr": ["author_category_type"], "converter": "id"}]},
            "author_user_good_count": {"attrs": [{"key_type": 1203, "attr": ["author_user_good_count"], "converter": "discrete", "converter_args": "1,0,100,10,0"}]},
        },
      is_common_attr=False,
      slots_output="item_pxtr_slots",
      parameters_output="item_pxtr_signs",
    ) \
    .extract_kuiba_parameter(
     config={
       "click_pid_list": {"attrs": [{"mio_slot_key_type": 2201, "key_type": 2201, "attr": ["click_pid_list"], **kuiba_list_converter_config_list_limit(30)}]},
       "like_pid_list": {"attrs": [{"mio_slot_key_type": 2202, "key_type": 2202, "attr": ["like_pid_list"], **kuiba_list_converter_config_list_limit(30)}]},
       "follow_pid_list": {"attrs": [{"mio_slot_key_type": 2203, "key_type": 2203, "attr": ["follow_pid_list"], **kuiba_list_converter_config_list_limit(30)}]},
     },
     is_common_attr=True,
     slots_output="common_kuiba_slots",
     parameters_output="common_kuiba_signs",
     slot_as_attr_name=False,
   ) \
      .enrich_attr_by_lua(
        import_item_attr=["pctr", "pltr", "pftr", "pwtr", "plvtr", "pvtr", "pptr", 
                  "emp_ctr", "emp_ltr","emp_ftr","emp_wtr", "author_healthiness"],
        function_for_item="tolist",
        export_item_attr=["pctrs", "pltrs", "pftrs", "pwtrs", "plvtrs", "pvtrs", "pptrs",
                  "empirical_ctrs","empirical_ltrs","empirical_ftrs","empirical_wtrs", "author_healthiness_list"],
        lua_script="""
          function tolist()
            local pctr = pctr or 0.0
            local pltr = pltr or 0.0
            local pftr = pftr or 0.0
            local pwtr = pwtr or 0.0
            local plvtr = plvtr or 0.0
            local pvtr = pvtr or 0.0
            local pptr = pptr or 0.0
            local empirical_ctr = emp_ctr or 0.0
            local empirical_ltr = emp_ltr or 0.0
            local empirical_ftr = emp_ftr or 0.0
            local empirical_wtr = emp_wtr or 0.0
            local author_healthiness = author_healthiness or 0.0

            return {pctr}, {pltr}, {pftr}, {pwtr}, {plvtr},  {pvtr}, {pptr}, 
                  {empirical_ctr}, {empirical_ltr}, {empirical_ftr}, {empirical_wtr}, {author_healthiness}
          end
        """) \
      .uni_predict_fused(
        embedding_fetchers = [
          dict(
            fetcher_type="ColossusdbEmbeddingServerFetcher", 
            colossusdb_embd_service_name=embed_service_name,
            colossusdb_embd_table_name=embed_table_name,
            timeout_ms=50,
            max_signs_per_request=500,
            slots_inputs=["item_slots", "item_pxtr_slots"] + extra_slots,
            parameters_inputs=["item_signs", "item_pxtr_signs"] + extra_signs,
            common_slots_inputs=["common_slots", "common_kuiba_slots"],
            common_parameters_inputs=["common_signs", "common_kuiba_signs"],
            slots_config=[dict(dtype='scale_int8' if use_scale_int8 else 'mio_int16', **sc) for sc in model_config.slots_config],
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
          max_enqueued_batches=24,
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
      )
    return self \
      .arrange_by_sub_flow(sub_flow=predict_subflow, expected_partition_size=256)\
  
  # 这里都是自定义的后处理逻辑，按需使用
  def calc_interact_score(self):
    return self.get_abtest_params(
      biz_name="THANOS_RECO",
      ab_params=[
        ("slide_up_attribution_fr_weights_gamora", "lplay:0.0,eplay:0.0,time:0.0,p_staytime:0.0,like:0.0,follow:0.0,forward:0.0,comment:0.0,profile_enter:0"),
        ("slide_up_attribution_fr_weights_nebula", "lplay:0.0,eplay:0.0,time:0.0,p_staytime:0.0,like:0.0,follow:0.0,forward:0.0,comment:0.0.profile_enter:0"),
      ],
    ) \
    .perflog_attr_value(check_point="model_output",
                        item_attrs=["long_play_value", "effective_play_value", "play_time_min", "profile_stay_time_min", "like_value", "follow_value", "forward_value", "comment_value", "profile_enter_value"]) \
    .enrich_attr_by_lua(
      import_common_attr = ["slide_up_attribution_fr_weights_gamora", "slide_up_attribution_fr_weights_nebula", "tab_id"],
      function_for_common = "split_weights",
      export_common_attr = ["w_lplay", "w_eplay", "w_time", "w_p_staytime", "w_like", "w_follow", "w_forward", "w_comment", "w_profile_enter"],
      lua_script = """
        function split_weights()
          local slide_author_longterm_value_fr_weights = slide_up_attribution_fr_weights_gamora
          if tab_id == 30000 then
            slide_author_longterm_value_fr_weights = slide_up_attribution_fr_weights_nebula
          end
          local weights = {}
          for word in string.gmatch(slide_author_longterm_value_fr_weights, '([^,]+)') do
            key, value = string.match(word, '([^:]+):([^:]+)')
            table.insert(weights, tonumber(value))
          end
          local w_lplay, w_eplay, w_time, w_p_staytime, w_like, w_follow, w_forward, w_comment, w_profile_enter = weights[1], weights[2], weights[3], weights[4], weights[5], weights[6]
          
          return w_lplay, w_eplay, w_time, w_p_staytime, w_like, w_follow, w_forward, w_comment, w_profile_enter
        end
      """
    ) \
    .perflog_attr_value(check_point="weights",
                        common_attrs=["w_lplay", "w_eplay", "w_time", "w_p_staytime", "w_like", "w_follow", "w_forward", "w_comment", "w_profile_enter"]) \
    .calc_weighted_sum(
      channels=[
        {"name": "long_play_value", "weight":"{{w_lplay}}"},
        {"name": "effective_play_value", "weight": "{{w_eplay}}"},
        {"name": "play_time_min", "weight": "{{w_time}}"},
        {"name": "profile_stay_time_min", "weight": "{{w_p_staytime}}"},
        {"name": "like_value", "weight": "{{w_like}}"},
        {"name": "follow_value", "weight": "{{w_follow}}"},
        {"name": "forward_value", "weight": "{{w_forward}}"},
        {"name": "comment_value", "weight": "{{w_comment}}"},
        {"name": "profile_enter_value", "weight": "{{w_profile_enter}}"},
      ],
      output_item_attr="longterm_up_attribution_score",
    ) \
    .perflog_attr_value(check_point="final_output",
                        item_attrs=["longterm_up_attribution_score"]) \
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
# 前面访问分布式索引返回的attr没再用会报错，这里跳过报错
service.IGNORE_UNUSED_ATTR = [
  "photo_id",
  "upload_time",
  "duration_ms",
  "click_count",
  "like_count",
  "long_play_count",
  "short_play_count",
  "tag",
  "explore_stat",
  "nebula_stats",
  "hetu_tag_level_info",
  "hetu_tag_level_info_v2",
  "online_lda_topic",
  "author",
  "exp_stat_on_duration",
]

service.AUTO_INJECT_ITEM_ATTR = False
service.AUTO_INJECT_SAMPLE_LIST_USER_ATTR = False
# Infer服务最终返回的值
service.return_item_attrs(["longterm_up_attribution_score"])


predict_graph_file_path = "../train/predict/config"
slide_follow_reward_fm_config = load_mio_tf_model(predict_graph_file_path)

slide_follow_reward_fm = PredictServerFlow(name = "fr_up_longterm_attribution_model_v1") \
  .prepare() \
  .predict_with_mio_model(
    model_config=slide_follow_reward_fm_config,
    receive_dnn_model_as_macro_block=True,
    queue_prefix=btq_queue_prefix,
    embedding_protocol=0,
    batch_sizes=[512, 1024, 2048],
    implicit_batch=True,
  ) \
  .calc_interact_score()

service.add_leaf_flows(leaf_flows = [slide_follow_reward_fm],
                       request_type = "fr_up_longterm_attribution_model_v1")

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

