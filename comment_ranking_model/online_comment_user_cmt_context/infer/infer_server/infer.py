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

PXTRS = ["click_comment_score", "comment_stay_time_score", "forward_inside", "interact_score", "follow",
        "comment_unfold", "comment_like", "comment_effective_read", "comment_copyward", "comment_consume_depth",
        "comment_slide_down", "uplift_comment_consume_depth_score", "uplift_comment_stay_duration_score",
        "sub_comment", "emoji_comment", "gif_comment", "at_comment", "image_comment", "text_comment",
        "playtime_after_click_comment_score", "eft_click_cmt", "eft_write_cmt",
        'long_view_counter_factual_score_cmt','long_view_counter_factual_score_no_cmt','effective_read_comment_fresh_score']
# 最终krp部署时会覆盖这个kess_name
infer_kess_name = "grpc_interactUserClusterDebiasInfer"
# 请求带过来的attrs
pxtr_list = ["pctr", "pltr", "pcltr", "pftr", "pwtr", "plvtr", "pvtr", "pptr", "pcmtr", "phtr",
             "pepstr", "pcmef", "pwtd",
             "empirical_ctr", "empirical_ltr", "empirical_ftr", "empirical_wtr", "empirical_ptr",
             "empirical_htr", "empirical_cmtr",
             "cascade_pctr", "cascade_plvtr", "cascade_psvr", "cascade_pltr", 
             "cascade_pwtr", "cascade_pftr", "cascade_phtr", "cascade_pepstr",
             "cascade_pcestr"]
pxtrs_list = [xtr+'s' for xtr in pxtr_list]

item_attrs_from_req = ["reason", "photo_info_str"] + pxtr_list
common_attrs_from_req = ["tab_id", "user_info_str"]
model_btq_prefix = "slide_multi_task_comment_user_cluster_debias"
embed_server_shards = 1
# krp 上 embedding_server的kess_name
embed_kess_name = "grpc_interactCommentUserClusterDebiasEmb"


user_embedding_server = False
# 统一 embedding 存储配置
colossusdb_embd_service_name = "interact_comment_user_cluster_debias"
colossusdb_embd_table_name = "interact_user_cluster_debias_emb"

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
          # dict(name="click_photo_ids", path="user_profile_v1.click_list.photo_id"),
          # dict(name="click_hetu_one", path="user_profile_v1.click_list.hetu_tag_level_info.hetu_level_one",
          #       repeat_limit={"user_profile_v1.click_list.hetu_tag_level_info.hetu_level_one": 1},
          #       repeat_align=True),
          # dict(name="click_hetu_two", path="user_profile_v1.click_list.hetu_tag_level_info.hetu_level_two",
          #       repeat_limit={"user_profile_v1.click_list.hetu_tag_level_info.hetu_level_two": 1},
          #       repeat_align=True),

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
          
          dict(name="profile_enter_photo_ids", path="user_profile_v1.profile_enter_list.photo_id"),
          dict(name="profile_enter_hetu_one", path="user_profile_v1.profile_enter_list.hetu_tag_level_info.hetu_level_one",
                repeat_limit={"user_profile_v1.profile_enter_list.hetu_tag_level_info.hetu_level_one": 1},
                repeat_align=True),
          dict(name="profile_enter_hetu_two", path="user_profile_v1.profile_enter_list.hetu_tag_level_info.hetu_level_two",
                repeat_limit={"user_profile_v1.profile_enter_list.hetu_tag_level_info.hetu_level_two": 1},
                repeat_align=True),
          dict(name="human_action", path="device_stat.human_action"),
          dict(name="device_status_flags", path="device_stat.device_status_flags"),
          dict(name="FollowList", path="user_profile_v1.follow_list.author_id"),
          dict(name="BidFollowList", path="friend_info_v2.bid_follow_list.friend_id"),
      ]) \
      .enrich_attr_by_lua(
        import_common_attr=["device_status_flags"],
        function_for_common="split_device_status",
        export_common_attr=["screen_light", "net_state", 
                            "battery_level", "battery_charging", "headset_state"],
        lua_script="""
          function split_device_status()
            local device_status_flags = device_status_flags or 0
            -- 定义示例的device_status_flags值

            -- 从低位到高位提取各个位域的值
            local screen_light = device_status_flags & 0xFF  -- Bits 0-7，屏幕亮度
            local net_state = (device_status_flags >> 8) & 0xF  -- Bits 8-11，网络状态
            local battery_level = (device_status_flags >> 12) & 0xFF  -- Bits 12-19，电池电量
            local battery_charging = (device_status_flags >> 20) & 0xF  -- Bits 20-23，电池充电状态
            local headset_state = (device_status_flags >> 24) & 0xF  -- Bits 24-27，耳机状态
            return screen_light, net_state, battery_level, battery_charging, headset_state
          end
        """ 
      ) \
      .enrich_with_protobuf(
        from_extra_var="photo_info",
        is_common_attr=False,
        attrs=[
          dict(name="author_id", path="author.id"),
        ]
      ) \
      .copy_user_meta_info(
        save_user_id_to_attr="uid",
      ) \
      .enrich_attr_by_lua(
        import_item_attr = ["author_id"],
        import_common_attr = ["FollowList", "BidFollowList"],
        function_for_item = "reset_follow",
        export_item_attr = ["follow_status"],
        lua_script = """
          function value_include(tab, value)
            if tab ~= nil then
              for k, v in pairs(tab) do
                if v == value then
                  return 1
                end
              end
            end
            return 0
          end

          function reset_follow()
            local follow_status = 0  -- 标记单关 已关 双关
            local author_id = author_id or 0
            local FollowList = FollowList or {}
            local BidFollowList = BidFollowList or {}
            if value_include(FollowList, author_id) == 1 then
              follow_status = 1
            end
            if value_include(BidFollowList, author_id) == 1 then
              follow_status = 2
            end
            return follow_status
          end
        """
      ) \
      .if_("(tab_id or 0) == 10000") \
        .enrich_attr_by_lua(
            function_for_common = "calculate",
            import_common_attr = ["uid"],
            export_common_attr = ["cmt_cluste_uid_str"],
            lua_script = """
                function calculate()
                    cmt_cluste_uid_str = "cmt_cluster_cmt-" .. tostring(uid) .. "-KUAISHOU"
                    return cmt_cluste_uid_str
                end
            """
        ) \
      .else_if_("(tab_id or 0) == 30000")  \
        .enrich_attr_by_lua(
            function_for_common = "calculate",
            import_common_attr = ["uid"],
            export_common_attr = ["cmt_cluste_uid_str"],
            lua_script = """
                function calculate()
                    cmt_cluste_uid_str = "cmt_cluster_cmt-" .. tostring(uid) .. "-NEBULA"
                    return cmt_cluste_uid_str
                end
            """
        ) \
      .end_() \
      .get_common_attr_from_redis(
        cluster_name="slideLeafRecoHighFansBoost",
        is_async = True,
        redis_params=[
            {
                "redis_key": "{{cmt_cluste_uid_str}}",
                "redis_value_type": "string",
                "output_attr_name": "user_comment_cluster_level_pre",
                "output_attr_type": 'string'
            }
        ]
      ) \
      .enrich_attr_by_lua(
          function_for_common = "calculate",
          import_common_attr = ["user_comment_cluster_level_pre"],
          export_common_attr = ["user_comment_cluster_level_pre"],
          lua_script = """
              function calculate()
                  user_comment_cluster_level_pre = tonumber(user_comment_cluster_level_pre) * 1.0 or 0.0
                  return {user_comment_cluster_level_pre}
              end
          """
      ) \
      .copy_attr(
        attrs=[{
          "from_common": "user_comment_cluster_level_pre",
          "to_item": "user_comment_cluster_level"
        }]
      ) \
      .if_("(tab_id or 0) == 10000") \
        .enrich_attr_by_lua(
            function_for_common = "calculate",
            import_common_attr = ["uid"],
            export_common_attr = ["app_cluster_uid_str"],
            lua_script = """
                function calculate()
                    app_cluster_uid_str = "cmt_cluster_app-" .. tostring(uid) .. "-KUAISHOU"
                    return app_cluster_uid_str
                end
            """
        ) \
      .else_if_("(tab_id or 0) == 30000")  \
        .enrich_attr_by_lua(
            function_for_common = "calculate",
            import_common_attr = ["uid"],
            export_common_attr = ["app_cluster_uid_str"],
            lua_script = """
                function calculate()
                    app_cluster_uid_str = "cmt_cluster_app-" .. tostring(uid) .. "-NEBULA"
                    return app_cluster_uid_str
                end
            """
        ) \
      .end_() \
      .get_common_attr_from_redis(
        cluster_name="slideLeafRecoHighFansBoost",
        is_async = True,
        redis_params=[
            {
                "redis_key": "{{app_cluster_uid_str}}",
                "redis_value_type": "string",
                "output_attr_name": "user_app_cluster_level_pre",
                "output_attr_type": 'string'
            }
        ]
      ) \
      .enrich_attr_by_lua(
          function_for_common = "calculate",
          import_common_attr = ["user_app_cluster_level_pre"],
          export_common_attr = ["user_app_cluster_level_pre"],
          lua_script = """
              function calculate()
                  user_app_cluster_level_pre = tonumber(user_app_cluster_level_pre) * 1.0 or 0.0
                  return {user_app_cluster_level_pre}
              end
          """
      ) \
      .copy_attr(
        attrs=[{
          "from_common": "user_app_cluster_level_pre",
          "to_item": "user_app_cluster_level"
        }]
      ) \
      .copy_attr(
        attrs=[{
          "from_common": "tab_id",
          "to_item": "tab_id"
        }]
      )
  
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
    extra_inputs.append(("user_comment_cluster_level", "user_comment_cluster_level", False, 1))
    extra_inputs.append(("user_app_cluster_level", "user_app_cluster_level", False, 1))
    extra_inputs.append(("tab_id", "tab_id", False, 1))
    class PredictSubFlow(LeafFlow, KuibaApiMixin, MioApiMixin, OfflineApiMixin, GsuApiMixin, UniPredictApiMixin, KgnnApiMixin):
       pass

    embedding_dict = dict(
      slots_inputs=["item_slots", "kuiba_item_slots"] + extra_slots,
      parameters_inputs=["item_signs", "kuiba_item_signs"] + extra_signs,
      common_slots_inputs=["common_slots", "kuiba_common_slots"],
      common_parameters_inputs=["common_signs", "kuiba_common_signs"],
      slots_config=[dict(dtype='scale_int8' if use_scale_int8 else 'mio_int16', **sc) for sc in model_config.slots_config],
      max_signs_per_request=500,
      timeout_ms=50,
    )
    embedding_server_dict = {}
    if user_embedding_server:
      embedding_server_dict = dict(
        fetcher_type="BtEmbeddingServerFetcher", 
        kess_service=embedding_kess_name,
        shards=shards,
        client_side_shard=True,
      )
    else:
      embedding_server_dict = dict(
        fetcher_type="ColossusdbEmbeddingServerFetcher",
        colossusdb_embd_service_name=colossusdb_embd_service_name,
        colossusdb_embd_table_name=colossusdb_embd_table_name,
      )
    embedding_dict.update(embedding_server_dict)

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
            local pctr = pctr or 0.0
            local pltr = pltr or 0.0
            local pcltr = pcltr or 0.0
            local pftr = pftr or 0.0
            local pwtr = pwtr or 0.0
            local plvtr = plvtr or 0.0
            local pvtr = pvtr or 0.0
            local pptr = pptr or 0.0
            local pcmtr = pcmtr or 0.0
            local phtr = phtr or 0.0
            local pepstr = pepstr or 0.0
            local pcmef = pcmef or 0.0
            local pwtd = pwtd or 0.0
            local empirical_ctr = empirical_ctr or 0.0
            local empirical_ltr = empirical_ltr or 0.0
            local empirical_ftr = empirical_ftr or 0.0
            local empirical_wtr = empirical_wtr or 0.0
            local empirical_ptr = empirical_ptr or 0.0
            local empirical_htr = empirical_htr or 0.0
            local empirical_cmtr = empirical_cmtr or 0.0
            local cascade_pctr = cascade_pctr or 0.0 
            local cascade_plvtr = cascade_plvtr or 0.0
            local cascade_psvr = cascade_psvr or 0.0
            local cascade_pltr = cascade_pltr or 0.0
            local cascade_pwtr = cascade_pwtr or 0.0 
            local cascade_pftr = cascade_pftr or 0.0
            local cascade_phtr = cascade_phtr or 0.0
            local cascade_pepstr = cascade_pepstr or 0.0
            local cascade_pcestr = cascade_pcestr or 0.0

            return {pctr}, {pltr}, {pcltr}, {pftr}, {pwtr}, {plvtr}, {pvtr}, {pptr}, {pcmtr},
                  {phtr}, {pepstr}, {pcmef}, {pwtd}, 
                  {empirical_ctr}, {empirical_ltr}, {empirical_ftr}, {empirical_wtr},
                  {empirical_ptr}, {empirical_htr}, {empirical_cmtr},
                  {cascade_pctr}, {cascade_plvtr}, {cascade_psvr}, {cascade_pltr},
                  {cascade_pwtr}, {cascade_pftr}, {cascade_phtr}, {cascade_pepstr}, {cascade_pcestr}
          end
        """) \
      .uni_predict_fused(
        embedding_fetchers = [
          embedding_dict
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
      ) 
    return self \
      .arrange_by_sub_flow(sub_flow=predict_subflow, expected_partition_size=64)

  def calc_revert_score(self):
    return self.get_abtest_params(
      biz_name="THANOS_RECO",
      ab_params=[
        ("slide_fr_interact_wt_score_infer_norm_gamora", False),
        ("slide_fr_interact_wt_score_infer_norm_nebula", False),
        ("enable_playtime_causal_debias_model_gamora", False),
        ("enable_playtime_causal_debias_model_nebula", False),
        ("enable_long_view_counter_causal_model_gamora", False),
        ("enable_long_view_counter_causal_model_nebula", False),
      ],
    ) \
    .perflog_attr_value(check_point="model.output",
                          item_attrs=["click_comment_score", "comment_stay_time_score"]) \
    .if_("(tab_id or 0) == 10000 and (slide_fr_interact_wt_score_infer_norm_gamora or 0) == 1 or (tab_id or 0) == 30000 and (slide_fr_interact_wt_score_infer_norm_nebula or 0) == 1") \
      .enrich_attr_by_lua(
        import_item_attr=["comment_stay_time_score"],
        function_for_item = "revert",
        export_item_attr= ["comment_stay_time_score"],
        lua_script = """
          function revert()
            local comment_score = comment_stay_time_score or 0
            if comment_score > 0 and comment_score < 1 then
                comment_score = comment_score / (1.0 - comment_score)
            end
            return comment_score
          end
        """
      ) \
    .end_if_() \
    .enrich_attr_by_lua(
        import_item_attr=["click_comment_score", "comment_stay_time_score", "comment_unfold"],
        function_for_item = "rename",
        export_item_attr= ["click_comment_button", "post_at_comment"],
        lua_script = """
          function rename()
            return click_comment_score * comment_stay_time_score, comment_unfold
          end
        """
      ) \
    .if_("(tab_id or 0) == 10000 and (enable_playtime_causal_debias_model_gamora or 0) == 1 or (tab_id or 0) == 30000 and (enable_playtime_causal_debias_model_nebula or 0) == 1") \
      .enrich_attr_by_lua(
        import_item_attr=["click_comment_score", "long_view_counter_factual_score_cmt", "long_view_counter_factual_score_no_cmt"],
        function_for_item = "calculate",
        export_item_attr= ["playtime_after_click_comment_score"],
        lua_script = """
          function calculate()
            return long_view_counter_factual_score_no_cmt 
          end
        """
      ) \
    .end_if_() \
    .if_("(tab_id or 0) == 10000 and (enable_long_view_counter_causal_model_gamora or 0) == 1 ") \
      .calc_by_formula1(
          kconf_key = "formula.scenarioKey92.slide_gamora_cmt_long_view_counter_causal_f1",
          import_item_attr = [
          {"name": "long_view_counter_factual_score_cmt", "as": "lvf_score", "default_val": 0.0},
          {"name": "long_view_counter_factual_score_no_cmt", "as": "lvcf_score", "default_val": 0.0},
          {"name": "click_comment_score", "as": "pccmt", "default_val": 0.0},
          ],
          export_formula_value = [
          {"name": "score", "as": "playtime_after_click_comment_score"},
          ],
          abtest_biz_name = "THANOS_RECO"
      ) \
    .end_if_() \
    .if_("(tab_id or 0) == 30000 and (enable_long_view_counter_causal_model_nebula or 0) == 1") \
      .calc_by_formula1(
          kconf_key = "formula.scenarioKey81.slide_nebula_cmt_long_view_counter_causal_f1",
          import_item_attr = [
          {"name": "long_view_counter_factual_score_cmt", "as": "lvf_score", "default_val": 0.0},
          {"name": "long_view_counter_factual_score_no_cmt", "as": "lvcf_score", "default_val": 0.0},
          {"name": "click_comment_score", "as": "pccmt", "default_val": 0.0},
          ],
          export_formula_value = [
          {"name": "score", "as": "playtime_after_click_comment_score"},
          ],
          abtest_biz_name = "THANOS_RECO"
      ) \
    .end_if_() \
    .perflog_attr_value(check_point="final.output",
                          item_attrs=["click_comment_button", "follow", "forward_inside", "interact_score", "post_at_comment", "comment_like", "comment_effective_read", "comment_copyward", "comment_consume_depth",
                                      "comment_slide_down", "uplift_comment_consume_depth_score", "uplift_comment_stay_duration_score",
                                      "sub_comment", "emoji_comment", "gif_comment", "at_comment", "image_comment", "text_comment",
                                      "playtime_after_click_comment_score", "eft_click_cmt", "eft_write_cmt","effective_read_comment_fresh_score"]) \

    
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
  print(q_names)
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
service.return_item_attrs(["interact_score", "forward_inside",  
                           "click_comment_button", "follow", "post_at_comment", "comment_like", "comment_effective_read", "comment_copyward", "comment_consume_depth",
                           "comment_slide_down", "uplift_comment_consume_depth_score", "uplift_comment_stay_duration_score",
                           "sub_comment", "emoji_comment", "gif_comment", "at_comment", "image_comment", "text_comment",
                           "playtime_after_click_comment_score","click_comment_score", "eft_click_cmt", "eft_write_cmt","effective_read_comment_fresh_score"])

print(current_folder)
predict_graph_file_path = "../../train/predict/config"
current_model_config = load_mio_tf_model(predict_graph_file_path)

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
  ) \
  .calc_revert_score()

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

