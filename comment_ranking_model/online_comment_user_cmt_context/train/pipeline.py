import os
import sys
import argparse
import itertools
import operator
import yaml
import json

import lua_script

current_dir = os.path.dirname(__file__)
#sys.path.append(os.path.join(current_dir, '../../../ks/common_reco/leaf/tools/pypi/'))

from dragonfly.common_leaf_dsl import LeafFlow, OfflineRunner
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.gsu.gsu_api_mixin import GsuApiMixin
from dragonfly.ext.kgnn.kgnn_api_mixin import KgnnApiMixin
from dragonfly.ext.kgnn.node_attr_schema import NodeAttrSchema

parser = argparse.ArgumentParser()
parser.add_argument('--run', dest="run", default=False, action='store_true')
parser.add_argument('--eval', dest="eval", default=False, action='store_true')
parser.add_argument('src', nargs="*", choices=["hdfs", "kafka"], default="kafka")
args = parser.parse_args()

ineffective_view_threshold = 3000
effective_view_threshold = 7000
long_view_threshold = 18000
very_long_view_threshold = 36000
watch_time_unit = 1000
comment_stay_time_threshold = 2000
effective_profile_stay_time_threshold = 4000
comment_long_stay_time_threshold = 5000
comment_tree_node_num = 8
# optional float user_active_level = 134; // 用户活跃程度，0～1之间的值: 0为完全不活跃，1为完全活跃; buildDetailUserinfo生产, leaf消费
# optional int32 active_days = 149;

kuiba_list_converter_config_list_limit = lambda limit_n:  {
  "converter": "list",
  "type":5,
  "converter_args": {
    "reversed": False,
    "enable_filter": False,
    "limit": limit_n,
  },
 }

SEQ_SIZE = 100

def load_duration_playing_table(filename):
  ret = list()
  with open(filename) as f:
    for i, line in enumerate(f):
      key, thresh = line.strip().split(' ')
      assert int(key) == i, (key, i)
      ret.append(int(thresh))
  return ret

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

# load plugins
class DataReaderFlow(LeafFlow, MioApiMixin, KuibaApiMixin, OfflineApiMixin, GsuApiMixin, KgnnApiMixin):
  pass

labels = ["forward_inside", "click_comment_button", "follow",  "comment_watch_time", "comment_coeff", "comment_action_coeff", "comment_stay_coeff", "effective_view", "action_expand_secondary_comment_count",
          "action_like_comment", "action_comment_content_copy", "action_comment_content_forward", "comment_long_stay", "effective_read_comment", "comment_copyward", "action_comment_slide_down",
          "action_sub_comment",
          "action_emoji_comment",
          "action_gif_comment",
          "action_at_comment",
          "action_image_comment",
          "action_text_comment",
          "action_video_comment",
          "eft_click_cmt", "eft_write_cmt"]
pxtr_list = ["pctr", "pltr", "pcltr", "pftr", "pwtr", "plvtr", "pvtr", "pptr", "pcmtr", "phtr",
             "pepstr", "pcmef", "pwtd",
             "empirical_ctr", "empirical_ltr", "empirical_ftr", "empirical_wtr", "empirical_ptr",
             "empirical_htr", "empirical_cmtr",
             "cascade_pctr", "cascade_plvtr", "cascade_psvr", "cascade_pltr", 
             "cascade_pwtr", "cascade_pftr", "cascade_phtr", "cascade_pepstr",
             "cascade_pcestr"]

# group_id = slide_related_cascade_rank
# slide_related_reco_rerank_log
data_sources = ['kafka']
hdfs_path = "viewfs:///home/reco_5/mpi/mtl_interact/interact_data/20240617/09/*/*"
new_hdfs_path = "viewfs:///home/reco_5/mpi/mtl_interact/interact_data_new/10/*/*"


fetch_message_configs = dict()
if 'kafka' in data_sources:
  fetch_message_configs.update(dict(
    # group_id='slide_multi_task_interact_fm',
    group_id='slide_multi_task_interact_user_cmt_context',
    kafka_topic="slide_new_hot_train",
  ))
if 'hdfs' in data_sources:
  fetch_message_configs.update(dict(
    hdfs_path=hdfs_path,
    hdfs_read_thread_num=12,
    hdfs_timeout_ms=60 * 60 * 1000,
  ))

read_joint_reco_log = DataReaderFlow(name = "read_joint_reco_log") \
  .fetch_message(
    output_attr="ks_reco_log_str",
    begin_time_ms='2022-08-10 18',
    **fetch_message_configs) \
  .parse_protobuf_from_string(
    input_attr="ks_reco_log_str",
    output_attr="ks_reco_log",
    class_name="ks::reco::RecoLog") \
  .enrich_with_protobuf(
    from_extra_var="ks_reco_log",
    is_common_attr=True,
    attrs=[
      dict(path="user", name="user_info"),
      dict(path="tab", name="tab_id"),
    ]) \
  .retrieve_from_ks_reco_log(
    from_extra_var="ks_reco_log",
    save_reco_photo_to="reco_photo_info") \
  .enrich_with_protobuf(
    from_extra_var="reco_photo_info",
    is_common_attr=False,
    attrs=[
      dict(path="photo", name="photo_info"),
      "reason", "context_info"] + pxtr_list
  ) \
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

      dict(name="FollowList", path="user_profile_v1.follow_list.author_id"),
      dict(name="BidFollowList", path="friend_info_v2.bid_follow_list.friend_id"),
      dict(name="human_action", path="device_stat.human_action"),
      dict(name="device_status_flags", path="device_stat.device_status_flags"),
  ]) \
  .get_kconf_params(
    kconf_configs = [
      {"kconf_key": "reco.GamoraInteract.WtdPredictionLTRConf","export_common_attr": "comment_coeff","json_path" : "comment_coeff3"},
      {"kconf_key": "reco.GamoraInteract.WtdPredictionLTRConf","export_common_attr": "comment_action_coeff","json_path" : "comment_action_coeff3"},
      {"kconf_key": "reco.GamoraInteract.WtdPredictionLTRConf","export_common_attr": "comment_stay_coeff","json_path" : "comment_stay_coeff3"},
      {"kconf_key": "reco.GamoraInteract.WtdPredictionLTRConf","export_common_attr": "comment_watch_num_label_div_list","json_path" : "comment_watch_num_label_div_list"},
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
      "duration_ms",
      "audit_b_second_tag",
      dict(name="author_id", path="author.id"),
    ]
  ) \
  .enrich_with_protobuf(
    from_extra_var="context_info",
    is_common_attr=False, 
    attrs=["real_show", "playing_time", 
           "like", "follow", "forward", "comment", "collect", 
          "unfollow", "unlike", "discard", "dislike", "cancel_collect", "feedback_negative",
          "down_load", "video_screen_shot", "post_at_comment", "click_comment_button", 
          "comment_stay_time", "comment_reply", 
           dict(name="label_names", 
                path="reco_labels.name_value"),
           dict(name="label_values", 
                path="reco_labels.int_value"),
           dict(name="label_values_bool",
                path="reco_labels.bool_value"),
           dict(name="real_show_timestamp_ms",
                path="reco_labels",
                sample_attr_name_value=442),
           dict(name="action_click_comment_timestamp_ms",
                path="reco_labels",
                sample_attr_name_value=1139),  # 看评
           dict(name="action_write_comment_timestamp_ms",
                path="reco_labels",
                sample_attr_name_value=630),   # 写评
          ]
  ) \
  .enrich_attr_by_lua(
    import_item_attr = ["reason"],
    function_for_item = "parse_int",
    export_item_attr = ["exptag_int"],
    lua_script = """
      function parse_int()
        local number = tonumber(string.match(reason, "%d+"))
        return number
      end
    """
  ) \
  .filter_by_rule(rule={"attr_name":"exptag_int", "remove_if":"in", "compare_to": [278, 1821, 1822, 1823, 1824, 1841, 1842, 503, 2800]}, name="remove_social_samples") \
  .filter_by_attr(attr_name="real_show", remove_if="==", compare_to=0, name="remove_no_real_show") \
  .enrich_attr_by_lua(
    import_item_attr = ["author_id", "follow"],
    import_common_attr = ["FollowList", "BidFollowList"],
    function_for_item = "reset_follow",
    export_item_attr = ["follow", "reset_count", "follow_weight", "follow_status"],
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
        local follow = follow or 0
        local reset_count = 0
        local follow_weight = 1
        if author_id == 0 then
          return follow, reset_count, follow_weight, follow_status
        end
        local FollowList = FollowList or {}
        local BidFollowList = BidFollowList or {}
        if value_include(FollowList, author_id) == 1 then
          follow = 0
          reset_count = reset_count + 1
          follow_weight = 0
          follow_status = 1
        end
        if value_include(BidFollowList, author_id) == 1 then
          follow_status = 2
        end
        return follow, reset_count, follow_weight, follow_status
      end
    """
  ) \
  .enrich_attr_by_lua(
    import_item_attr=["label_names", "label_values"],
    function_for_item="calculate",
    export_item_attr=["forward_inside_status", "forward_in_count"],
    lua_script="""
      function calculate()
        local names = label_names or {}
        if #names == 0 then
          return 0, 0
        end
        forward_status = 0
        forward_count = 1
        for i = 1, #label_names do
          if label_names[i] == 404 then
            forward_status = label_values[i]
          end
          if label_names[i] == 1036 then
            forward_count = label_values[i]
          end
        end
        return forward_status, forward_count
      end
    """) \
  .enrich_attr_by_lua(
    import_item_attr=["forward_inside_status"],
    function_for_item="calculate",
    export_item_attr=["forward_inside"],
    lua_script=f"""
      function calculate()
        local forward_inside_status = forward_inside_status or 0
        local success_in = (forward_inside_status >> 2) & 1
        return success_in
      end
    """) \
  .enrich_attr_by_lua(
    import_item_attr=["collect", "cancel_collect", "like", "follow", 
                      "comment", "click_comment_button", "audit_b_second_tag"],
    function_for_item="reset",
    export_item_attr=["collect", "like", "follow", 
                      "comment", "click_comment_button", "audit_count"],
    lua_script="""
      function reset()
        local audit_count = 0
        if collect == 1 and cancel_collect == 1 then
          collect = 0
        end
        if audit_b_second_tag == 2037808 or audit_b_second_tag == 2111600 then
           like = 0
           follow = 0
           audit_count = audit_count + 1
        end
        if audit_b_second_tag == 2037809 or audit_b_second_tag == 2111602 then
           follow = 0
           audit_count = audit_count + 1
        end
        if audit_b_second_tag == 2037810 or audit_b_second_tag == 2111601 then
           like = 0
           audit_count = audit_count + 1
        end
        if audit_b_second_tag == 2037811 or audit_b_second_tag == 2111603 then
           comment = 0
           click_comment_button = 0
           audit_count = audit_count + 1
        end
        if audit_b_second_tag == 2147251 then
          like = 0
          follow = 0
          comment = 0
          click_comment_button = 0
          collect = 0
        end
        return collect, like, follow, comment, click_comment_button, audit_count
      end
    """) \
  .enrich_attr_by_lua(
    import_item_attr=["label_names", "label_values", "label_values_bool"],
    function_for_item="calculate",
    export_item_attr=["action_like_comment", "action_expand_secondary_comment_count", "action_comment_click_head", "action_comment_click_nickname", "action_comment_content_copy", "action_comment_content_forward", "action_comment_search_highlight_click", "action_comment_search_trending_click", "comment_copyward",
                      "action_expand_secondary_comment_count_abs", "action_comment_click_head_abs", "action_comment_click_nickname_abs", "action_comment_content_copy_abs", "action_comment_content_forward_abs", "watch_comment_num", "action_comment_slide_down"],
    lua_script="""
      function calculate()
        local names = label_names or {}
        if #names == 0 then
          return 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0, 0
        end
        action_like_comment = 0
        action_expand_secondary_comment_count = 0
        action_comment_click_head = 0
        action_comment_click_nickname = 0
        action_comment_content_copy = 0
        action_comment_content_forward = 0
        action_comment_slide_down = 0
        action_comment_search_highlight_click = 0
        action_comment_search_trending_click = 0
        action_expand_secondary_comment_count_abs = 0
        action_comment_click_head_abs = 0
        action_comment_click_nickname_abs = 0
        action_comment_content_copy_abs = 0
        action_comment_content_forward_abs = 0
        comment_copyward = 0
        watch_comment_num = 0
        for i = 1, #label_names do
          if label_names[i] == 335 and label_values_bool[i] > 0 then
            action_like_comment = 1
          end
          if label_names[i] == 336 and label_values[i] > 0 then
            action_expand_secondary_comment_count = 1
            action_expand_secondary_comment_count_abs = label_values[i]
          end
          if label_names[i] == 337 and label_values[i] > 0 then
            watch_comment_num = label_values[i]
          end
          if label_names[i] == 481 and label_values[i] > 0 then
            action_comment_click_head = 1
            action_comment_click_head_abs = label_values[i]
          end
          if label_names[i] == 482 and label_values[i] > 0 then
            action_comment_click_nickname = 1
            action_comment_click_nickname_abs = label_values[i]
          end
          if label_names[i] == 483 and label_values[i] > 0 then
            action_comment_content_copy = 1
            action_comment_content_copy_abs = label_values[i]
          end
          if label_names[i] == 484 and label_values[i] > 0 then
            action_comment_content_forward = 1
            action_comment_content_forward_abs = label_values[i]
          end
          if label_names[i] == 398 and label_values[i] ~= '' then
            action_comment_search_highlight_click = 1
          end
          if label_names[i] == 400 and label_values[i] ~= '' then
            action_comment_search_trending_click = 1
          end
          if label_names[i] == 1113 and label_values[i] > 0 then
            action_comment_slide_down = 1
          end
        end
        comment_copyward = math.max(action_comment_content_forward, action_comment_content_copy)
        return action_like_comment, action_expand_secondary_comment_count, action_comment_click_head, action_comment_click_nickname, action_comment_content_copy, action_comment_content_forward, action_comment_search_highlight_click, action_comment_search_trending_click, comment_copyward,
               action_expand_secondary_comment_count_abs, action_comment_click_head_abs, action_comment_click_nickname_abs, action_comment_content_copy_abs, action_comment_content_forward_abs, watch_comment_num, action_comment_slide_down
      end
    """) \
  .enrich_attr_by_lua(
    import_item_attr=["comment_stay_time", "action_like_comment", "action_expand_secondary_comment_count_abs", "action_comment_click_head_abs", "action_comment_click_nickname_abs", "action_comment_content_copy_abs", "action_comment_content_forward_abs", "comment_reply", "action_comment_search_highlight_click", "action_comment_search_trending_click"],
    function_for_item="calculate",
    export_item_attr=["effective_read_comment"],
    lua_script=f"""
      function calculate()
        effective_read_comment = 0
        effective_read_comment_score = 0.0
        action_comment_effective_stay_time = 0

        action_like_comment = action_like_comment or 0
        action_expand_secondary_comment_count_abs = action_expand_secondary_comment_count_abs or 0
        action_comment_click_head_abs = action_comment_click_head_abs or 0
        action_comment_click_nickname_abs = action_comment_click_nickname_abs or 0
        action_comment_content_copy_abs = action_comment_content_copy_abs or 0
        action_comment_content_forward_abs = action_comment_content_forward_abs or 0
        action_comment_reply = comment_reply or 0
        action_comment_search_trending_click = action_comment_search_trending_click or 0
        action_comment_search_highlight_click = action_comment_search_highlight_click or 0
        action_comment_search = action_comment_search_trending_click + action_comment_search_highlight_click
        action_comment_enter_profile = action_comment_click_head_abs + action_comment_click_nickname_abs       
        if comment_stay_time > 10000 then
          action_comment_effective_stay_time = 1
        end

        comment_like_log = math.log(action_like_comment + 1)
        comment_unfold_log = math.log(action_expand_secondary_comment_count_abs + 1)
        comment_enter_profile_log = math.log(action_comment_enter_profile + 1)
        action_comment_search_log = math.log(action_comment_search + 1)
        comment_copy_log = math.log(action_comment_content_copy_abs + 1)
        comment_forward_log = math.log(action_comment_content_forward_abs + 1)
        comment_effective_stay_time_log = math.log(action_comment_effective_stay_time + 1)
        comment_reply_log = math.log(action_comment_reply + 1)

        effective_read_comment_score = 100 * (comment_like_log * 0.0838 + comment_unfold_log * 0.0457 + comment_enter_profile_log * 0.1269
                                       + comment_copy_log * 0.1721 + comment_forward_log * 0.24 + comment_effective_stay_time_log * 0.0119
                                       + comment_reply_log * 0.1298 + action_comment_search_log * 0.1897)

        if effective_read_comment_score >= 0.2 then
          effective_read_comment = 1
        end
        return effective_read_comment
      end
    """
  ) \
.pack_item_attr(
  item_source={
    "reco_results": True
  },
  mappings=[
      {
        "from_item_attr": "effective_read_comment",
        "to_common_attr": "effective_read_comment_fresh",
        "aggregator": "sum"
      }
  ]
)\
.enrich_attr_by_lua(
    import_common_attr = ["effective_read_comment_fresh"],
    function_for_item = "calculate",
    export_item_attr = ["effective_read_comment_fresh_weight", "fresh_true_sample_label"],
    lua_script = """
      function calculate(seq, item_key, reason, score)
        local effective_read_comment_fresh_weight = 0
        local fresh_true_sample_label = 0
        
        if seq == 0 then 
            effective_read_comment_fresh_weight = effective_read_comment_fresh
            fresh_true_sample_label = 1
        end
        
        return effective_read_comment_fresh_weight, fresh_true_sample_label
        
      end
    """
    )\
  .enrich_attr_by_lua(
    import_item_attr=["label_names", "label_values"],
    function_for_item="calculate",
    export_item_attr=["action_sub_comment",
                      "action_emoji_comment",
                      "action_gif_comment",
                      "action_at_comment",
                      "action_image_comment",
                      "action_text_comment",
                      "action_video_comment"
                      ],
    lua_script="""
      function calculate()
          local sub_comment = 0
          local emoji_comment = 0
          local gif_comment = 0
          local at_comment = 0
          local image_comment = 0
          local text_comment = 0
          local video_comment = 0
          local names = label_names or {}
          if #(label_names or {}) > 0 and #(label_names or {}) == #(label_values or {}) then
              for i = 1, #label_names do
                  if label_names[i] == 1120 then
                      genre = label_values[i]
                      sub_comment = genre & 1
                      emoji_comment = (genre >> 1) & 1
                      gif_comment = ((genre >> 2) & 1) | ((genre >> 4) & 1)
                      at_comment = (genre >> 3) & 1
                      image_comment = (genre >> 5) & 1
                      text_comment = (genre >> 6) & 1
                      video_comment = (genre >> 7) & 1
                      
                      break
                  end
              end
          end
          return sub_comment, emoji_comment, gif_comment, at_comment, image_comment, text_comment, video_comment
      end
    """
  ) \
  .perflog_attr_value(
    check_point="comment.genre",
    item_attrs=["action_sub_comment",
                "action_emoji_comment",
                "action_gif_comment",
                "action_at_comment",
                "action_image_comment",
                "action_text_comment",
                "action_video_comment"],
  ) \
  .enrich_attr_by_lua(
    import_item_attr=["playing_time", "duration_ms"],
    function_for_item="calculate",
    export_item_attr=["effective_view"],
    lua_script=f"""
      function calculate()
        effective_view = 0
        if duration_ms < 7000 then
          if playing_time > 7000 then
            effective_view = 1
          end
        elseif duration_ms >= 7000 and duration_ms < 14000 then
          if playing_time > 0.636788*duration_ms + 2417.629 then
            effective_view = 1
          end
        elseif duration_ms >= 14000 and duration_ms < 93000 then
          if playing_time > 0.104396*duration_ms + 8957.299 then
            effective_view = 1
          end
        elseif duration_ms >= 93000 and duration_ms < 155000 then
          if playing_time > -0.018168*duration_ms + 20215.178 then
            effective_view = 1
          end
        elseif duration_ms >= 155000 and duration_ms < 220000 then
          if playing_time > -0.072866*duration_ms + 28352.154 then
            effective_view = 1
          end
        else
          if playing_time > -0.038242*duration_ms + 21259.100 and playing_time > 7000 then
            effective_view = 1
          end
        end
        return effective_view
      end
    """) \
  .enrich_attr_by_lua(
    import_item_attr=["playing_time", "duration_ms"],
    function_for_item="calculate",
    export_item_attr=["long_view"],
    lua_script=f"""
      function calculate()
        long_view = 0
        if duration_ms < 14000 then
          if playing_time > 0.881369*duration_ms + 5184.729 then
            long_view = 1
          end
        elseif duration_ms >= 14000 and duration_ms < 53000 then
          if playing_time > 0.918007*duration_ms + 3847.164 then
            long_view = 1
          end
        elseif duration_ms >= 53000 and duration_ms < 95000 then
          if playing_time > 0.732063*duration_ms + 13416.789 then
            long_view = 1
          end
        elseif duration_ms >= 95000 and duration_ms < 170000 then
          if playing_time > 0.226594*duration_ms + 62817.905 then
            long_view = 1
          end
        else
          if playing_time > -0.180531*duration_ms + 131062.293 and playing_time > 36000 then
            long_view = 1
          end
        end
        return long_view
      end
    """) \
  .enrich_attr_by_lua(
    import_item_attr=["playing_time", "duration_ms", "effective_view", "long_view"],
    function_for_item="calculate",
    export_item_attr=["time_weight"],
    lua_script=f"""
      function calculate()
        time_weight = 0.0
        if duration_ms == 0 then
          time_weight = 0.0
        else 
          time_weight = math.min(math.max(playing_time / duration_ms, 0.0), 1.0)
        end
        time_weight = time_weight + effective_view
        time_weight = time_weight + long_view
        return time_weight
      end
    """) \
  .enrich_attr_by_lua(
    import_item_attr=["playing_time", "comment_stay_time"],
    function_for_item="calculate",
    export_item_attr=["comment_watch_time", "comment_effective_stay", "comment_long_stay"],
    lua_script=f"""
      function calculate()
        local comment_watch_time = math.min(64, math.ceil(comment_stay_time / {watch_time_unit}))
        return comment_watch_time, comment_stay_time >= {comment_stay_time_threshold}, comment_stay_time >= {comment_long_stay_time_threshold}
      end
    """) \
  .enrich_attr_by_lua(
    import_item_attr=["playing_time", "comment_stay_time","real_show_timestamp_ms","action_click_comment_timestamp_ms","long_view"],
    function_for_item="calculate",
    export_item_attr=["playing_time_after_click_comment","playing_time_after_out_of_click_comment","playing_time_after_out_of_click_comment_label","action_click_comment_timestamp_ms_label"],
    lua_script=f"""
      function calculate()
        local playing_time_after_click_comment = 0
        local playing_time_after_out_of_click_comment = 0
        local playing_time_after_out_of_click_comment_label = 0
        local action_click_comment_timestamp_ms_label = 0
        local action_click_comment_timestamp_ms = action_click_comment_timestamp_ms or 0
        local real_show_timestamp_ms = real_show_timestamp_ms or 0
        if action_click_comment_timestamp_ms > 0 then
            action_click_comment_timestamp_ms_label = 1
            playing_time_after_click_comment = math.max(real_show_timestamp_ms + playing_time - action_click_comment_timestamp_ms, 0)
            playing_time_after_out_of_click_comment = math.max(real_show_timestamp_ms + playing_time - action_click_comment_timestamp_ms - comment_stay_time, 0)
            if playing_time_after_out_of_click_comment > 2000 then
                playing_time_after_out_of_click_comment_label = 1
            end
        end
        return  playing_time_after_click_comment,playing_time_after_out_of_click_comment,playing_time_after_out_of_click_comment_label,action_click_comment_timestamp_ms_label
      end
    """) \
  .get_kconf_params(
      kconf_configs = [
          {
              "kconf_key": "cc.knowledgeGraph.playClickCommentTrainConf",
              "export_common_attr": "effective_play_video_ms",
              "json_path" : "effective_play_video_ms",
              "default_value": 7000
          },
          {
              "kconf_key": "cc.knowledgeGraph.playClickCommentTrainConf",
              "export_common_attr": "effective_play_comment_ms",
              "json_path" : "effective_play_comment_ms",
              "default_value": 2000
          },
  ]) \
  .enrich_attr_by_lua(
      import_common_attr=["effective_play_video_ms", "effective_play_comment_ms"],
      import_item_attr=["real_show_timestamp_ms", "action_click_comment_timestamp_ms", "action_write_comment_timestamp_ms", "duration_ms"],
      export_item_attr=["wt_before_click_comment", "wt_before_write_comment", "eft_click_cmt", "eft_write_cmt"],
      function_for_item="cal_progressive_watch_click_write_comment",
      lua_script="""
          function cal_progressive_watch_click_write_comment()
              local real_show_ts = real_show_timestamp_ms or 0
              local click_comment_ts = action_click_comment_timestamp_ms or 0
              local write_comment_ts = action_write_comment_timestamp_ms or 0
              
              local wt_before_click_comment = math.max(click_comment_ts - real_show_ts, 0)
              local wt_before_write_comment = math.max(write_comment_ts - click_comment_ts, 0)

              local eft_click_cmt = 0
              local eft_write_cmt = 0 
              if (click_comment_ts > 0) and (wt_before_click_comment > duration_ms or wt_before_click_comment > effective_play_video_ms) then
                eft_click_cmt = 1
              end
              if write_comment_ts > 0 and wt_before_write_comment > effective_play_comment_ms then
                eft_write_cmt = 1
              end

              return wt_before_click_comment / 1000, wt_before_write_comment / 1000, eft_click_cmt, eft_write_cmt
          end
      """
  ) \
  .perflog_attr_value(
      check_point="comment.progressive",
      item_attrs=["wt_before_click_comment", "eft_click_cmt"],
      select_item={
          "join": "and",
          "filters": [
            {
              "attr_name": "real_show_timestamp_ms",
              "select_if": ">",
              "compare_to": 0,
              "select_if_attr_missing": False
            }, 
            {
              "attr_name": "action_click_comment_timestamp_ms",
              "select_if": ">",
              "compare_to": 0,
              "select_if_attr_missing": False
            }
          ],
      }
  ) \
  .perflog_attr_value(
      check_point="comment.progressive",
      item_attrs=["wt_before_write_comment", "eft_write_cmt"],
      select_item={
          "join": "and",
          "filters": [
            {
              "attr_name": "action_click_comment_timestamp_ms",
              "select_if": ">",
              "compare_to": 0,
              "select_if_attr_missing": False
            },
            {
              "attr_name": "action_write_comment_timestamp_ms",
              "select_if": ">",
              "compare_to": 0,
              "select_if_attr_missing": False
            },
          ],
      }
  ) \
  .enrich_attr_by_lua(
    import_item_attr=["feedback_negative", "dislike", "discard"],
    function_for_item="calculate",
    export_item_attr=["feedback_negative"],
    lua_script="""
      function calculate()
        feedback_negative = feedback_negative or 0
        if feedback_negative > 0 or dislike > 0 or discard > 0 then
          feedback_negative = 1
        end
        return feedback_negative
      end
    """) \
  .enrich_attr_by_lua(
    import_item_attr=["feedback_negative", "like", "follow", "forward", "comment", "collect", 
                      "forward_inside", "click_comment_button"],
    function_for_item="calculate",
    export_item_attr=["like", "follow", "forward", "comment", "collect", 
                      "forward_inside", "click_comment_button"],
    lua_script="""
      function calculate()
        if feedback_negative > 0 then
          return 0, 0, 0, 0, 0, 0, 0
        else
          return like, follow, forward, comment, collect, forward_inside, click_comment_button
        end
      end
    """) \
  .enrich_attr_by_lua(
    import_item_attr=["action_like_comment", 
                      "action_expand_secondary_comment_count", "action_comment_click_head", 
                      "action_comment_click_nickname", "action_comment_content_copy", 
                      "action_comment_content_forward", "click_comment_button",
                      "action_comment_slide_down",
                      "feedback_negative"],
    function_for_item="calculate",
    export_item_attr=["comment_action_weight"],
    lua_script="""
      function calculate()
        local weight = 1
        if action_like_comment > 0 then
          weight = weight + 5
        end
        if action_expand_secondary_comment_count > 0 then
          weight = weight + 5
        end
        if action_comment_click_head > 0 then
          weight = weight + 1
        end
        if action_comment_click_nickname > 0 then
          weight = weight + 1
        end
        if action_comment_content_copy > 0 then
          weight = weight + 1
        end
        if action_comment_content_forward > 0 then
          weight = weight + 1
        end
        if action_comment_slide_down > 0 then
          weight = weight + 1
        end
        if feedback_negative > 0 and click_comment_button > 0 then
          weight = weight + 100
        end
        return weight
      end
    """) \
  .enrich_attr_by_lua(
    import_item_attr=["like", "follow", "forward", "comment", "collect", 
          "unfollow", "unlike", "cancel_collect", "feedback_negative",
          "forward_inside", "forward_in_count", "time_weight", 'down_load', 
          'video_screen_shot', "post_at_comment", "effective_view", "long_view"],
    function_for_item="calculate",
    export_item_attr=["interact_weight", "interact_label", 
                      "forward_inside_weight"],
    lua_script="""
      function calculate()
        forward_in_count = forward_in_count or 1.0
        interact_label = 0
        if (like > 0 or follow > 0 or forward > 0  or comment > 0 or collect > 0 or forward_inside > 0 or down_load > 0 or video_screen_shot > 0) 
        and (effective_view > 0 or long_view > 0) then
          interact_label = 1
        end
        interactive_weight = 1.0
        forward_inside_weight = 1.0
        if feedback_negative > 0 then 
          interactive_weight = 100.0
          forward_inside_weight = 50.0
        end
        if forward_inside > 0 then
          forward_inside_weight = forward_inside_weight * time_weight
        end
        if interact_label > 0 then
          if effective_view > 0 or long_view > 0 then
            interactive_weight = interactive_weight + 1.0
          end
          if like > 0 then
            interactive_weight = interactive_weight + 1.0
          end
          if follow > 0 then
            interactive_weight = interactive_weight + 6.5
          end
          if forward > 0 then
            interactive_weight = interactive_weight + 4.5
          end
          if comment > 0 then
            interactive_weight = interactive_weight + 4.0
          end
          if collect > 0 then
            interactive_weight = interactive_weight + 1.5
          end
          if forward_inside > 0 then
            interactive_weight = interactive_weight + 6.0
          end
          if video_screen_shot > 0 then
            interactive_weight = interactive_weight + 1.0
          end
          if down_load > 0 then
            interactive_weight = interactive_weight + 1.0
          end
        end
        return interactive_weight, interact_label, forward_inside_weight
      end
    """) \
  .enrich_attr_by_lua(
    import_common_attr=["comment_watch_num_label_div_list"],
    import_item_attr=["watch_comment_num"],
    function_for_item="calculate",
    export_item_attr=["comment_watch_num_label_slot1", "comment_watch_num_label_slot2", "comment_watch_num_label_slot3",
      "comment_watch_num_label_slot4", "comment_watch_num_label_slot5", "comment_watch_num_label_slot6", "comment_watch_num_label_slot7"],
    lua_script=f"""
      function calculate()
        local div_list = comment_watch_num_label_div_list
        local comment_watch_num_label_slot1 = 0
        local comment_watch_num_label_slot2 = 0
        local comment_watch_num_label_slot3 = 0
        local comment_watch_num_label_slot4 = 0
        local comment_watch_num_label_slot5 = 0
        local comment_watch_num_label_slot6 = 0
        local comment_watch_num_label_slot7 = 0
        if div_list[2] <= watch_comment_num and watch_comment_num < div_list[3] then
          comment_watch_num_label_slot4 = 1
        end
        if div_list[3] <= watch_comment_num and watch_comment_num < div_list[4] then
          comment_watch_num_label_slot2 = 1
        end
        if div_list[4] <= watch_comment_num and watch_comment_num < div_list[5] then
          comment_watch_num_label_slot2 = 1
          comment_watch_num_label_slot5 = 1
        end
        if div_list[5] <= watch_comment_num and watch_comment_num < div_list[6] then
          comment_watch_num_label_slot1 = 1
        end
        if div_list[6] <= watch_comment_num and watch_comment_num < div_list[7] then
          comment_watch_num_label_slot1 = 1
          comment_watch_num_label_slot6 = 1
        end
        if div_list[7] <= watch_comment_num and watch_comment_num < div_list[8] then
          comment_watch_num_label_slot1 = 1
          comment_watch_num_label_slot3 = 1
        end
        if div_list[8] <= watch_comment_num then
          comment_watch_num_label_slot1 = 1
          comment_watch_num_label_slot3 = 1
          comment_watch_num_label_slot7 = 1
        end
        return comment_watch_num_label_slot1, comment_watch_num_label_slot2, comment_watch_num_label_slot3, comment_watch_num_label_slot4, comment_watch_num_label_slot5, comment_watch_num_label_slot6, comment_watch_num_label_slot7
      end
    """) \
  .copy_user_meta_info(
    save_user_id_to_attr="uid",
  ) \
  .fetch_kgnn_neighbors(
    id_from_common_attr="uid",
    save_neighbors_to="user_ids",
    edge_attr_schema=NodeAttrSchema(0, 1).add_float_list_attr("emp_click_comment_rate_list", 1),
    kess_service="grpc_kgnn_user_click_comment_photo_emp_click_comment_rate_info-U2I",
    relation_name='U2I',
    shard_num=1,
    sample_num=1,
    timeout_ms=20,
    sample_type="topn",
    padding_type="zero",
  ) \
  .fetch_kgnn_neighbors(
    id_from_common_attr="uid",
    save_neighbors_to="user_ids",
    edge_attr_schema=NodeAttrSchema(0, 1).add_float_list_attr("emp_comment_consume_depth_list", 1),
    kess_service="grpc_kgnn_user_click_comment_photo_emp_comment_consume_depth_info-U2I",
    relation_name='U2I',
    shard_num=1,
    sample_num=1,
    timeout_ms=20,
    sample_type="topn",
    padding_type="zero",
  ) \
  .fetch_kgnn_neighbors(
    id_from_common_attr="uid",
    save_neighbors_to="user_ids",
    edge_attr_schema=NodeAttrSchema(0, 1).add_float_list_attr("emp_comment_stay_duration_list", 1),
    kess_service="grpc_kgnn_user_click_comment_photo_emp_comment_stay_duration-U2I",
    relation_name='U2I',
    shard_num=1,
    sample_num=1,
    timeout_ms=20,
    sample_type="topn",
    padding_type="zero",
  ) \
  .enrich_attr_by_lua(
      import_common_attr=["emp_comment_consume_depth_list", "emp_click_comment_rate_list", "emp_comment_stay_duration_list"],
      export_common_attr=["emp_comment_consume_depth", "emp_click_comment_rate", "emp_comment_stay_duration"],
      function_for_common="trans",
      lua_script="""
        function trans()
          local emp_comment_consume_depth = 0.0
          local emp_click_comment_rate = 0.0
          local emp_comment_stay_duration = 0.0

          if emp_comment_consume_depth_list and #emp_comment_consume_depth_list > 0 then
            emp_comment_consume_depth = emp_comment_consume_depth_list[1]
          end

          if emp_click_comment_rate_list and #emp_click_comment_rate_list > 0 then
            emp_click_comment_rate = emp_click_comment_rate_list[1]
          end

          if emp_comment_stay_duration_list and #emp_comment_stay_duration_list > 0 then
            emp_comment_stay_duration = emp_comment_stay_duration_list[1]
          end
          return emp_comment_consume_depth, emp_click_comment_rate, emp_comment_stay_duration
        end
      """
  ) \
  .enrich_attr_by_lua(
    import_item_attr=["playing_time" , "comment_stay_time"],
    function_for_item="calculate",
    export_item_attr=["playing_time_s", "comment_stay_time_s"],
    lua_script="""
      function calculate()
        local playing_time_s = playing_time / 1000 
        local comment_stay_time_s = comment_stay_time / 1000 
        return playing_time_s, comment_stay_time_s
      end
    """) \
  .if_("(tab_id or 0) == 10000") \
    .enrich_attr_by_lua(
        function_for_common = "calculate",
        import_common_attr = ["uid"],
        export_common_attr = ["cmt_cluster_uid_str"],
        lua_script = """
            function calculate()
                cmt_cluster_uid_str = "cmt_cluster_cmt-" .. tostring(uid) .. "-KUAISHOU"
                return cmt_cluster_uid_str
            end
        """
    ) \
  .else_if_("(tab_id or 0) == 30000")  \
    .enrich_attr_by_lua(
        function_for_common = "calculate",
        import_common_attr = ["uid"],
        export_common_attr = ["cmt_cluster_uid_str"],
        lua_script = """
            function calculate()
                cmt_cluster_uid_str = "cmt_cluster_cmt-" .. tostring(uid) .. "-NEBULA"
                return cmt_cluster_uid_str
            end
        """
    ) \
  .end_() \
  .get_common_attr_from_redis(
    cluster_name="slideLeafRecoHighFansBoost",
    is_async = True,
    redis_params=[
        {
            "redis_key": "{{cmt_cluster_uid_str}}",
            "redis_value_type": "string",
            "output_attr_name": "user_comment_cluster_level",
            "output_attr_type": 'string'
        }
    ]
  ) \
  .enrich_attr_by_lua(
      function_for_common = "calculate",
      import_common_attr = ["user_comment_cluster_level"],
      export_common_attr = ["user_comment_cluster_level"],
      lua_script = """
          function calculate()
              user_comment_cluster_level = tonumber(user_comment_cluster_level) * 1.0 or 0.0
              return user_comment_cluster_level
          end
      """
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
            "output_attr_name": "user_app_cluster_level",
            "output_attr_type": 'string'
        }
    ]
  ) \
  .enrich_attr_by_lua(
      function_for_common = "calculate",
      import_common_attr = ["user_app_cluster_level"],
      export_common_attr = ["user_app_cluster_level"],
      lua_script = """
          function calculate()
              user_app_cluster_level = tonumber(user_app_cluster_level) * 1.0 or 0.0
              return user_app_cluster_level
          end
      """
  ) \
  .fetch_kgnn_neighbors(
    id_from_common_attr="user_id",
    save_weight_to="comment_weights",  # like + reply
    save_neighbors_to="user_action_comment_ids",
    edge_attr_schema=NodeAttrSchema(1, 0).add_int64_list_attr("comment_mmu_categories", 1),
    kess_service="grpc_kgnn_user_interact_comment_info-U2I",
    relation_name='U2I',
    shard_num=4,
    sample_num=20,
    timeout_ms=100,
    sample_type="topn",
    padding_type="zero",
  ) \
  .retrieve_by_remote_index(
    kess_service="reco-comment-ordered-index-server-main",
    shard_num=8,
    timeout_ms=20,
    reason=100,
    common_query="",
    querys=[{
        "query": "likeCount:{{photo_id}}",
        "search_num": 5
    }, {
        "query": "replyCount:{{photo_id}}",
        "search_num": 5
    }],
    default_search_num=5,
    ordered_index_send_by_shard=True,
    save_result_to_common_attr="photo_top_comment_ids"
  ) \
  .perflog_attr_value(
    check_point="default.training",
    item_attrs=["forward_inside", "interact_weight", "interact_label", "reset_count","playing_time","playing_time_s","comment_stay_time_s",
                  "time_weight", "forward_inside_weight", "click_comment_weight", "audit_count", "action_comment_content_forward", "action_comment_content_copy", "comment_copyward", "watch_comment_num",
                  "comment_watch_num_label_slot1", "comment_watch_num_label_slot2", "comment_watch_num_label_slot3", "comment_watch_num_label_slot4", 
                  "comment_watch_num_label_slot5", "comment_watch_num_label_slot6", "comment_watch_num_label_slot7",
                  "emp_comment_consume_depth", "emp_click_comment_rate", "emp_comment_stay_duration", "action_comment_slide_down",
                  "playing_time_after_click_comment","playing_time_after_out_of_click_comment","playing_time_after_out_of_click_comment_label","action_click_comment_timestamp_ms_label",
                  "wt_before_click_comment", "wt_before_write_comment", "eft_click_cmt", "eft_write_cmt", "comment"],
    common_attrs=["uid"],
  ) \
  .extract_with_ks_sign_feature(
    feature_list=load_feature_list_sign(os.path.join(current_dir, "./feature_list_sign.txt")),
    user_info_attr="user_info",
    tab_id_attr="tab_id",
    photo_info_attr="photo_info",
    context_info_attr="context_info",
    reason_attr="reason",
    pctr_attr="pctr",
    pltr_attr="pltr",
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
  )
    
send_to_mio_learner = DataReaderFlow(name = "send_to_mio_learner") \
  .send_to_mio_learner(
    attrs=labels+["interact_weight", "interact_label","playing_time","playing_time_s","comment_stay_time_s","user_comment_cluster_level","user_app_cluster_level",
                  "time_weight", "forward_in_count", "forward_inside_weight","long_view",
                  "comment", "comment_effective_stay", "follow_weight", "comment_action_weight", "comment_copyward", "watch_comment_num", "comment_stay_time",
                  "comment_watch_num_label_slot1", "comment_watch_num_label_slot2", "comment_watch_num_label_slot3", "comment_watch_num_label_slot4", 
                  "comment_watch_num_label_slot5", "comment_watch_num_label_slot6", "comment_watch_num_label_slot7",
                  "emp_comment_consume_depth", "emp_click_comment_rate", "emp_comment_stay_duration",
                  "playing_time_after_click_comment","real_show_timestamp_ms","playing_time","action_click_comment_timestamp_ms","action_click_comment_timestamp_ms_label",
                  "playing_time_after_out_of_click_comment_label","playing_time_after_out_of_click_comment", "effective_read_comment_fresh_weight", "fresh_true_sample_label"] + pxtr_list,
    slots_attrs = ["common_slots", "item_slots", "kuiba_common_slots", "kuiba_item_slots"],
    signs_attrs = ["common_signs", "item_signs", "kuiba_common_signs", "kuiba_item_signs"],
    label_attr="click")


runner = OfflineRunner("mio_offline_runner")

pipelines = [read_joint_reco_log, send_to_mio_learner]

runner.IGNORE_UNUSED_ATTR=["comment_watch_num_label_div_list_str", "emp_click_comment_rate_list", "emp_comment_consume_depth_list", "emp_comment_stay_duration_list", "user_ids",'comment_mmu_categories', 'comment_weights']
runner.add_leaf_flows(leaf_flows = pipelines)
runner.build(output_file=os.path.join(current_dir, "cofea_reader.json"))