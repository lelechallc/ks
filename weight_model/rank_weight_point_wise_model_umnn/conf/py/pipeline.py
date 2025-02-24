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
os.environ["DRAGONFLY_CHECK_UNUSED_ATTR"] = "false"
from dragonfly.common_leaf_dsl import LeafFlow, OfflineRunner
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.gsu.gsu_api_mixin import GsuApiMixin
from dragonfly.ext.kgnn.kgnn_api_mixin import KgnnApiMixin
from dragonfly.ext.cofea.cofea_api_mixin import CofeaApiMixin

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
class DataReaderFlow(LeafFlow, MioApiMixin, KuibaApiMixin, OfflineApiMixin, GsuApiMixin, KgnnApiMixin, CofeaApiMixin):
  pass

labels = ["forward_inside", "click_comment_button", "follow",  "comment_watch_time", "comment_coeff", "comment_action_coeff", "comment_stay_coeff", "effective_view", "action_expand_secondary_comment_count",
          "action_like_comment", "action_comment_content_copy", "action_comment_content_forward", "comment_long_stay", "effective_read_comment", "comment_copyward"]
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
hdfs_path = "viewfs:///home/reco_5/mpi/mtl_interact/interact_data/20240619/09/*/*"
new_hdfs_path = "viewfs:///home/reco_5/mpi/mtl_interact/interact_data_new/10/*/*"

# pairwise user侧特征保持一份 item侧特征要复制为2份
all_photo_slots = [201,202]
common_slots = [38, 34, 190, 35, 184, 189, 603, 3621,
901,902,903,904,905,906,907,908,909,910,911,912,913,914,915,916,917,918,919,920,921,950,951,952,953,954,956,957,958,959,960,961]
pair_slot_offset_list = [0,10000]
all_need_p_labels = ["level_reward", "level_reward_v2", "click", "photo_id", "pltr","pwtr", "pftr", "pcmtr", "plvtr", "pctr",
"like","follow","click_comment_button","long_view","forward","comment","effective_view"]
all_pass_p_labels = [label + "_idx%d"%idx for idx in range(2) for label in all_need_p_labels]


fetch_message_configs = dict()
if 'kafka' in data_sources:
  fetch_message_configs.update(dict(
    group_id='slide_multi_task_interact_v1_relative_quantile_v2',
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
                path="reco_labels.bool_value")
          ]
  ) \
  .filter_by_attr(attr_name="real_show", remove_if="==", compare_to=0, name="remove_no_real_show") \
  .gsu_common_colossusv2_enricher(  # colossus 新架构访问
    kconf="colossus.kconf_client.video_item",
    limit=1000,
    filter_future_items=True,
    item_fields=dict(
        photo_id="colossus_photo_id_list",
        author_id="colossus_author_id_list",
        duration="colossus_duration_list",
        play_time="colossus_play_time_list",
        tag="colossus_tag_list",
        channel="colossus_channel_list",
        label="colossus_label_list",
        timestamp="colossus_timestamp_list")
  ) \
  .enrich_attr_by_lua(
    import_common_attr = ['colossus_label_list', 'colossus_play_time_list', 'colossus_duration_list'],
    function_for_common = 'calcHistoryConsumeList', # 从colossus中把label提取出来
    export_common_attr = ['history_like_list', 'history_follow_list', 'history_forward_list', 'history_comment_list', 'history_click_comment_list',
    'history_effective_veiw_list', 'history_long_veiw_list'],
    lua_script = """
      function extractLabel(label_list)
        local likeList = {}
        local followList = {}
        local forwardList = {}
        local commentList = {}
        local clickCommentList = {}
        
        for i=1, #label_list do
          local is_like = label_list[i] & 1 
          local is_follow = (label_list[i] >> 1) & 1
          local is_forward = (label_list[i] >> 2) & 1
          local is_comment = (label_list[i] >> 4) & 1
          local is_click_comment = (label_list[i] >> 8) & 1
          table.insert(likeList, is_like)
          table.insert(followList, is_follow)
          table.insert(forwardList, is_forward)
          table.insert(commentList, is_comment)
          table.insert(clickCommentList, is_click_comment)
        end

        return likeList, followList, forwardList, commentList, clickCommentList
      end
      
      function calcEffectiveView(playing_time, duration_ms)
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

      function calcLongView(playing_time, duration_ms)
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

      function extractPlayTimeList(play_time_list, duration_list)
        local effectiveViewList = {}
        local longViewList = {}
        if #play_time_list ~= #duration_list then
          return effectiveViewList, longViewList
        end
        for i=1, #play_time_list do
          local play_time_ms = play_time_list[i] * 1000
          local duration_ms = duration_list[i] * 1000
          local is_effective_view = calcEffectiveView(play_time_ms, duration_ms)
          local is_long_view = calcLongView(play_time_ms, duration_ms)
          table.insert(effectiveViewList, is_effective_view)
          table.insert(longViewList, is_long_view)
        end
        return effectiveViewList, longViewList
      end 

      function calcHistoryConsumeList(seq, item_key, reason, score)
        local likeList = {}
        local followList = {}
        local forwardList = {}
        local commentList = {}
        local clickCommentList = {}
        local effectiveViewList = {}
        local longViewList = {}

        likeList, followList, forwardList, commentList, clickCommentList = extractLabel(colossus_label_list)
        effectiveViewList, longViewList = extractPlayTimeList(colossus_play_time_list, colossus_duration_list)

        return likeList, followList, forwardList, commentList, clickCommentList, effectiveViewList, longViewList
      end
    """
  ) \
  .enrich_attr_by_lua(
    import_common_attr = ['history_like_list', 'history_follow_list', 'history_forward_list', 'history_comment_list', 'history_click_comment_list',
    'history_effective_veiw_list', 'history_long_veiw_list', 'colossus_timestamp_list'],
    function_for_common = 'calcTimeWindowPxtr', # 最近x天内用户的pxtr
    export_common_attr = ['recent_ltr','recent_wtr','recent_ftr','recent_cmtr','recent_cmef', 'recent_ctr', 'recent_lvtr'],
    lua_script = """
      function calcTimeWindowPxtr(seq, item_key, reason, score)
        local targetDiffDays = 30 -- 计算目标天数内 用户的习惯
        local recent_realshow_cnt = 0
        local recent_like_cnt = 0
        local recent_follow_cnt = 0
        local recent_forward_cnt = 0
        local recent_comment_cnt = 0
        local recent_click_comment_cnt = 0
        local recent_effective_view_cnt = 0
        local recent_long_view_cnt = 0
        local cur_timestamp = util.GetTimestamp() / 1000000 -- 转换成秒
        
        for i=1,#colossus_timestamp_list do
          local diffDays = (cur_timestamp - colossus_timestamp_list[i]) / (60 * 60 * 24)
          if diffDays <= targetDiffDays and diffDays > 0 then
            recent_realshow_cnt = recent_realshow_cnt + 1
            if history_like_list[i] > 0 then
              recent_like_cnt = recent_like_cnt + 1
            end
            if history_follow_list[i] > 0 then
              recent_follow_cnt = recent_follow_cnt + 1
            end
            if history_forward_list[i] > 0 then
              recent_forward_cnt = recent_forward_cnt + 1
            end
            if history_comment_list[i] > 0 then
              recent_comment_cnt = recent_comment_cnt + 1
            end
            if history_click_comment_list[i] > 0 then
              recent_click_comment_cnt = recent_click_comment_cnt + 1
            end
            if history_effective_veiw_list[i] > 0 then
              recent_effective_view_cnt = recent_effective_view_cnt + 1
            end
            if history_long_veiw_list[i] > 0 then
              recent_long_view_cnt = recent_long_view_cnt + 1
            end
          end
        end

        local recent_ltr = 0
        local recent_wtr = 0
        local recent_cmtr = 0
        local recent_ftr = 0
        local recent_cmef = 0
        local recent_ctr = 0
        local recent_lvtr = 0
        if recent_realshow_cnt > 0 then
          recent_ltr = recent_like_cnt / recent_realshow_cnt
          recent_wtr = recent_follow_cnt / recent_realshow_cnt
          recent_ftr = recent_forward_cnt / recent_realshow_cnt
          recent_cmtr = recent_comment_cnt / recent_realshow_cnt
          recent_cmef = recent_click_comment_cnt / recent_realshow_cnt
          recent_ctr = recent_effective_view_cnt / recent_realshow_cnt
          recent_lvtr = recent_long_view_cnt / recent_realshow_cnt
        end

        return recent_ltr,recent_wtr,recent_ftr,recent_cmtr,recent_cmef,recent_ctr,recent_lvtr
      end
    """
  ) \
  .enrich_attr_by_lua(
    import_common_attr = ['tab_id'],
    function_for_common = 'getProduct',
    export_common_attr = ['product','preferLtrKey','preferWtrKey','preferCmtrKey','preferFtrKey','preferCmefKey','preferCtrKey','preferLvtrKey'],
    lua_script="""
        function getProduct()
          local tab_id = tab_id or 0
          local product = ""
          if tab_id == 30000 then
            product = "NEBULA"
          elseif tab_id == 10000 then
            product = "KUAISHOU"
          end
          local preferLtrKey = product .. "_percentile_ltr"
          local preferWtrKey = product .. "_percentile_wtr"
          local preferCmtrKey = product .. "_percentile_cmtr"
          local preferFtrKey = product .. "_percentile_ftr"
          local preferCmefKey = product .. "_percentile_cmef"
          local preferCtrKey = product .. "_percentile_ctr"
          local preferLvtrKey = product .. "_percentile_lvtr"
          return product,preferLtrKey,preferWtrKey,preferCmtrKey,preferFtrKey,preferCmefKey,preferCtrKey,preferLvtrKey
        end
    """,
    debug_log = True
  ) \
  .get_common_attr_from_redis(
    cluster_name = "slideLeafRecoHighFansBoost",
    redis_params = [
      {
        "redis_key": "{{preferLtrKey}}",
        "output_attr_name": "preferLtrPercentile"
      }, {
        "redis_key": "{{preferWtrKey}}",
        "output_attr_name": "preferWtrPercentile"
      }, {
        "redis_key": "{{preferCmtrKey}}",
        "output_attr_name": "preferCmtrPercentile"
      }, {
        "redis_key": "{{preferFtrKey}}",
        "output_attr_name": "preferFtrPercentile"
      }, {
        "redis_key": "{{preferCmefKey}}",
        "output_attr_name": "preferCmefPercentile"
      }, {
        "redis_key": "{{preferCtrKey}}",
        "output_attr_name": "preferCtrPercentile"
      }, {
        "redis_key": "{{preferLvtrKey}}",
        "output_attr_name": "preferLvtrPercentile"
      }
    ]
  ) \
  .split_string(
    input_common_attr = "preferLvtrPercentile",
    output_common_attr = "preferlvtrPercentile_list",
    delimiters=",",
    parse_to_double = True
  ) \
  .split_string(
    input_common_attr = "preferLtrPercentile",
    output_common_attr = "preferltrPercentile_list",
    delimiters=",",
    parse_to_double = True
  ) \
  .split_string(
    input_common_attr = "preferWtrPercentile",
    output_common_attr = "preferwtrPercentile_list",
    delimiters=",",
    parse_to_double = True
  ) \
  .split_string(
    input_common_attr = "preferCmtrPercentile",
    output_common_attr = "prefercmtrPercentile_list",
    delimiters=",",
    parse_to_double = True
  ) \
  .split_string(
    input_common_attr = "preferFtrPercentile",
    output_common_attr = "preferftrPercentile_list",
    delimiters=",",
    parse_to_double = True
  ) \
  .split_string(
    input_common_attr = "preferCmefPercentile",
    output_common_attr = "prefercmefPercentile_list",
    delimiters=",",
    parse_to_double = True
  ) \
  .split_string(
    input_common_attr = "preferCtrPercentile",
    output_common_attr = "preferctrPercentile_list",
    delimiters=",",
    parse_to_double = True
  ) \
  .enrich_attr_by_lua(
    import_common_attr=["preferlvtrPercentile_list", "preferltrPercentile_list", "preferwtrPercentile_list", "prefercmtrPercentile_list", 
                      "preferftrPercentile_list", "prefercmefPercentile_list", "preferctrPercentile_list",
                      'recent_ltr','recent_wtr','recent_ftr','recent_cmtr','recent_cmef', 'recent_ctr', 'recent_lvtr',
                      'tab_id'],
    function_for_common="calculate",
    export_common_attr=["UserlvtrLevelKey", "UserlvtrHourlyLevelKey", "UserltrLevelKey", "UserltrHourlyLevelKey",
                      "UserwtrLevelKey", "UserwtrHourlyLevelKey", "UsercmtrLevelKey", "UsercmtrHourlyLevelKey",
                      "UserftrLevelKey", "UserftrHourlyLevelKey", "UsercmefLevelKey", "UsercmefHourlyLevelKey", 
                      "UserctrLevelKey", "UserctrHourlyLevelKey"],
    lua_script="""
      function calculate()
        local tab_id = tab_id or 0
          local product = ""
          if tab_id == 30000 then
            product = "NEBULA"
          elseif tab_id == 10000 then
            product = "KUAISHOU"
          end

        local firstFloat_ltr = preferltrPercentile_list[1]
        local secondFloat_ltr = preferltrPercentile_list[2]
        local UserltrLevelKey = ""
        local UserltrHourlyLevelKey = ""
        if recent_ltr > 0 and recent_ltr < firstFloat_ltr then
            UserltrLevelKey = product .. "_avg_ltr_1"
            UserltrHourlyLevelKey =  product .. "_hr_avg_ltr_1"
        elseif firstFloat_ltr < recent_ltr and recent_ltr < secondFloat_ltr then
            UserltrLevelKey =  product .. "_avg_ltr_2"
            UserltrHourlyLevelKey =  product .. "_hr_avg_ltr_2"
        elseif recent_ltr > secondFloat_ltr then
            UserltrLevelKey =  product .. "_avg_ltr_3"
            UserltrHourlyLevelKey =  product .. "_hr_avg_ltr_3"
        end

        local firstFloat_wtr = preferwtrPercentile_list[1]
        local secondFloat_wtr = preferwtrPercentile_list[2]
        local UserwtrLevelKey = ""
        local UserwtrHourlyLevelKey = ""
        if recent_wtr > 0 and recent_wtr < firstFloat_wtr then
            UserwtrLevelKey = product .. "_avg_wtr_1"
            UserwtrHourlyLevelKey =  product .. "_hr_avg_wtr_1"
        elseif firstFloat_wtr < recent_wtr and recent_wtr < secondFloat_wtr then
            UserwtrLevelKey =  product .. "_avg_wtr_2"
            UserwtrHourlyLevelKey =  product .. "_hr_avg_wtr_2"
        elseif recent_wtr > secondFloat_wtr then
            UserwtrLevelKey =  product .. "_avg_wtr_3"
            UserwtrHourlyLevelKey =  product .. "_hr_avg_wtr_3"
        end

        local firstFloat_cmtr = prefercmtrPercentile_list[1]
        local secondFloat_cmtr = prefercmtrPercentile_list[2]
        local UsercmtrLevelKey = ""
        local UsercmtrHourlyLevelKey = ""
        if recent_cmtr > 0 and recent_cmtr < firstFloat_cmtr then
            UsercmtrLevelKey = product .. "_avg_cmtr_1"
            UsercmtrHourlyLevelKey =  product .. "_hr_avg_cmtr_1"
        elseif firstFloat_cmtr < recent_cmtr and recent_cmtr < secondFloat_cmtr then
            UsercmtrLevelKey =  product .. "_avg_cmtr_2"
            UsercmtrHourlyLevelKey =  product .. "_hr_avg_cmtr_2"
        elseif recent_cmtr > secondFloat_cmtr then
            UsercmtrLevelKey =  product .. "_avg_cmtr_3"
            UsercmtrHourlyLevelKey =  product .. "_hr_avg_cmtr_3"
        end

        local firstFloat_ftr = preferftrPercentile_list[1]
        local secondFloat_ftr = preferftrPercentile_list[2]
        local UserftrLevelKey = ""
        local UserftrHourlyLevelKey = ""
        if recent_ftr > 0 and recent_ftr < firstFloat_ftr then
            UserftrLevelKey = product .. "_avg_ftr_1"
            UserftrHourlyLevelKey =  product .. "_hr_avg_ftr_1"
        elseif firstFloat_ftr < recent_ftr and recent_ftr < secondFloat_ftr then
            UserftrLevelKey =  product .. "_avg_ftr_2"
            UserftrHourlyLevelKey =  product .. "_hr_avg_ftr_2"
        elseif recent_ftr > secondFloat_ftr then
            UserftrLevelKey =  product .. "_avg_ftr_3"
            UserftrHourlyLevelKey =  product .. "_hr_avg_ftr_3"
        end

        local firstFloat_cmef = prefercmefPercentile_list[1]
        local secondFloat_cmef = prefercmefPercentile_list[2]
        local UsercmefLevelKey = ""
        local UsercmefHourlyLevelKey = ""
        if recent_cmef > 0 and recent_cmef < firstFloat_cmef then
            UsercmefLevelKey = product .. "_avg_cmef_1"
            UsercmefHourlyLevelKey =  product .. "_hr_avg_cmef_1"
        elseif firstFloat_cmef < recent_cmef and recent_cmef < secondFloat_cmef then
            UsercmefLevelKey =  product .. "_avg_cmef_2"
            UsercmefHourlyLevelKey =  product .. "_hr_avg_cmef_2"
        elseif recent_cmef > secondFloat_cmef then
            UsercmefLevelKey =  product .. "_avg_cmef_3"
            UsercmefHourlyLevelKey =  product .. "_hr_avg_cmef_3"
        end

        local firstFloat_ctr = preferctrPercentile_list[1]
        local secondFloat_ctr = preferctrPercentile_list[2]
        local UserctrLevelKey = ""
        local UserctrHourlyLevelKey = ""
        if recent_ctr > 0 and recent_ctr < firstFloat_ctr then
            UserctrLevelKey = product .. "_avg_ctr_1"
            UserctrHourlyLevelKey =  product .. "_hr_avg_ctr_1"
        elseif firstFloat_ctr < recent_ctr and recent_ctr < secondFloat_ctr then
            UserctrLevelKey =  product .. "_avg_ctr_2"
            UserctrHourlyLevelKey =  product .. "_hr_avg_ctr_2"
        elseif recent_ctr > secondFloat_ctr then
            UserctrLevelKey =  product .. "_avg_ctr_3"
            UserctrHourlyLevelKey =  product .. "_hr_avg_ctr_3"
        end

        local firstFloat_lvtr = preferlvtrPercentile_list[1]
        local secondFloat_lvtr = preferlvtrPercentile_list[2]
        local UserlvtrLevelKey = ""
        local UserlvtrHourlyLevelKey = ""
        if recent_lvtr > 0 and recent_lvtr < firstFloat_lvtr then
            UserlvtrLevelKey = product .. "_avg_lvtr_1"
            UserlvtrHourlyLevelKey =  product .. "_hr_avg_lvtr_1"
        elseif firstFloat_lvtr < recent_lvtr and recent_lvtr < secondFloat_lvtr then
            UserlvtrLevelKey =  product .. "_avg_lvtr_2"
            UserlvtrHourlyLevelKey =  product .. "_hr_avg_lvtr_2"
        elseif recent_lvtr > secondFloat_lvtr then
            UserlvtrLevelKey =  product .. "_avg_lvtr_3"
            UserlvtrHourlyLevelKey =  product .. "_hr_avg_lvtr_3"
        end

        return UserlvtrLevelKey, UserlvtrHourlyLevelKey, UserltrLevelKey, UserltrHourlyLevelKey,
               UserwtrLevelKey, UserwtrHourlyLevelKey, UsercmtrLevelKey, UsercmtrHourlyLevelKey,
               UserftrLevelKey, UserftrHourlyLevelKey, UsercmefLevelKey, UsercmefHourlyLevelKey, 
               UserctrLevelKey, UserctrHourlyLevelKey
      end
    """) \
  .get_common_attr_from_redis(
    cluster_name = "slideLeafRecoHighFansBoost",
    redis_params = [
      {
        "redis_key": "{{UserlvtrLevelKey}}",
        "output_attr_name": "UserlvtrLevelAVG"
      }, {
        "redis_key": "{{UserltrLevelKey}}",
        "output_attr_name": "UserltrLevelAVG"
      }, {
        "redis_key": "{{UserwtrLevelKey}}",
        "output_attr_name": "UserwtrLevelAVG"
      }, {
        "redis_key": "{{UsercmtrLevelKey}}",
        "output_attr_name": "UsercmtrLevelAVG"
      }, {
        "redis_key": "{{UserftrLevelKey}}",
        "output_attr_name": "UserftrLevelAVG"
      }, {
        "redis_key": "{{UsercmefLevelKey}}",
        "output_attr_name": "UsercmefLevelAVG"
      }, {
        "redis_key": "{{UserctrLevelKey}}",
        "output_attr_name": "UserctrLevelAVG"
      },      {
        "redis_key": "{{UserlvtrHourlyLevelKey}}",
        "output_attr_name": "UserlvtrHourlyLevelAVG"
      }, {
        "redis_key": "{{UserltrHourlyLevelKey}}",
        "output_attr_name": "UserltrHourlyLevelAVG"
      }, {
        "redis_key": "{{UserwtrHourlyLevelKey}}",
        "output_attr_name": "UserwtrHourlyLevelAVG"
      }, {
        "redis_key": "{{UsercmtrHourlyLevelKey}}",
        "output_attr_name": "UsercmtrHourlyLevelAVG"
      }, {
        "redis_key": "{{UserftrHourlyLevelKey}}",
        "output_attr_name": "UserftrHourlyLevelAVG"
      }, {
        "redis_key": "{{UsercmefHourlyLevelKey}}",
        "output_attr_name": "UsercmefHourlyLevelAVG"
      }, {
        "redis_key": "{{UserctrHourlyLevelKey}}",
        "output_attr_name": "UserctrHourlyLevelAVG"
      }
    ]
  ) \
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
                      "action_expand_secondary_comment_count_abs", "action_comment_click_head_abs", "action_comment_click_nickname_abs", "action_comment_content_copy_abs", "action_comment_content_forward_abs", "watch_comment_num"],
    lua_script="""
      function calculate()
        local names = label_names or {}
        if #names == 0 then
          return 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0
        end
        action_like_comment = 0
        action_expand_secondary_comment_count = 0
        action_comment_click_head = 0
        action_comment_click_nickname = 0
        action_comment_content_copy = 0
        action_comment_content_forward = 0
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
        end
        comment_copyward = math.max(action_comment_content_forward, action_comment_content_copy)
        return action_like_comment, action_expand_secondary_comment_count, action_comment_click_head, action_comment_click_nickname, action_comment_content_copy, action_comment_content_forward, action_comment_search_highlight_click, action_comment_search_trending_click, comment_copyward,
               action_expand_secondary_comment_count_abs, action_comment_click_head_abs, action_comment_click_nickname_abs, action_comment_content_copy_abs, action_comment_content_forward_abs, watch_comment_num
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
            interactive_weight = interactive_weight + 3.0
          end
          if like > 0 then
            interactive_weight = interactive_weight + 1.0
          end
          if follow > 0 then
            interactive_weight = interactive_weight + 10.0
          end
          if forward > 0 then
            interactive_weight = interactive_weight + 1.0
          end
          if comment > 0 then
            interactive_weight = interactive_weight + 10.0
          end
          if collect > 0 then
            interactive_weight = interactive_weight + 1.0
          end
          if forward_inside > 0 then
            interactive_weight = interactive_weight + 1.0
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
      import_item_attr=["like", "follow", "forward", "comment", "collect", 
            "unfollow", "unlike", "cancel_collect", "feedback_negative", 
            "playing_time", "duration_ms", "effective_view", "long_view",
            "forward_outside", "forward_inside", "click_comment_button"],
      import_common_attr=['UserlvtrLevelAVG','UserltrLevelAVG','UserwtrLevelAVG','UsercmtrLevelAVG','UserftrLevelAVG','UsercmefLevelAVG','UserctrLevelAVG',
            'UserlvtrHourlyLevelAVG','UserltrHourlyLevelAVG','UserwtrHourlyLevelAVG','UsercmtrHourlyLevelAVG','UserftrHourlyLevelAVG','UsercmefHourlyLevelAVG','UserctrHourlyLevelAVG'],
      function_for_item="calculate",
      export_item_attr=["good_quality", "level_reward", "level_reward_v2","ltr_alpha", "wtr_alpha", "ftr_alpha", "cmtr_alpha", 
                        "cmef_alpha", "ctr_alpha", "lvtr_alpha","ctr_weight_label", "lvtr_weight_label", "ltr_weight_label", "cmtr_weight_label", "wtr_weight_label", "ftr_weight_label"],
      lua_script="""
        function calculate()
          local ltr_alpha = 1.0
          if #(UserltrLevelAVG or '') > 0 and #(UserltrHourlyLevelAVG or '') > 0 then
            ltr_alpha = tonumber(UserlvtrHourlyLevelAVG) / tonumber(UserlvtrLevelAVG)
            if ltr_alpha <= 1.0 then
              ltr_alpha = ltr_alpha ^ -10
            else
              ltr_alpha = ltr_alpha ^ 3
            end
          end

          local wtr_alpha = 1.0
          if #(UserwtrLevelAVG or '') > 0 and #(UserwtrHourlyLevelAVG or '') > 0 then
            wtr_alpha = tonumber(UserwtrHourlyLevelAVG) / tonumber(UserwtrLevelAVG)
            if wtr_alpha <= 1.0 then
              wtr_alpha = wtr_alpha ^ -10
            else
              wtr_alpha = wtr_alpha ^ 3
            end
          end

          local ftr_alpha = 1.0
          if #(UserftrLevelAVG or '') > 0 and #(UserftrHourlyLevelAVG or '') > 0 then
            ftr_alpha = tonumber(UserftrHourlyLevelAVG) / tonumber(UserftrLevelAVG)
            if ftr_alpha <= 1.0 then
              ftr_alpha = ftr_alpha ^ -10
            else
              ftr_alpha = ftr_alpha ^ 3
            end
          end

          local cmtr_alpha = 1.0
          if #(UsercmtrLevelAVG or '') > 0 and #(UsercmtrHourlyLevelAVG or '') > 0 then
            cmtr_alpha = tonumber(UsercmtrHourlyLevelAVG) / tonumber(UsercmtrLevelAVG)
            if cmtr_alpha <= 1.0 then
              cmtr_alpha = cmtr_alpha ^ -10
            else
              cmtr_alpha = cmtr_alpha ^ 3
            end
          end

          local cmef_alpha = 1.0
          if #(UsercmefLevelAVG or '') > 0 and #(UsercmefHourlyLevelAVG or '') > 0 then
            cmef_alpha = tonumber(UsercmefHourlyLevelAVG) / tonumber(UsercmefLevelAVG)
            if cmef_alpha <= 1.0 then
              cmef_alpha = cmef_alpha ^ -10
            else
              cmef_alpha = cmef_alpha ^ 3
            end
          end

          local ctr_alpha = 1.0
          if #(UserctrLevelAVG or '') > 0 and #(UserctrHourlyLevelAVG or '') > 0 then
            ctr_alpha = tonumber(UserctrHourlyLevelAVG) / tonumber(UserctrLevelAVG)
            if ctr_alpha <= 1.0 then
              ctr_alpha = ctr_alpha ^ -10
            else
              ctr_alpha = ctr_alpha ^ 3
            end
          end
          
          local lvtr_alpha = 1.0
          if #(UserlvtrLevelAVG or '') > 0 and #(UserlvtrHourlyLevelAVG or '') > 0 then
            lvtr_alpha = tonumber(UserlvtrHourlyLevelAVG) / tonumber(UserlvtrLevelAVG)
            if lvtr_alpha <= 1.0 then
              lvtr_alpha = lvtr_alpha ^ -10
            else
              lvtr_alpha = lvtr_alpha ^ 3
            end
          end
          
          local finish_rate = math.min(2.0, playing_time * 1.0 / (duration_ms + 1.0))
          local ctr_weight_label = (ctr_alpha * effective_view > 0) and 1 or 0
          local lvtr_weight_label = (lvtr_alpha * long_view > 0) and 1 or 0
          local ltr_weight_label = (ltr_alpha * like > 0) and 1 or 0
          local cmtr_weight_label = (cmtr_alpha * comment > 0) and 1 or 0
          local wtr_weight_label = (wtr_alpha * follow > 0) and 1 or 0
          local ftr_weight_label = (ftr_alpha * forward > 0) and 1 or 0

          local level_reward = 1.0* finish_rate * ((0.00001 + 2 * effective_view + 5 * long_view + 1 * like + 10 * comment + 10 * follow  + 1 * forward + 1 * click_comment_button))
          local level_reward_v2 = 0.00001 + ctr_alpha * effective_view + lvtr_alpha * long_view + ltr_alpha * like + cmtr_alpha * comment + wtr_alpha * follow  + ftr_alpha * forward + cmef_alpha * click_comment_button
          local good_quality = 0
          if level_reward > 0 then
            good_quality = 1
          end
          return good_quality, level_reward,level_reward_v2, ltr_alpha, wtr_alpha, ftr_alpha, cmtr_alpha, cmef_alpha, ctr_alpha, lvtr_alpha,
                ctr_weight_label, lvtr_weight_label, ltr_weight_label, cmtr_weight_label, wtr_weight_label, ftr_weight_label
        end
      """) \
  .log_debug_info(
    common_attrs = ['product','preferlvtrPercentile','preferLtrPercentile','preferWtrPercentile','preferCmtrPercentile','preferFtrPercentile',
    'preferCmefPercentile','preferCtrPercentile','colossus_duration_list', 'colossus_timestamp_list', 'history_like_list', 'history_follow_list', 'history_forward_list', 'history_comment_list', 'history_click_comment_list',
    'history_effective_veiw_list', 'history_long_veiw_list',
    'recent_ltr','recent_wtr','recent_ftr','recent_cmtr','recent_cmef', 'recent_ctr', 'recent_lvtr',
    "UserlvtrLevelKey", "UserlvtrHourlyLevelKey", "UserltrLevelKey", "UserltrHourlyLevelKey",
    "UserwtrLevelKey", "UserwtrHourlyLevelKey", "UsercmtrLevelKey", "UsercmtrHourlyLevelKey",
    "UserftrLevelKey", "UserftrHourlyLevelKey", "UsercmefLevelKey", "UsercmefHourlyLevelKey", "UserctrLevelKey", "UserctrHourlyLevelKey",
    'UserlvtrLevelAVG','UserltrLevelAVG','UserwtrLevelAVG','UsercmtrLevelAVG','UserftrLevelAVG','UsercmefLevelAVG','UserctrLevelAVG',
    'UserlvtrHourlyLevelAVG','UserltrHourlyLevelAVG','UserwtrHourlyLevelAVG','UsercmtrHourlyLevelAVG','UserftrHourlyLevelAVG','UsercmefHourlyLevelAVG','UserctrHourlyLevelAVG',
    "level_reward", "level_reward_v2","ltr_alpha", "wtr_alpha", "ftr_alpha", "cmtr_alpha", "cmef_alpha", "ctr_alpha", "lvtr_alpha",
    "ctr_weight_label", "lvtr_weight_label", "ltr_weight_label", "cmtr_weight_label", "wtr_weight_label", "ftr_weight_label"
    ],
    for_debug_request_only = True,
    respect_sample_logging = True
  ) \
  .log_debug_info(
    common_attrs = ['colossus_author_id_list', 'colossus_channel_list', 'colossus_duration_list', 'colossus_label_list', 'colossus_photo_id_list', 'colossus_play_time_list', 'colossus_tag_list', 'colossus_timestamp_list']
  ) \
  .pack_item_attr(
    item_source = {
      "reco_results": True
    },
    mappings = [
      {
        "from_item_attr": "pltr",
        "to_common_attr": "u_mean_pltr",
        "aggregator": "avg"
      },
      {
        "from_item_attr": "pltr",
        "to_common_attr": "u_std_pltr",
        "aggregator": "dev"
      },
      {
        "from_item_attr": "pwtr",
        "to_common_attr": "u_mean_pwtr",
        "aggregator": "avg"
      },
      {
        "from_item_attr": "pwtr",
        "to_common_attr": "u_std_pwtr",
        "aggregator": "dev"
      },
      {
        "from_item_attr": "pftr",
        "to_common_attr": "u_mean_pftr",
        "aggregator": "avg"
      },
      {
        "from_item_attr": "pftr",
        "to_common_attr": "u_std_pftr",
        "aggregator": "dev"
      },
      {
        "from_item_attr": "pcmtr",
        "to_common_attr": "u_mean_pcmtr",
        "aggregator": "avg"
      },
      {
        "from_item_attr": "pcmtr",
        "to_common_attr": "u_std_pcmtr",
        "aggregator": "dev"
      },
      {
        "from_item_attr": "pcltr",
        "to_common_attr": "u_mean_pcltr",
        "aggregator": "avg"
      },
      {
        "from_item_attr": "pcltr",
        "to_common_attr": "u_std_pcltr",
        "aggregator": "dev"
      },
      {
        "from_item_attr": "pdtr",
        "to_common_attr": "u_mean_pdtr",
        "aggregator": "avg"
      },
      {
        "from_item_attr": "pdtr",
        "to_common_attr": "u_std_pdtr",
        "aggregator": "dev"
      }
    ]
) \
  .perflog_attr_value(
    check_point="default.training",
    item_attrs=["forward_inside", "interact_weight", "interact_label", "reset_count",
                "time_weight", "forward_inside_weight", "click_comment_weight", "reward"],
  ) \
  .extract_with_ks_sign_feature(
    feature_list=load_feature_list_sign(os.path.join(current_dir, "../feature_list_sign.txt")),
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
    slot_as_attr_name = True
  ) \
  .extract_kuiba_parameter(
    slots_output="kuiba_common_slots",
    parameters_output="kuiba_common_signs",
    is_common_attr=True,
    config=lua_script.COMMON_KUIBA_CONFIG,
    slot_as_attr_name=True
  ) \
  .extract_kuiba_parameter(
    slots_output="kuiba_item_slots",
    parameters_output="kuiba_item_signs",
    is_common_attr=False,
    config=lua_script.ITEM_KUIBA_CONFIG,
    slot_as_attr_name=True
  ) 

prepare_pair_item_flow = DataReaderFlow(name="prepare_pair_item_flow").cofea_sample_combine_pair(
    slot_id_list=all_photo_slots,
    label_attr_list=all_need_p_labels,
    slot_offset_list=[0,10000],
    filter_same_item_pair=True,
    save_combine_item_flag_to="is_combine_result"
  ).filter_by_rule(
    # 只保留pair_item
    rule = {
      "attr_name": "is_combine_result",
      "remove_if": "!=",
      "compare_to": 1,
      "remove_if_attr_missing": True
    }
  ) \
  .enrich_attr_by_lua(
    import_item_attr=['level_reward_idx0','level_reward_idx1'],
    function_for_item="judge",
      export_item_attr=["is_valid_pair"],
      lua_script="""
        function judge()
          local level_reward_idx0 = level_reward_idx0 or 0
          local level_reward_idx1 = level_reward_idx1 or 0
          local is_valid_pair = 0
          if (level_reward_idx0 - level_reward_idx1) > 1e-2 then
            is_valid_pair = 1
          end
          return is_valid_pair
        end
      """
  ) \
  .filter_by_attr(attr_name="is_valid_pair", remove_if="!=", compare_to=1, name="remove_no_valid") \


    
send_to_mio_learner = DataReaderFlow(name = "send_to_mio_learner") \
  .send_to_mio_learner(
    slots=common_slots + [i+j for j in pair_slot_offset_list for i in all_photo_slots],
    attrs=all_pass_p_labels,
    lineid_attr="user_id",
    user_hash_attr="device_id",
    time_ms_attr="_REQ_TIME_",
    pid_attr="photo_id_idx0",
    label_attr="click_idx0",
    slot_as_attr_name=True,
    debug_log=True)


runner = OfflineRunner("mio_offline_runner")

pipelines = [read_joint_reco_log, prepare_pair_item_flow, send_to_mio_learner]

runner.IGNORE_UNUSED_ATTR=["comment_watch_num_label_div_list_str",'battery_charging', 'battery_level', 'headset_state', 'net_state', 'screen_light','human_action',"audit_count","comment_watch_num_label_div_list"]
runner.add_leaf_flows(leaf_flows = pipelines)
runner.build(output_file=os.path.join(current_dir, "cofea_reader.json"))