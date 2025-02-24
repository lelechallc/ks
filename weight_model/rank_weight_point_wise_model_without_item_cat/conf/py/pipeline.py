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
comment_stay_time_threshold = 4000
effective_profile_stay_time_threshold = 4000
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

labels = ["interact"]
pxtr_weight = ["like_weight", "follow_weight", "collect_weight", "forward_weight", "comment_weight", "download_weight"]
pxtr_list = ["pltr","pwtr", "pftr", "pcmtr", "pcltr", "pdtr"]

# group_id = slide_related_cascade_rank
# slide_related_reco_rerank_log
data_sources = ['kafka']
hdfs_path = "viewfs:///home/reco_5/mpi/mtl_interact/interact_data/20231008_09/*/*"
new_hdfs_path = "viewfs:///home/reco_6/rawdata/light_joint_reco_log_slide_optimized/2023-10-08/09/*/*"


fetch_message_configs = dict()
if 'kafka' in data_sources:
  fetch_message_configs.update(dict(
    group_id='slide_multi_task_interact_fm',
    kafka_topic="slide_new_hot_train",
  ))
if 'hdfs' in data_sources:
  fetch_message_configs.update(dict(
    hdfs_path=new_hdfs_path,
    hdfs_read_thread_num=12,
    hdfs_timeout_ms=60 * 60 * 1000,
  ))

read_joint_reco_log = DataReaderFlow(name = "read_joint_reco_log") \
  .get_kconf_params(
    kconf_configs = 
    [
      {   
        "kconf_key": "cc.knowledgeGraph.slideInteractiveModelTrain",
        "json_path": "sample_rate_lag",
        "export_common_attr": "sample_rate",
        "default_value": 0.2,
      }, 
    ] 
  ) \
  .fetch_message(
    group_id='slide_multi_task_rank_weight_point_wise',
    kafka_topic='slide_new_hot_train',
    output_attr="ks_reco_log_str",
    begin_time_ms='2022-08-10 18') \
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
      # dict(path="time", name="req_time_ms" ),
    ]) \
  .retrieve_from_ks_reco_log(
    from_extra_var="ks_reco_log",
    save_reco_photo_to="reco_photo_info") \
  .enrich_with_protobuf(
    from_extra_var="reco_photo_info",
    is_common_attr=False,
    attrs=[
      dict(path="photo", name="photo_info"),
      "reason", "context_info", "pctr",
      "cascade_pctr", "cascade_plvtr", "cascade_psvr", "cascade_pltr"] + pxtr_list
  ) \
  .enrich_with_protobuf(
    from_extra_var="user_info",
    is_common_attr=True,
    attrs=[
      # dict(name="user_id", path="id"),
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
      # dict(name="FollowList", path="user_profile_v1.follow_list.author_id"),
  ]) \
  .enrich_with_protobuf(
    from_extra_var="photo_info",
    is_common_attr=False,
    attrs=[
      "duration_ms",
      "audit_b_second_tag",
      dict(name = "photo_hetu_one", path="hetu_tag_level_info.hetu_level_one",
            repeat_limit={"hetu_tag_level_info.hetu_level_one": 1},
            repeat_align=True),
      dict(name = "photo_hetu_two", path="hetu_tag_level_info.hetu_level_two",
            repeat_limit={"hetu_tag_level_info.hetu_level_two": 1},
            repeat_align=True),
      # dict(name="author_id", path="author.id"),
    ]
  ) \
  .enrich_with_protobuf(
    from_extra_var="context_info",
    is_common_attr=False, 
    attrs=["real_show", "playing_time", 
           "like", "follow", "forward", "comment", "collect", 
          "unfollow", "unlike", "discard", "dislike", "cancel_collect", "feedback_negative", "down_load",
           dict(name="label_names", 
                path="reco_labels.name_value"),
           dict(name="label_values", 
                path="reco_labels.int_value"),
          ]
  ) \
  .filter_by_attr(attr_name="real_show", remove_if="==", compare_to=0, name="remove_no_real_show") \
  .enrich_attr_by_lua(
    import_item_attr=["label_names", "label_values"],
    function_for_item="calculate",
    export_item_attr=["forward_inside_status"],
    lua_script="""
      function calculate()
        local names = label_names or {}
        if #names == 0 then
          return 0, 0
        end
        forward_status = 0
        for i = 1, #label_names do
          if label_names[i] == 404 then
            forward_status = label_values[i]
          end
        end
        return forward_status
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
                      "comment", "audit_b_second_tag"],
    function_for_item="reset",
    export_item_attr=["collect", "like", "follow", 
                      "comment", "audit_count"],
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
           audit_count = audit_count + 1
        end
        if audit_b_second_tag == 2147251 then
          like = 0
          follow = 0
          comment = 0
          collect = 0
        end
        return collect, like, follow, comment, audit_count
      end
    """) \
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
                      "forward_inside"],
    function_for_item="calculate",
    export_item_attr=["like", "follow", "forward", "comment", "collect", 
                      "forward_inside"],
    lua_script="""
      function calculate()
        if feedback_negative > 0 then
          return 0, 0, 0, 0, 0, 0, 0
        else
          return like, follow, forward, comment, collect, forward_inside
        end
      end
    """) \
  .enrich_attr_by_lua(
    import_item_attr=["like", "follow", "forward", "comment", "collect", 
          "unfollow", "unlike", "cancel_collect", "feedback_negative",
          "forward_inside", "time_weight", 'down_load', 
          "effective_view", "long_view"],
    function_for_item="calculate",
    export_item_attr=["interact_weight", "interact_label"],
    lua_script="""
      function calculate()
        interact_label = 0
        if (like > 0 or follow > 0 or forward > 0  or comment > 0 or collect > 0 or forward_inside > 0 or down_load > 0) and (effective_view > 0 or long_view > 0) then
          interact_label = 1
        end
        interactive_weight = 1.0
        if feedback_negative > 0 then
          interact_label = 0
          interactive_weight = 100.0
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
          if down_load > 0 then
            interactive_weight = interactive_weight + 1.0
          end
        end
        return interactive_weight, interact_label
      end
    """) \
  .enrich_attr_by_lua(
    import_common_attr=["sample_rate"],
    import_item_attr=["interact_label"],
    function_for_item="sample_process",
    export_item_attr=["sample_filter_flag"],
    lua_script="""
      function sample_process()
        local sample_filter_flag = 0
        if interact_label == 0 then
          if math.random() > sample_rate then
              sample_filter_flag = 1
          end
        end
        return sample_filter_flag
      end
    """
  ) \
  .filter_by_attr(attr_name="sample_filter_flag", remove_if="==", compare_to=1, name="data_sampler_filter") \
  .perflog_attr_value(
    check_point="default.training",
    item_attrs=["interact_weight", "interact_label", "reset_count", "audit_count"],
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
    attrs=["interact_label", "interact_weight"] + pxtr_list,
    slots_attrs = ["common_slots", "item_slots", "kuiba_common_slots", "kuiba_item_slots"],
    signs_attrs = ["common_signs", "item_signs", "kuiba_common_signs", "kuiba_item_signs"],
    label_attr="click")


runner = OfflineRunner("mio_offline_runner")

pipelines = [read_joint_reco_log, send_to_mio_learner]

runner.add_leaf_flows(leaf_flows = pipelines)
runner.build(output_file=os.path.join(current_dir, "cofea_reader.json"))