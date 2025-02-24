import os
import sys
import argparse
import itertools
import operator
import yaml
import time
import datetime

current_dir = os.path.dirname(__file__)
#sys.path.append(os.path.join(current_dir, '../../../ks/common_reco/leaf/tools/pypi/'))

from dragonfly.common_leaf_dsl import LeafFlow, OfflineRunner
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.gsu.gsu_api_mixin import GsuApiMixin
from dragonfly.ext.embedding.embedding_api_mixin import EmbeddingApiMixin
from dragonfly.ext.livestream.livestream_api_mixin import LivestreamApiMixin

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

def parse_args():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--run", default=False, action="store_true")
  return parser.parse_args()

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
class DataReaderFlow(LeafFlow, MioApiMixin, KuibaApiMixin, OfflineApiMixin, GsuApiMixin, EmbeddingApiMixin, LivestreamApiMixin):
  def clean_all(self, reason, **kwargs):
    return self.limit(0, name="clean_all_for_" + reason, **kwargs)


uid_emb = [38, 34]
author_emb = [128, 519, 1142]
pid_emb = [26]
pid_stat = [185, 685, 686, 141]
pid_gate = [786, 787]
pid_xtr = [1101, 1102, 1103, 1104, 1105, 1107, 1108, 1109, 1110, 1111, 1112]

labels = ["short_play_value", "long_play_value", "effective_play_value", "play_time_min",
          "like_value", "follow_value", "forward_value","comment_value","negative_action_value", "profile_enter_value", "profile_stay_time_min", "profile_stay_time"]


all_slots = uid_emb + author_emb + pid_emb + pid_stat + pid_gate + pid_xtr

import time
import datetime

value_dict = {
  "uid": "int",
  "did": "string",
  "aid": "int",
  "pid": "int",
  "gender": "int",
  "time_ms": "int",
  "upload_time": "int",
  "photo_tag": "int",
  "hetu_level_one_tag": "int_list",
  "hetu_level_two_tag": "int_list",
  "duration_ms": "int",
  "play_time_ms": "int",
  "click_count": "int",
  "like_count": "int",
  "follow_list": "int_list",
  "is_follow": "int",
  "is_like": "int",
  "is_comment": "int",
  "is_forward": "int",
  "is_collect": "int",
  "is_profile": "int",
  "is_search": "int",
  "is_hate": "int",
  "long_view": "int",
  "like_list": "int_list",
  "click_list": "int_list",
  "author_fans_count": "int",
  "author_gender": "int",
  "author_upload_count": "int", 
  "author_healthiness": "double", 
  "author_dnn_cluster_id": "int", 
  "author_category_type": "int", 
  "author_user_good_count": "int",
  "pctr": "double",
  "pltr": "double",
  "pftr": "double",
  "pwtr": "double",
  "plvtr": "double",
  "pvtr": "double",
  "pptr": "double",
  "emp_ctr": "double",
  "emp_ltr": "double",
  "emp_wtr": "double",
  "emp_ftr": "double",
  "emp_lvtr": "double",
  "mc_pctr": "double",
  "mc_pltr": "double",
  "mc_pwtr": "double",
  "mc_pftr": "double"
}

common_attr_set = set(["uid", "did", "gender", "click_list", "like_list", "follow_list"])

sevenDayAgo = (datetime.datetime.now() - datetime.timedelta(days = 9))
# 转换为其他字符串格式
seven_time_ago = sevenDayAgo.strftime("%Y%m%d")

hdfs_path = "viewfs://hadoop-lt-cluster/home/reco_algorithm/dw/reco_algorithm_dev.db/reco_slide_vv_base_fea_table_xxj/p_date=" + seven_time_ago  + "/*"    

fetch_message = DataReaderFlow(name="fetch_message") \
  .fetch_message(
    group_id="reco_emb_sim_model",
    hdfs_path=hdfs_path,
    hdfs_format="raw_text",
    output_attr="csv_str",
  ) \
  .log_debug_info(
    common_attrs = ["csv_str"],
    for_debug_request_only=True,
    respect_sample_logging=True,
  ) \

read_joint_reco_log = DataReaderFlow(name="read_joint_reco_log") \
  .convert_csv_to_tf_sequence_example(
    from_extra_var="csv_str",
    common_attrs=[
        dict(column_index=i, column_name=v[0], type=v[1]) for (i, v) in enumerate(value_dict.items()) if v[0] in common_attr_set
    ],
    item_attrs=[
        dict(column_index=i, column_name=v[0], type=v[1]) for (i, v) in enumerate(value_dict.items()) if v[0] not in common_attr_set
    ],
    column_separator="|",
    item_separator=",",
    list_separator=" ",
    save_result_to="tf_sequence_example"
  ) \
  .retrieve_from_tf_sequence_example(
    from_extra_var="tf_sequence_example",
    item_key_attr="pid",
    user_id_attr="uid",
    device_id_attr="did",
    time_ms_attr="time_ms",
    reason=111,
  ) \
  .pack_item_attr(
    item_source={
      "reco_results":True,
    },
    mappings = [{
      "from_item_attr":"time_ms",
      "to_common_attr":"time_ms_common",
      "aggregator":"copy",
    }],
  ) \
  .count_reco_result(
    save_count_to="retrieve_num"
  ) \
  .if_("retrieve_num <= 0") \
    .return_(0) \
  .end_() \
  .get_kconf_params(
    kconf_configs=[{
      "kconf_key": "cc.knowledgeGraph.UpLongTermAttributionModelPipeline",
      "value_type": "json",
      "json_path": "colossus_time_day_gap_window",
      "export_common_attr": "time_gap_window"
    }]
  ) \
  .gsu_common_colossusv2_enricher(kconf="colossus.kconf_client.video_item",
        # 按需填写自己需要的 field，可以节省网络带宽和 CPU 资源
        item_fields=dict(photo_id="photo_id_list",
                          timestamp="timestamp_list",
                          play_time="play_time_list",
                          profile_stay_time = "profile_stay_time_list",
                          duration="duration_list",
                          label="label_list")) \
  .set_attr_value(
    common_attrs=[{
      "name":"vv_pid_timestamp",
      "type":"int_list",
      "value": [0]
      },{
      "name":"vv_pid_play_time",
      "type":"int_list",
      "value": [0]
      },{
      "name":"vv_pid_label",
      "type":"int_list",
      "value": [0]
      },{
      "name":"vv_pid_profile_stay_time",
      "type":"int_list",
      "value": [0]
      },{
      "name":"vv_pid_duration",
      "type":"int_list",
      "value": [0]
      }
    ]
  ) \
  .pack_common_attr(
    input_common_attrs = ["vv_pid_timestamp", "timestamp_list"],
    output_common_attr="timestamp_list_total",
  ) \
  .pack_common_attr(
    input_common_attrs = ["vv_pid_play_time", "play_time_list"],
    output_common_attr="play_time_list_total",
  ) \
  .pack_common_attr(
    input_common_attrs = ["vv_pid_label", "label_list"],
    output_common_attr="label_list_total",
  ) \
  .pack_common_attr(
    input_common_attrs = ["vv_pid_profile_stay_time", "profile_stay_time_list"],
    output_common_attr="profile_stay_time_list_total",
  ) \
  .pack_common_attr(
    input_common_attrs = ["vv_pid_duration", "duration_list"],
    output_common_attr="duration_list_total",
  ) \
  .retrieve_by_common_attr(
    attr = "photo_id_list",
    reason=222,
  ) \
  .dispatch_common_attr(
    dispatch_config = [
        {
          "from_common_attr" : "play_time_list_total",
          "to_item_attr" : "play_time_colossus"
        },{
          "from_common_attr" : "label_list_total",
          "to_item_attr" : "label_colossus"
        },{
          "from_common_attr" : "profile_stay_time_list_total",
          "to_item_attr" : "profile_stay_time_colossus"
        },{
          "from_common_attr" : "duration_list_total",
          "to_item_attr" : "duration_colossus"
        },{
          "from_common_attr" : "timestamp_list_total",
          "to_item_attr" : "timestamp_colossus"
        }
      ]
  ) \
  .count_reco_result(
    save_count_to="colossus_item_num"
  ) \
  .filter_by_rule(
    rule = {
      "join": "and",
      "filters": [{
        "attr_name": "timestamp_colossus",
        "remove_if": ">",
        "compare_to": "{{return time_ms_common//1000+(time_gap_window+1)*86400}}"
      }, {
        "attr_name": "reason",
        "check_reason": True,
        "remove_if": "==",
        "compare_to": 222
      }]
    }
  ) \
  .filter_by_rule(
    rule = {
      "join": "and",
      "filters": [{
        "attr_name": "timestamp_colossus",
        "remove_if": "<=",
        "compare_to": "{{return time_ms_common//1000 + 86400}}"
      }, {
        "attr_name": "reason",
        "check_reason": True,
        "remove_if": "==",
        "compare_to": 222
      }]
    }
  ) \
  .count_reco_result(
    save_count_to="colossus_item_num_filter_by_timestamp"
  ) \
  .fetch_remote_embedding(
        protocol=1,
        colossusdb_embd_model_name="fr_slide_multi_task_with_global_embd",
        colossusdb_embd_table_name="fr_slide_multi_task_with_global_embd_1",
        id_converter={"type_name": "mioEmbeddingIdConverter"},
        client_side_shard=True,
        is_raw_data=True,
        raw_data_type="scale_int8",
        shard_num=4,
        timeout_ms=100,
        slot=26,
        size=64,
        query_source_type="item_key",
        output_attr_name="photo_id_emb",
  ) \
  .pack_item_attr(
    item_source={
      "reco_results":True,
    },
    mappings = [{
      "from_item_attr":"photo_id_emb",
      "to_common_attr":"vv_pid_emb",
      "aggregator":"copy",
    },{
      "from_item_attr":"pid",
      "to_common_attr":"vv_pid",
      "aggregator":"copy",
    }],
  ) \
  .get_kconf_params(
    kconf_configs=[{
      "kconf_key": "cc.knowledgeGraph.UpLongTermAttributionModelPipeline",
      "value_type": "json",
      "json_path": "emb_similarity_threshold",
      "export_common_attr": "emb_similarity_threshold"
    }]
  ) \
  .livestream_enrich_embedding_similarity(
    seed_embedding_attr="vv_pid_emb",
    item_embedding_attr="photo_id_emb",
    similarity_attr="similarity",
    norm_similarity_attr="norm_similarity"
  ) \
  .filter_by_rule(
    rule = {  
      "attr_name": "norm_similarity",
      "remove_if": "<",
      "compare_to": "{{emb_similarity_threshold}}",
      "remove_if_attr_missing": True
    }
  ) \
  .count_reco_result(
    save_count_to="sim_cut_num",
  ) \
  .enrich_attr_by_lua(
    import_item_attr=["label_colossus", "play_time_colossus", "duration_colossus"],
    export_item_attr=["like_colossus", "follow_colossus", "forward_colossus", "comment_colossus", "profile_enter_colossus", "long_play_colossus", "effective_play_colossus", "short_play_colossus", "negative_action_colossus"],
    function_for_item="parser",
    lua_script="""
      function isLongPlay(duration_s, play_time_s)
        if duration_s < 14.0 then
          return play_time_s >= (0.881369 * duration_s + 5.184729)
        elseif duration_s >= 14.0 and duration_s < 53.0 then
          return play_time_s >= (0.918007 * duration_s + 3.847164)
        elseif duration_s >= 53.0 and duration_s < 95.0 then
          return play_time_s >= (0.732063 * duration_s + 13.416789)
        elseif duration_s >= 95.0 and duration_s < 170.0 then
          return play_time_s >= 0.226594 * duration_s + 62.817905
        else
          return play_time_s >= (-0.180531 * duration_s + 131.062293) and play_time_s >= 36.0
        end
      end
      function isEffectivePlay(duration_s, play_time_s)
        if duration_s < 7.0 then
          return play_time_s > 7.0
        elseif duration_s >= 7.0 and duration_s < 14.0 then 
          return play_time_s > 0.636788*duration_s + 2.417629
        elseif duration_s >= 14.0 and duration_s < 93.0 then
          return play_time_s > 0.104396*duration_s + 8.957299
        elseif duration_s >= 93.0 and duration_s < 155.0 then
          return play_time_s > -0.018168*duration_s + 20.215178
        elseif duration_s >= 155.0 and duration_s < 220.0 then
          return play_time_s > -0.072866*duration_s + 28.352154
        elseif duration_s >= 220.0 then
          return play_time_s > (-0.038242 * duration_s + 21.259100) and play_time_s >= 7.0
        else
          return 0
        end
      end
      function isShortPlay(duration_s, play_time_s)
        if duration_s < 3.0 then
          return play_time_s < duration_s
        elseif duration_s >= 3.0 then
          return play_time_s < 3.0
        else
          return 0
        end
      end
      function parser()
        like_count =  ((label_colossus & (1 << 0)) > 0) and 1 or 0
        follow_count = ((label_colossus & (1 << 1)) > 0) and 1 or 0
        forward_count = ((label_colossus & (1 << 2)) > 0) and 1 or 0
        comment_count = ((label_colossus & (1 << 4)) > 0) and 1 or 0
        profile_enter_count = ((label_colossus & (1 << 6)) > 0) and 1 or 0
        long_play_count = isLongPlay(duration_colossus, play_time_colossus) and 1 or 0
        effective_play_count = isEffectivePlay(duration_colossus, play_time_colossus) and 1 or 0
        short_play_count = isShortPlay(duration_colossus, play_time_colossus) and 1 or 0
        neg_action_count = ((label_colossus & (1 << 13)) > 0) and 1 or 0
        return like_count, follow_count, forward_count, comment_count, profile_enter_count, long_play_count, effective_play_count, short_play_count, neg_action_count
      end
    """,
  ) \
  .enrich_attr_by_lua(
    export_item_attr = ["check_if_pack"],
    function_for_item = "set_by_reason",
    lua_script="""
      function set_by_reason(seq, item_key, reason, score)
        local check_if_pack = 0
        if reason == 111 then
          check_if_pack = 0
        else
          check_if_pack = 1
        end
        return check_if_pack
      end
    """,
  ) \
  .pack_item_attr(
    item_source={
      "reco_results":False,
      "common_attr":["photo_id_list"]
    },
    mappings = [{
      "from_item_attr":"like_colossus",
      "to_common_attr":"like_value_common",
      "aggregator":"sum",
      "pack_if":"check_if_pack",
    },{
      "from_item_attr":"follow_colossus",
      "to_common_attr":"follow_value_common",
      "aggregator":"sum",
      "pack_if":"check_if_pack",
    },{
      "from_item_attr":"forward_colossus",
      "to_common_attr":"forward_value_common",
      "aggregator":"sum",
      "pack_if":"check_if_pack",
    },{
      "from_item_attr":"comment_colossus",
      "to_common_attr":"comment_value_common",
      "aggregator":"sum",
      "pack_if":"check_if_pack",
    },{
      "from_item_attr":"profile_enter_colossus",
      "to_common_attr":"profile_enter_value_common",
      "aggregator":"sum",
      "pack_if":"check_if_pack",
    },{
      "from_item_attr":"long_play_colossus",
      "to_common_attr":"long_play_value_common",
      "aggregator":"sum",
      "pack_if":"check_if_pack",
    },{
      "from_item_attr":"effective_play_colossus",
      "to_common_attr":"effective_play_value_common",
      "aggregator":"sum",
      "pack_if":"check_if_pack",
    },{
      "from_item_attr":"short_play_colossus",
      "to_common_attr":"short_play_value_common",
      "aggregator":"sum",
      "pack_if":"check_if_pack",
    },{
      "from_item_attr":"negative_action_colossus",
      "to_common_attr":"negative_action_value_common",
      "aggregator":"sum",
      "pack_if":"check_if_pack",
    },{
      "from_item_attr":"play_time_colossus",
      "to_common_attr":"play_time_value_common",
      "aggregator":"sum",
      "pack_if":"check_if_pack",
    },{
      "from_item_attr":"profile_stay_time_colossus",
      "to_common_attr":"profile_stay_time_value_common",
      "aggregator":"sum",
      "pack_if":"check_if_pack",
    }],
  ) \
  .log_debug_info(
      for_debug_request_only=True,
      item_attrs = [
        "photo_id_emb",
        "like_count",
        "follow_count",
        "forward_count",
        "comment_count",
        "profile_enter_count",
        "play_time",
        "profile_stay_time",
      ]
  ) \
  .limit("{{retrieve_num}}") \
  .set_attr_value(
    item_attrs=[
    {
      "name": "like_value",
      "type": "int",
      "value": 0
    },{
      "name": "follow_value",
      "type": "int",
      "value": 0
    },{
      "name": "forward_value",
      "type": "int",
      "value": 0
    },{
      "name": "comment_value",
      "type": "int",
      "value": 0
    },{
      "name": "profile_enter_value",
      "type": "int",
      "value": 0
    },{
      "name": "long_play_value",
      "type": "int",
      "value": 0
    },{
      "name": "effective_play_value",
      "type": "int",
      "value": 0
    },{
      "name": "short_play_value",
      "type": "int",
      "value": 0
    },{
      "name": "negative_action_value",
      "type": "int",
      "value": 0
    },{
      "name": "play_time",
      "type": "int",
      "value": 0
    },{
      "name": "profile_stay_time",
      "type": "int",
      "value": 0
    }]
  ) \
  .dispatch_common_attr(
    dispatch_config = [
    {
      "from_common_attr" : "like_value_common",
      "to_item_attr" : "like_value"
    },{
      "from_common_attr" : "follow_value_common",
      "to_item_attr" : "follow_value"
    },{
      "from_common_attr" : "forward_value_common",
      "to_item_attr" : "forward_value"
    },{
      "from_common_attr" : "comment_value_common",
      "to_item_attr" : "comment_value"
    },{
      "from_common_attr" : "profile_enter_value_common",
      "to_item_attr" : "profile_enter_value"
    },{
      "from_common_attr" : "long_play_value_common",
      "to_item_attr" : "long_play_value"
    },{
      "from_common_attr" : "effective_play_value_common",
      "to_item_attr" : "effective_play_value"
    },{
      "from_common_attr" : "short_play_value_common",
      "to_item_attr" : "short_play_value"
    },{
      "from_common_attr" : "negative_action_value_common",
      "to_item_attr" : "negative_action_value"
    },{
      "from_common_attr" : "play_time_value_common",
      "to_item_attr" : "play_time"
    },{
      "from_common_attr" : "profile_stay_time_value_common",
      "to_item_attr" : "profile_stay_time"
    }
  ]) \
  .build_protobuf(
    class_name="ks::reco::UserInfo",
    inputs=[
      { "common_attr": "uid", "path": "id"},#slot 38
      { "common_attr": "did", "path": "device_id"},#slot 34
      { "common_attr": "gender", "path": "gender"},#slot 184
    ],
    output_common_attr="user_info",
  ) \
  .build_protobuf(
    class_name="ks::reco::PhotoInfo",
    inputs=[
      {"item_attr": "pid", "path": "photo_id"},# slot 26
      {"item_attr": "aid", "path": "author.id"},#slot 128
      {"item_attr":"upload_time", "path": "upload_time"},# slot 110
      {"item_attr":"photo_tag", "path": "tag"},# slot 185
      {"item_attr":"hetu_level_one_tag", "path": "hetu_tag_level_info.hetu_level_one","append":True},# slot 685
      {"item_attr":"hetu_level_two_tag", "path": "hetu_tag_level_info.hetu_level_two","append":True},# slot 686
      {"item_attr":"duration_ms", "path": "duration_ms"},# slot 141
      {"item_attr":"click_count", "path": "click_count"}, # slot 786
      {"item_attr":"like_count", "path": "like_count"},# slot 787
      {"item_attr":"author_fans_count", "path": "author.fans_count"},# slot 1142
      {"item_attr":"author_gender", "path": "author.gender"},# slot 519
    ],
    output_item_attr="photo_info",
  ) \
  .build_protobuf(
    class_name="ks::reco::ContextInfo",
    inputs=[
      {"item_attr": "time_ms", "path": "time_ms"},# slot 110
      {"item_attr":"is_follow", "path": "follow"},
      {"item_attr":"is_like", "path": "like"},
      {"item_attr":"is_forward", "path": "forward"},
      {"item_attr":"is_profile", "path": "profile_enter"},
      {"item_attr":"is_comment", "path": "comment"},
      {"item_attr":"is_collect", "path": "collect"},
    ],
    output_item_attr="context_info",
  ) \
  .log_debug_info( 
      for_debug_request_only=True,
      common_attrs = [
          "photo_id_list",
          "play_time_list",
          "label_list"]) \

post_process = DataReaderFlow(name="post_process") \
  .enrich_attr_by_lua(
    import_item_attr=["play_time", "profile_stay_time"],
    function_for_item="calculate",
    export_item_attr=["play_time_min", "profile_stay_time_min"],
    lua_script=f"""
      function calculate()
        local play_time_min = play_time / 30
        local profile_stay_time_min = (profile_stay_time) / 10
        return play_time_min, profile_stay_time_min
      end
    """) \
  .enrich_attr_by_lua(
    import_item_attr=["play_time_min","short_play_value", "long_play_value", "effective_play_value", "profile_stay_time_min"],
    function_for_item="calculate",
    export_item_attr=["play_time_min","short_play_value", "long_play_value", "effective_play_value", "profile_stay_time_min"],
    lua_script=f"""
      function calculate()
        return math.min(play_time_min, 20), math.min(short_play_value, 20), math.min(long_play_value, 20), math.min(effective_play_value, 20) , math.min(profile_stay_time_min, 20) 
      end
    """) \
  .log_debug_info(
    item_attrs = ["mc_pctr", "mc_pltr", "mc_pwtr", "mc_pftr", "emp_lvtr", "play_time_min","short_play_value", "long_play_value", "effective_play_value",
          "like_value", "follow_value", "forward_value", "comment_value", "collect_value", "negative_action_value"],
    for_debug_request_only=True,
    respect_sample_logging=True,
  ) \
  .perflog_attr_value(
    check_point="default.training",
    common_attrs=["sim_cut_num", "colossus_item_num_filter_by_timestamp", "colossus_item_num", "time_ms_common"],
    item_attrs=labels,
  ) \
  .export_attr_to_kafka(
    kafka_topic="fr_slide_multi_task_with_global_embd_dump",
    common_attrs=["vv_pid", "sim_cut_num", "colossus_item_num_filter_by_timestamp", "colossus_item_num", "time_ms_common"],
  ) \
  .extract_with_ks_sign_feature(
    feature_list=load_feature_list_sign(os.path.join(current_dir, "./feature_list_sign.txt")),
    user_info_attr="user_info",
    photo_info_attr="photo_info",
    context_info_attr="context_info",
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
  )\
  .extract_kuiba_parameter(
    config={
      "click_pid_list": {"attrs": [{"mio_slot_key_type": 2201, "key_type": 2201, "attr": ["click_list"], **kuiba_list_converter_config_list_limit(30)}]},
      "like_pid_list": {"attrs": [{"mio_slot_key_type": 2202, "key_type": 2202, "attr": ["like_list"], **kuiba_list_converter_config_list_limit(30)}]},
      "follow_pid_list": {"attrs": [{"mio_slot_key_type": 2203, "key_type": 2203, "attr": ["follow_list"], **kuiba_list_converter_config_list_limit(30)}]},
    },
    is_common_attr=True,
    slots_output="common_kuiba_slots",
    parameters_output="common_kuiba_signs",
    slot_as_attr_name=False,
  ) \


send_to_mio_learner = DataReaderFlow(name = "send_to_mio_learner") \
  .enrich_attr_by_lua(
    function_for_item = "get_time_ms_now",
    export_item_attr = ["curr_time_ms"],
    lua_script = """
      function get_time_ms_now()
        local curr_time_us = util.GetTimestamp()
        local curr_time_ms = math.floor(curr_time_us / 1000)
        return curr_time_ms
      end
    """
  ) \
  .send_to_mio_learner(
    attrs=labels+["pctr", "pftr", "pltr", "plvtr", "pptr", "pvtr", "pwtr", "empirical_ctr", "empirical_ftr", "empirical_ltr", "empirical_wtr", "author_healthiness"],
    #slots=all_slots,
    slots_attrs = ["common_slots", "item_slots", "item_pxtr_slots", "common_kuiba_slots"],
    signs_attrs = ["common_signs", "item_signs", "item_pxtr_signs", "common_kuiba_signs"],
    label_attr="click",
    time_ms_attr="curr_time_ms",
    user_hash_attr="did",
    ) \

runner = OfflineRunner("mio_offline_runner")
runner.IGNORE_UNUSED_ATTR=["photo_id_emb_cp"] 
runner.CHECK_UNUSED_ATTR = False
pipelines = [fetch_message, read_joint_reco_log, post_process, send_to_mio_learner]

if args.run:
    runner.ENABLE_ATTR_CHECK = False
    run_pipelines = pipelines[:-1]
    runner.add_leaf_flows(leaf_flows = run_pipelines)
    exe = runner.executor()
    while not exe["MESSAGE_END"]:
        exe.reset()
        for pipeline in run_pipelines:
            exe.run(pipeline.name)

        for item in exe.items:
            print(item.item_key)
            for slot in mapping_input_slots + mapping_output_slots:
                print(f"{slot}: {item[slot]}")
else:
    runner.add_leaf_flows(leaf_flows = pipelines, name="last")
    runner.build(output_file=os.path.join(current_dir, "cofea_reader.json"))
