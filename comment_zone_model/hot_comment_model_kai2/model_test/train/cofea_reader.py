from dragonfly.ext.cofea.cofea_api_mixin import CofeaApiMixin
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin
from dragonfly.ext.kgnn.kgnn_api_mixin import KgnnApiMixin
from dragonfly.ext.kgnn.node_attr_schema import NodeAttrSchema
from dragonfly.common_leaf_dsl import LeafFlow, OfflineRunner
import os
import sys

current_dir = os.path.dirname(__file__)


class DataReaderFlow(LeafFlow, MioApiMixin, OfflineApiMixin, CofeaApiMixin, KuibaApiMixin, KgnnApiMixin):
    def clean_all(self, reason, **kwargs):
        return self.limit(0, name="clean_all_for_" + reason, **kwargs)


common_attrs=["user_id", "device_id", "photo_id", "photo_author_id", "gender", "age_segment", "llsid"]
item_attrs=["comment_id", "author_id", "like_cnt", "reply_cnt", "minute_diff", "realshow_cnt", "dislike_cnt",
            "showAction", "expandAction", "replyAction", "likeAction", "audienceAction", "reportAction",
            "comment_genre", "risk_insult_tag", "risk_negative_tag", "risk_inactive_tag",
            "mmu_category_tag", "mmu_emotion_tag", "sample_weight",
            "mmu_entity_list_v", "comment_content_segs_v"]



value_dict = {
    
  "comment_id": "int",
  "author_id": "int",
  "like_cnt": "int",
  "reply_cnt": "int",
  "dislike_cnt": "int",
  "realshow_cnt": "int",
  "minute_diff": "float",
  "ltr": "float",
  "rtr": "float",
  "showaction": "int",
  "replyaction": "int",
  "likeaction": "int",
  "expandaction": "int",
  "audienceaction": "int",
  "reportaction": "int",
  "comment_content_segs": "string_list",
  "age_segment": "int",
  "gender": "int",
  "device_id": "string",
  "photo_id": "int",
  "photo_author_id": "int",
  "comment_genre": "int",
  "risk_insult_tag": "int",
  "risk_negative_tag": "int",
  "risk_inactive_tag": "int",
  "mmu_category_tag": "int",
  "mmu_emotion_tag": "int",
  "mmu_entity_list": "int_list",
  "user_id": "int",
  "predict_reply_score": "float",
  "quality_v2_score": "float",
  "predict_like_score": "float",
  "recall_type": "int",
  "expandaction_first": "int",
  "expandaction_second": "int",
  "likeaction_first": "int",
  "likeaction_second": "int",
  "replyaction_first": "int",
  "replyaction_second": "int",
  "reportaction_first": "int",
  "reportaction_second": "int",
  "audienceaction_first": "int",
  "audienceaction_second": "int",
  "like_rank_index": "int",
  "reply_rank_index": "int",
  "realshow_rank_index": "int",
  "dislike_rank_index": "int",
}

common_attr_set = set(["photo_author_id", "user_id", "photo_id", "age_segment", "gender", "device_id"])


read_data = DataReaderFlow(name="read_data") \
  .fetch_message(
    group_id="reco_forward_open_log",
    hdfs_path="viewfs:///home/reco_algorithm/dw/reco_algorithm.db/comment_model_data_zt08_dev1/datatype=0509_18_19_2hTrain_rank",
    hdfs_format="raw_text",
    output_attr="csv_sample_data",
  ) \
  .log_debug_info(
    common_attrs = ["csv_sample_data"],
    for_debug_request_only=False,
    respect_sample_logging=True,
  ) \
  .convert_csv_to_tf_sequence_example(
    from_extra_var="csv_sample_data",
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
    item_key_attr="comment_id",
    reason=111,
  ) \
  .perflog_attr_value(
      check_point="comment.test",
      common_attrs=["gender", "age_segment"],
      item_attrs=["like_cnt", "reply_cnt", "minute_diff", "realshow_cnt", "dislike_cnt",
          "showAction", "expandAction", "replyAction", "likeAction", "audienceAction", "reportAction",
          "comment_genre", "risk_insult_tag", "risk_negative_tag", "risk_inactive_tag",
          "mmu_category_tag", "mmu_emotion_tag", "like_rank_index", "reply_rank_index",
          "realshow_rank_index", "dislike_rank_index"]
  )

'''
read_data = DataReaderFlow(name="read_data") \
    .fetch_message(
        group_id="hot_comment_xtr",
        kafka_topic="reco_hot_comment_join_listwise_sample",
        output_attr="raw_sample_package_str",
    ) \
    .parse_protobuf_from_string(
        input_attr="raw_sample_package_str",
        output_attr="raw_sample_package",
        class_name="kuiba::RawSamplePackage",
    ) \
    .enrich_with_protobuf(
        from_extra_var="raw_sample_package",
        attrs=[dict(name=common_attr, path="common_attr", sample_attr_name=common_attr) for common_attr in common_attrs]
              + [dict(name=f"{item_attr}_list_common_retrieve", path="sample.attr", sample_attr_name=item_attr) for item_attr in item_attrs]
              + [dict(name="request_time", path="timestamp")]

    ) \
    .if_("#(comment_id_list_common_retrieve or {}) == 0") \
        .return_(0) \
    .end_() \
    .enrich_attr_by_lua(
        import_common_attr=["request_time"],
        export_common_attr=["time_ms", "sample_minute_diff"],
        function_for_common="cal",
        lua_script="""
            function cal()
                local time_ms = request_time // 1000
                local sample_minute_diff = (util.GetTimestamp() - request_time) / 1000 / 1000 / 60.0
                return time_ms, sample_minute_diff
            end
        """
    ) \
    .retrieve_by_common_attr(
        attr="comment_id_list_common_retrieve",
        reason=999
    ) \
    .dispatch_common_attr(
        dispatch_config=[
            dict(from_common_attr=f"{attr}_list_common_retrieve", to_item_attr=attr) for attr in item_attrs
        ]
    ) \
    .count_reco_result(save_count_to="retrieve_item_cnt") \
    .filter_by_attr(
        attr_name="showAction",
        remove_if="<=",
        compare_to=0,
        remove_if_attr_missing=True,
    ) \
    .count_reco_result(save_count_to="show_item_cnt") \
    .set_attr_default_value(
        item_attrs=[
          {
            "name": "expandAction",
            "type": "int",
            "value": 0
          },
          {
            "name": "replyAction",
            "type": "int",
            "value": 0
          },
          {
            "name": "likeAction",
            "type": "int",
            "value": 0
          },
          {
            "name": "audienceAction",
            "type": "int",
            "value": 0
          },
          {
            "name": "reportAction",
            "type": "int",
            "value": 0
          },
          {
            "name": "showAction",
            "type": "int",
            "value": 0
          }
        ]
    ) \
    .perflog_attr_value(
        check_point="comment.listwise",
        common_attrs=["retrieve_item_cnt", "show_item_cnt", "sample_minute_diff"],
        item_attrs=["like_cnt", "reply_cnt", "minute_diff", "realshow_cnt", "dislike_cnt",
            "showAction", "expandAction", "replyAction", "likeAction", "audienceAction", "reportAction",
            "comment_genre", "risk_insult_tag", "risk_negative_tag", "risk_inactive_tag",
            "mmu_category_tag", "mmu_emotion_tag"]
    ) \
    .if_("show_item_cnt <= 0") \
        .return_(0) \
    .end_() \
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
    .enrich_attr_by_lua(
      import_item_attr=["showAction", "expandAction", "replyAction", "likeAction", "audienceAction", "reportAction"],
      export_item_attr=["calced_sample_weight"],
      function_for_item="cal_sample_weight",
      lua_script="""
        function cal_sample_weight()
          local weight = (showAction or 1.0) * 1.0
          if (expandAction or 0) > 0 then
            weight = weight + 3.0
          end
          if (replyAction or 0) > 0 then
            weight = weight + 5.0
          end
          if (likeAction or 0) > 0 then
            weight = weight + 3.0
          end
          if (audienceAction or 0) > 0 then
            weight = weight + 3.0
          end
          if (reportAction or 0) > 0 then
            weight = weight + 8.0
          end
          return weight
        end
      """
    ) \
    .enrich_attr_by_lua(
        import_item_attr=["calced_sample_weight", "sample_weight"],
        export_item_attr=["legal_calced_sample_weight", "legal_sample_weight", "same_weight", "diff_weight"],
        function_for_item="is_legal",
        lua_script="""
            function is_legal() 
                local legal_calced_sample_weight = 0
                local legal_sample_weight = 0
                local same_weight = 0
                if calced_sample_weight > 0 then
                  legal_calced_sample_weight = 1
                end
                if sample_weight > 0 then
                  legal_sample_weight = 1
                end
                if math.abs(sample_weight -  calced_sample_weight) < 0.00001 then
                  same_weight = 1
                end
                return legal_calced_sample_weight, legal_sample_weight, same_weight, math.abs(sample_weight -  calced_sample_weight)
            end
        """
    ) \
    .log_debug_info(
        for_debug_request_only=False,
        respect_sample_logging=True,
        item_attrs=["calced_sample_weight", "sample_weight", "same_weight", "diff_weight"] + item_attrs,
        common_attrs=common_attrs,
        select_item={
            "attr_name": "diff_weight",
            "compare_to": 0.1,
            "select_if": ">",
            "select_if_attr_missing": False
        }
    ) \
    .log_debug_info(
        common_attrs=common_attrs,
        item_attrs=item_attrs,
        for_debug_request_only=False,
        respect_sample_logging=True
    )
'''

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
  attr_name: {"attrs": [{"mio_slot_key_type": mio_slot_key_type, "key_type": key_type, "attr": [attr_name], "converter": "id"}]},
}

kuiba_list_converter_config_list_limit = lambda limit_n:  {
  "converter": "list",
  "type":5,
  "converter_args": {
    "reversed": False,
    "enable_filter": False,
    "limit": limit_n,
  },
 }
list_config = lambda attr_name, mio_slot_key_type, key_type, limit: {
  attr_name: {"attrs": [{"mio_slot_key_type": mio_slot_key_type, "key_type": key_type, "attr": [attr_name], **kuiba_list_converter_config_list_limit(limit)}]},
}

fetch_profile = DataReaderFlow(name="fetch_profile") \
    .fetch_kgnn_neighbors(
      id_from_common_attr="user_id",
    #   save_weight_to="interact_comment_weights",  # like + reply
      save_neighbors_to="interact_comment_ids",
      edge_attr_schema=NodeAttrSchema(1, 0).add_int64_list_attr("interact_comment_mmu_categories", 1),
      kess_service="grpc_kgnn_user_interact_comment_info-U2I",
      relation_name='U2I',
      shard_num=4,
      sample_num=10,
      timeout_ms=50,
      sample_type="topn",
      padding_type="zero",
    ) \
    .fetch_kgnn_neighbors(
      id_from_common_attr="user_id",
    #   save_weight_to="write_comment_weights",  # like + reply
      save_neighbors_to="write_comment_ids",
      edge_attr_schema=NodeAttrSchema(1, 0).add_int64_list_attr("write_comment_mmu_categories", 1),
      kess_service="grpc_kgnn_user_write_comment_info-U2I",
      relation_name='U2I',
      shard_num=4,
      sample_num=10,
      timeout_ms=50,
      sample_type="topn",
      padding_type="zero",
    ) \
    .fetch_kgnn_neighbors(
        id_from_item_attr="author_id",
        save_neighbors_to="author_write_comment_ids",
        edge_attr_schema=NodeAttrSchema(1, 0).add_int64_list_attr("author_write_comment_mmu_categories", 1),
        kess_service="grpc_kgnn_user_write_comment_info-U2I",
        relation_name='U2I',
        shard_num=4,
        sample_num=10,
        timeout_ms=50,
        sample_type="topn",
        padding_type="zero",
        sample_without_replacement=True
    )


extract_fea = DataReaderFlow(name="extract_fea") \
    .extract_kuiba_parameter(
      config={
          **id_config("gender", 101),
          **id_config("age_segment", 102),
          
          # new_feature
          **id_config("photo_id", 103),
          **id_config_slot("photo_author_id", 104, 202),
          **id_config_slot("user_id", 105, 202),
          **id_config("device_id", 106),
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

          # new_feature
          **discreate_config("ltr_copy", 210, [0.001, 0, 1000, 137, 0]),
          **discreate_config("rtr_copy", 211, [0.001, 0, 1000, 137, 0]),
          **list_config("comment_content_segs", 212, 212, 16),

          # risk and mmu comment content label
          **id_config("comment_genre", 213),
          **id_config("risk_insult_tag", 214),
          **id_config("risk_negative_tag", 215),
          **id_config("risk_inactive_tag", 216),
          **id_config("mmu_category_tag", 217),
          **id_config("mmu_emotion_tag", 218),
          **list_config("mmu_entity_list", 219, 219, 10),

          # rank_index
          **id_config("like_rank_index", 220),
          **id_config("reply_rank_index", 221),
          **id_config("realshow_rank_index", 222),
          **id_config("dislike_rank_index", 223),
      },
      is_common_attr=False,
      slots_output="comment_item_slots",
      parameters_output="comment_item_signs",
    )
    

send_mio = DataReaderFlow(name="send_mio") \
    .count_reco_result(save_count_to="send_sample_cnt") \
    .enrich_attr_by_lua(
      import_item_attr=["expandaction_first", "likeaction_first", "replyaction_first"],
      export_item_attr=["expandAction_v", "likeAction_v", "replyAction_v"],
      function_for_item="trans",
      lua_script="""
          function trans() 
            return expandaction_first * 1.0, likeaction_first * 1.0, replyaction_first * 1.0
          end
      """
    ) \
    .perflog_attr_value(
        check_point="comment.mio.record",
        item_attrs=["expandAction_v", "likeAction_v", "replyAction_v"],
        common_attrs=["send_sample_cnt"]
    ) \
    .enrich_attr_by_lua(
      import_common_attr=["user_id", "device_id"],
      function_for_common="get_hash",
      export_common_attr=["user_hash"],
      lua_script=f"""
        function get_hash()
          if device_id == nil or device_id == "" then
            return tonumber(user_id or '0')
          end
          return util.CityHash64(device_id)
        end"""
    ) \
    .send_to_mio_learner(
        attrs=["expandAction_v", "likeAction_v", "replyAction_v"],
        slots_attrs=["comment_common_slots", "comment_item_slots"],
        signs_attrs=["comment_common_signs", "comment_item_signs"],
        lineid_attr="user_id",
        time_ms_attr="time_ms",
        label_attr="likeAction_v",
        user_hash_attr="user_hash"
    )

pipelines = [read_data, extract_fea, send_mio]
runner = OfflineRunner("comment_profile")
runner.IGNORE_UNUSED_ATTR=['llsid']
runner.add_leaf_flows(leaf_flows=pipelines)
runner.build(output_file=os.path.join(current_dir, "cofea_reader.json"))
