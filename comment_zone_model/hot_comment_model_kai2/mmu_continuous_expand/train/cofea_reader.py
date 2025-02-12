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


common_attrs=["user_id", "device_id", "photo_id", "photo_author_id", "gender", "age_segment", "llsid", "client_request_info"]
item_attrs=["comment_id", "author_id", "like_cnt", "reply_cnt", "dislike_cnt", "realshow_cnt", "minute_diff",
            "showAction", "expandAction", "replyAction", "likeAction", "audienceAction", "reportAction",
            "comment_genre", "risk_insult_tag", "risk_negative_tag", "risk_inactive_tag ",
            "mmu_category_tag", "mmu_emotion_tag", "mmu_entity_list",
            "sample_weight",
            "predict_reply_score", "quality_v2_score", "predict_like_score",
            "new_sample_weight", "expandAction_first", "replyAction_first", "likeAction_first", "audienceAction_first", "reportAction_first",
            "expandAction_second", "replyAction_second", "likeAction_second", "audienceAction_second", "reportAction_second"]


read_data_valid = DataReaderFlow(name="read_data") \
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
              + [dict(name=f"{item_attr}_list_common_retrieve", path="sample.attr", sample_attr_name=item_attr) for
                 item_attr in item_attrs]
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
    .count_reco_result(
        save_count_to="retrieve_num"
    ) \
    .if_("retrieve_num <= 0") \
        .return_(0) \
    .end_() \
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
        export_item_attr=["sample_weight"],
        function_for_item="cal_sample_weight",
        lua_script="""
            function cal_sample_weight()
                local weight = showAction * 1.0
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
      import_item_attr=["showAction", "expandAction_first", "replyAction_first", "likeAction_first", "audienceAction_first", "reportAction_first"],
      export_item_attr=["new_sample_weight"],
      function_for_item="cal_new_sample_weight",
      lua_script="""
        function cal_new_sample_weight()
          local weight = (showAction or 1.0) * 1.0
          if (expandAction_first or 0) > 0 then
            weight = weight + 3.0
          end
          if (replyAction_first or 0) > 0 then
            weight = weight + 5.0
          end
          if (likeAction_first or 0) > 0 then
            weight = weight + 3.0
          end
          if (audienceAction_first or 0) > 0 then
            weight = weight + 3.0
          end
          if (reportAction_first or 0) > 0 then
            weight = weight + 8.0
          end
          return weight
        end
      """
    ) \
    .get_remote_embedding_lite(
        # mmu提供的content embedding
        kess_service="grpc_mmuCommentContentEmb",
        id_converter={"type_name": "mioEmbeddingIdConverter"},
        query_source_type="item_attr",
        input_attr_name="comment_id",
        output_attr_name="mmu_hetu_content_emb",
        timeout_ms=50,
        slot=101,
        size=128,
        shard_num=4,
        client_side_shard=True,
    ) \
    .get_remote_embedding_lite(
        # mmu提供的content embedding
        kess_service="grpc_mmuCommentContentEmb",
        id_converter={"type_name": "mioEmbeddingIdConverter"},
        query_source_type="item_attr",
        input_attr_name="comment_id",
        output_attr_name="mmu_clip_content_emb",
        timeout_ms=50,
        slot=102,
        size=256,
        shard_num=4,
        client_side_shard=True,
    ) \
    .get_remote_embedding_lite(
        # mmu提供的content embedding
        kess_service="grpc_mmuCommentContentEmb",
        id_converter={"type_name": "mioEmbeddingIdConverter"},
        query_source_type="item_attr",
        input_attr_name="comment_id",
        output_attr_name="mmu_bert_content_emb",
        timeout_ms=50,
        slot=103,
        size=256,
        shard_num=4,
        client_side_shard=True,
    ) \
    .log_debug_info(
        for_debug_request_only = False,
        respect_sample_logging = True,
        common_attrs = common_attrs + ["sample_minute_diff"],
        item_attrs = item_attrs + ["ltr", "rtr"]
    )


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


extract_fea = DataReaderFlow(name="extract_fea") \
    .enrich_attr_by_lua(
        import_item_attr=["predict_reply_score", "quality_v2_score", "predict_like_score"],
        function_for_item="fre_group",
        export_item_attr=["fre_grouped_predict_reply_score", "fre_grouped_quality_v2_score", "fre_grouped_predict_like_score"],
        lua_script_file="./lua_scripts/fre_dis_100.lua"
    ) \
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

            **discreate_config("ltr", 206, [0.001, 0, 1000, 1, -1]),
            **discreate_config("rtr", 207, [0.001, 0, 1000, 1, -1]),

            **id_config("showAction", 208),
            **discreate_config("dislike_cnt", 209, [3, 0, 100000, 1, 0]),

            # 等频MMU提供的3个分数
            # **id_config("fre_grouped_predict_reply_score", 407),
            **id_config("fre_grouped_quality_v2_score", 408),
            # **id_config("fre_grouped_predict_like_score", 409),
        },
        is_common_attr=False,
        slots_output="comment_item_slots",
        parameters_output="comment_item_signs",
    )
    

send_mio = DataReaderFlow(name="send_mio") \
    .count_reco_result(save_count_to="send_sample_cnt") \
    .enrich_attr_by_lua(
        import_item_attr=["expandAction", "likeAction", "replyAction",
                          "expandAction_first", "likeAction_first", "likeAction_second", 
                          "replyAction_first", "replyAction_second"],
        export_item_attr=["expandAction_v", "likeAction_v", "replyAction_v",
                          "expandAction_first_v", "likeAction_first_v", "likeAction_second_v", 
                          "replyAction_first_v", "replyAction_second_v"],
        function_for_item="trans",
        lua_script="""
            function trans() 
                return expandAction * 1.0, likeAction * 1.0, replyAction * 1.0, expandAction_first * 1.0, likeAction_first * 1.0, likeAction_second * 1.0,replyAction_first * 1.0, replyAction_second * 1.0
            end
        """
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
            end
        """
    ) \
    .enrich_attr_by_lua(
        import_common_attr=["photo_id", "age_segment", "gender"],
        function_for_common="get_photo_hash",
        export_common_attr=["photo_hash"],
        lua_script="""
            function get_photo_hash()
                local photo_hash_str = tostring(photo_id) .. tostring(age_segment) .. tostring(gender)
                return util.CityHash64(photo_hash_str)
            end
        """
    ) \
    .perflog_attr_value(
        check_point="send.mio.before",
        item_attrs=["fre_grouped_ltr", "fre_grouped_rtr", "fre_grouped_like_cnt", "fre_grouped_reply_cnt", "field_emb", "like_reply_tr",
                    "mmu_hetu_content_emb", "mmu_clip_content_emb", "mmu_bert_content_emb",
                    "predict_reply_score", "quality_v2_score", "predict_like_score",
                    "fre_grouped_predict_reply_score", "fre_grouped_quality_v2_score", "fre_grouped_predict_like_score"
                    ],
        common_attrs=["send_sample_cnt", "user_hash", "photo_hash"]
    ) \
    .send_to_mio_learner(
        # 在这里就会丢失photo hash的int64精度，mio不支持int64
        attrs=["sample_weight",
               "expandAction", "likeAction", "replyAction",
               "expandAction_v", "likeAction_v", "replyAction_v",
               "mmu_hetu_content_emb", "mmu_clip_content_emb", "mmu_bert_content_emb",
               "new_sample_weight", 
               "expandAction_first_v", "likeAction_first_v", "likeAction_second_v",
               "replyAction_first_v", "replyAction_second_v"],
        slots_attrs=["comment_common_slots", "comment_item_slots"],
        signs_attrs=["comment_common_signs", "comment_item_signs"],
        lineid_attr="user_id",
        time_ms_attr="time_ms",
        label_attr="likeAction",
        user_hash_attr="user_hash"
    )

pipelines = [read_data_valid, extract_fea, send_mio]
runner = OfflineRunner("comment_profile")
runner.IGNORE_UNUSED_ATTR=['llsid']
runner.add_leaf_flows(leaf_flows=pipelines)
runner.build(output_file=os.path.join(current_dir, "cofea_reader.json"))
