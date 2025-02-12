from dragonfly.ext.cofea.cofea_api_mixin import CofeaApiMixin
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin
from dragonfly.ext.kgnn.kgnn_api_mixin import KgnnApiMixin
from dragonfly.common_leaf_dsl import LeafFlow, OfflineRunner
from dragonfly.ext.kgnn.node_attr_schema import NodeAttrSchema

import os
import sys

current_dir = os.path.dirname(__file__)


class DataReaderFlow(LeafFlow, MioApiMixin, OfflineApiMixin, CofeaApiMixin, KuibaApiMixin, KgnnApiMixin):
    def clean_all(self, reason, **kwargs):
        return self.limit(0, name="clean_all_for_" + reason, **kwargs)

SEQ_LEN = 15

# 要从 kafka 中读取的特征
common_attrs=["user_id", "device_id", "photo_id", "photo_author_id", "gender", "age_segment", "llsid",
              "mod", "client_request_info", "page_type_str", "is_political", "product_name", "photo_upload_time",
              "city_name", 
              "photo_real_show", "photo_click_count", "photo_like_count", "photo_follow_count", "photo_forward_count",
              "photo_long_play_count", "photo_short_play_count", "photo_comment_count", "photo_view_length_sum",
              "photo_effective_play_count", "photo_comment_stay_time_sum_ms", "photo_recommend_count",
              "photo_upload_type", "photo_duration_ms", "photo_author_low_score", "photo_author_high_score",
              "photo_hetuV1_level1_tag_id"
]

labels=["expandAction_first", "replyAction_first", "likeAction_first", "audienceAction_first", "reportAction_first",
        "expandAction_second", "replyAction_second", "likeAction_second", "audienceAction_second",
        "copyAction", "copyAction_first", "copyAction_second", "shareAction", "shareAction_first", "shareAction_second",
]

item_attrs=["comment_id", "author_id", "like_cnt", "reply_cnt", "minute_diff", "realshow_cnt", "dislike_cnt",
            "showAction", "expandAction", "replyAction", "likeAction", "audienceAction", "reportAction",
            "comment_genre", "risk_insult_tag", "risk_negative_tag", "risk_inactive_tag",
            "mmu_category_tag", "mmu_emotion_tag", "mmu_entity_list_v", "sample_weight",
            "comment_content_segs_v", "new_sample_weight", "expandAction_first", "replyAction_first", "likeAction_first", "audienceAction_first", "reportAction_first",
            "expandAction_second", "replyAction_second", "likeAction_second", "audienceAction_second", "reportAction_second",
            "related_score", "quality_score", "predict_like_score", "predict_reply_score", "quality_v2_score", "recall_type",
            "copyAction", "copyAction_first", "copyAction_second", "shareAction", "shareAction_first", "shareAction_second",
            "cancelLikeAction_first", "cancelLikeAction_second",
            "hateAction_first", "hateAction_second", "cancelHateAction_first", "cancelHateAction_second",
            "replyTaskAction_first", "replyTaskAction_second", "subAtAction_first", "subAtAction_second", "subShowCntAction",
            'expand_cnt', 'inform_cnt', 'copy_cnt', 'sub_like_cnt', 'first_level_like_cnt',
            "show_cnt_weekly", "like_cnt_weekly", "reply_cnt_weekly", "auto_expand", "first_like_cnt",
            'has_pic', 'has_emoji', 'is_text_pic', 'is_text_emoji', 'is_ai_play', 'is_ai_kwai_wonderful_rely',
            'content_length', 'content_segment_num', 'is_comment_contain_at', 'before_rerank_seq'
]


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
              + [dict(name=f"{item_attr}_list_common_retrieve", path="sample.attr", sample_attr_name=item_attr) for
                 item_attr in item_attrs]
              + [dict(name="request_time", path="timestamp")]
    ) \
    .enrich_attr_by_lua(
        import_common_attr=["request_time"],
        export_common_attr=["time_ms"],
        function_for_common="cal",
        lua_script="""
            function cal()
                return request_time // 1000
            end
        """
    ) \
    .if_("#(comment_id_list_common_retrieve or {}) == 0") \
        .return_(0) \
    .end_() \
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
    


gen_feature = DataReaderFlow(name="gen_feature") \
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
        export_item_attr=["cal_sample_weight"],
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
    .extract_kuiba_parameter(
        config={
            **id_config("gender", 101),     # dim=4
            **id_config("age_segment", 102),
        
            # new_feature
            **id_config("photo_id", 103),   # dim=64
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
            **id_config("comment_id", 201),     # dim=64
            **id_config("author_id", 202),
            
            **discreate_config("like_cnt", 203, [5, 0, 100000, 1, 0]),  # dim=32
            **discreate_config("reply_cnt", 204, [5, 0, 100000, 1, 0]),
            **discreate_config("minute_diff", 205, [36, 0, 336, 1, 0]),
            **discreate_config("ltr", 206, [0.001, 0, 1000, 1, 0]),
            **discreate_config("rtr", 207, [0.001, 0, 1000, 1, 0]),
            **discreate_config("dislike_cnt", 209, [3, 0, 100000, 1, 0]),
            
            **id_config("showAction", 208),         # dim=8

        },
        is_common_attr=False,
        slots_output="comment_item_slots",
        parameters_output="comment_item_signs",
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
    )

other_labels =["slideAction", "complementAction", "seqAction", "slideAction_copy", "depthWeight", "depthAction", "randomAction"]
other_label = DataReaderFlow(name="other_label") \
    .count_reco_result(
        save_count_to="firstCmtShowCnt"
    ) \
    .copy_item_meta_info(
        save_item_seq_to_attr="seqAction"
    ) \
    .pack_item_attr(
        item_source={
            "reco_results": True,
        },
        mappings=[
            {
                "aggregator": "max",
                "from_item_attr": "showAction",
                "to_common_attr": "max_showAction",
            },
            {
                "aggregator": "sum",
                "from_item_attr": "subShowCntAction",
                "to_common_attr": "secondCmtShowCnt",
            },
        ]
    ) \
    .enrich_attr_by_lua(
        import_common_attr=["max_showAction"],
        import_item_attr=["showAction"],
        export_item_attr=["slideAction"],
        function_for_item="calc_slide",
        lua_script=f"""
            function calc_slide()
               
                if showAction + 4 <= max_showAction then
                    return 1
                else
                    return 0
                end
            end
        """
    ) \
    .enrich_attr_by_lua(
        import_item_attr=["slideAction"],
        export_item_attr=["complementAction"],
        function_for_item="complete",
        lua_script=f"""
            function complete()
                return 1 - slideAction
            end
        """
    ) \
    .copy_attr(
        attrs=[
            {
                "from_item": "slideAction",
                "to_item": "slideAction_copy"
            }
        ]
    ) \
    .enrich_attr_by_lua(
        import_common_attr=["firstCmtShowCnt"],
        export_item_attr=["depthWeight", "depthAction", "randomAction"],
        function_for_common="cal_depth_action",
        function_for_item="cal_depth_weight",
        lua_script="""
            local depthAction = 0

            function cal_depth_action()
                if firstCmtShowCnt > 4 then
                    depthAction = 1
                else
                    depthAction = 0
                end
            end

            function cal_depth_weight(seq, item_key, reason, score)
                local rr = util.Random()
                if seq == 0 then
                    return 1.0, depthAction, rr
                else
                    return 0.0, 0, rr
                end
            end
        """
    ) \
    .gen_common_attr_by_lua(
        attr_map={
            "more_than_4": "firstCmtShowCnt > 4",
        }
    ) \
    .perflog_attr_value(
        check_point="other.label",
        item_attrs=["slideAction", "slideAction_copy", "showAction", "depthAction", "depthWeight", "randomAction", "complementAction", "seqAction"],
        common_attrs=["max_showAction", "firstCmtShowCnt", "secondCmtShowCnt", "more_than_4"],
    )

    

send_mio = DataReaderFlow(name="send_mio") \
    .count_reco_result(save_count_to="send_sample_cnt") \
    .perflog_attr_value(
        check_point="send.mio",
        item_attrs=["sample_weight", "new_sample_weight", "cal_sample_weight"] + labels + other_labels,
        common_attrs=["send_sample_cnt"]
    ) \
    .send_to_mio_learner(
        attrs=labels + other_labels,
        slots_attrs=["comment_common_slots", "comment_item_slots"],
        signs_attrs=["comment_common_signs", "comment_item_signs"],
        lineid_attr="user_id",
        time_ms_attr="time_ms",
        label_attr="likeAction_first",
        user_hash_attr="user_hash"
    )

flows = [read_data, gen_feature, other_label, send_mio]
runner = OfflineRunner("comment_profile")
runner.CHECK_UNUSED_ATTR=False
runner.add_leaf_flows(leaf_flows=flows)
runner.build(output_file=os.path.join(current_dir, "cofea_reader.json"))
