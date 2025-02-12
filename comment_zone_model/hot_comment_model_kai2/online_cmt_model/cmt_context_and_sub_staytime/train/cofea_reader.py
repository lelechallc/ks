from dragonfly.ext.cofea.cofea_api_mixin import CofeaApiMixin
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin
from dragonfly.ext.kgnn.kgnn_api_mixin import KgnnApiMixin
from dragonfly.common_leaf_dsl import LeafFlow, OfflineRunner
from dragonfly.ext.embedding.embedding_api_mixin import EmbeddingApiMixin

import os

current_dir = os.path.dirname(__file__)


class DataReaderFlow(LeafFlow, MioApiMixin, OfflineApiMixin, CofeaApiMixin, KuibaApiMixin, KgnnApiMixin, EmbeddingApiMixin):
    def clean_all(self, reason, **kwargs):
        return self.limit(0, name="clean_all_for_" + reason, **kwargs)


# 要从 kafka 中读取的特征
common_attrs=["user_id", "device_id", "photo_id", "photo_author_id", "gender", "age_segment", 
              "mod", "page_type_str",
]

labels=["expandAction_first", "replyAction_first", "likeAction_first", "audienceAction_first", "reportAction_first","reportAction_second",
        "expandAction_second", "replyAction_second", "likeAction_second", "audienceAction_second",
        "copyAction", "copyAction_first", "copyAction_second", "shareAction", "shareAction_first", "shareAction_second",
        "cancelHateAction_first", "cancelHateAction_second", 'cancelLikeAction_first', 'cancelLikeAction_second',
        'hateAction_first', 'hateAction_second', 'replyTaskAction_first', 'replyTaskAction_second', 
        'subAtAction_first', 'subAtAction_second', "stayDurationMs", "subShowCntAction","stayDurationMs_second"
]

item_attrs=["comment_id", "author_id", "like_cnt", "reply_cnt", "minute_diff", "realshow_cnt", "dislike_cnt",
            "showAction", "recall_type",
            "comment_genre", 'content_length',
]

item_attrs = item_attrs + labels

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
    enrich_attr_by_lua(
        import_item_attr=["content_length",
                          "hateAction_first", "hateAction_second",
                          "cancelHateAction_first", "cancelHateAction_second"],
        export_item_attr=["hate_label"],
        function_for_item="pure_hate_label",
        lua_script="""
            function pure_hate_label()
                local hate_label = 0
                if hateAction_first > cancelHateAction_first or hateAction_second > cancelHateAction_second then
                    hate_label = 1
                end
                if content_length >= 95 and cancelHateAction_first > 0 then
                    hate_label = 0
                end
                return hate_label
            end
        """
    ) \
    .extract_kuiba_parameter(
        config={
            **id_config("gender", 101),     # dim=4
            **id_config("age_segment", 102),
        
            **id_config("photo_id", 103),   # dim=64
            **id_config_slot("photo_author_id", 104, 202),
            **id_config_slot("user_id", 105, 202),
            **id_config("device_id", 106),

            # ## new feature
            **id_config("mod", 110),        # dim=32
            **id_config("page_type_str", 111),  

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

            # ## new feature
            **id_config("comment_genre", 250),      # dim=8
            **discreate_config("content_length", 251, [5, 0, 1000, 1, 0]),      # dim=32

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
    ) \
    # .enrich_attr_by_lua(
    #     import_common_attr=["user_id", "device_id", "photo_id"],
    #     function_for_common="get_user_photo_hash",
    #     export_common_attr=["user_photo_hash"],
    #     lua_script=f"""
    #         function get_user_photo_hash()
    #             if device_id == nil or device_id == "" then
    #                 local up_hash_str = tostring(user_id) .. tostring(photo_id)
    #                 return util.CityHash64(up_hash_str)
    #             end
    #             local dp_hash_str = tostring(device_id) .. tostring(photo_id)
    #             return util.CityHash64(dp_hash_str)
    #         end
    #     """
    # ) \
    

other_labels =["depthAction", "depthMask"]
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
        import_common_attr=["firstCmtShowCnt"],
        export_item_attr=["depthAction", "depthMask"],   # 曝光个数
        function_for_item="cal_depth",
        lua_script="""
            function cal_depth(seq, item_key, reason, score)
                local depthAction = 0
                local mask = 0
                
                if seq == 0 then
                    depthAction = math.max(firstCmtShowCnt - 4, 0)
                    mask = 1
                end
                
                return depthAction, mask
            end
        """
    ) \
    .perflog_attr_value(
        check_point="other.label",
        item_attrs=["showAction", "depthAction", "depthMask", "seqAction"],
        common_attrs=["max_showAction", "firstCmtShowCnt", "secondCmtShowCnt"],
    )
    

send_mio = DataReaderFlow(name="send_mio") \
    .count_reco_result(save_count_to="send_sample_cnt") \
    .perflog_attr_value(
        check_point="send.mio",
        item_attrs=labels,
        common_attrs=["send_sample_cnt"]
    ) \
    .send_to_mio_learner(
        attrs=labels + ['recall_type', 'comment_genre', 'hate_label'] + other_labels,
        slots_attrs=["comment_common_slots", "comment_item_slots"],
        signs_attrs=["comment_common_signs", "comment_item_signs"],
        lineid_attr="user_id",
        time_ms_attr="time_ms",
        label_attr="likeAction_first",
        user_hash_attr="user_hash"
    )

flows = [read_data, gen_feature, other_label, send_mio]
runner = OfflineRunner("comment_profile")
# runner.CHECK_UNUSED_ATTR=False        # 这个不要设置为False，否则不便于发现特征缺失
runner.add_leaf_flows(leaf_flows=flows)
runner.build(output_file=os.path.join(current_dir, "cofea_reader.json"))
