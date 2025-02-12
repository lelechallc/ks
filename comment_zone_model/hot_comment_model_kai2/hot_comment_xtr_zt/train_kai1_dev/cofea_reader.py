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
item_attrs=["comment_id", "author_id", "like_cnt", "reply_cnt", "dislike_cnt", "realshow_cnt", "minute_diff", "ltr", "rtr",
            "showaction", "expandaction", "replyaction", "likeaction", "audienceaction", "reportaction",
            "comment_content_segs", "comment_genre", "risk_insult_tag", "risk_negative_tag", "risk_inactive_tag ",
            "mmu_category_tag", "mmu_emotion_tag", "mmu_entity_list",
            "predict_reply_score", "quality_v2_score", "predict_like_score"]

value_dict = {
    # item
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
    "comment_genre": "int",
    "risk_insult_tag": "int",
    "risk_negative_tag": "int",
    "risk_inactive_tag": "int",
    "mmu_category_tag": "int",
    "mmu_emotion_tag": "int",
    "mmu_entity_list": "string_list",
    "predict_reply_score": "float",
    "quality_v2_score": "float",
    "predict_like_score": "float",

    # common
    "age_segment": "int",
    "gender": "int",
    "device_id": "string",
    "photo_id": "int",
    "photo_author_id": "int",
    "user_id": "int"
}

common_attr_set = {"user_id", "device_id", "photo_id", "photo_author_id", "gender", "age_segment"}

read_data_valid = DataReaderFlow(name="read_data") \
    .fetch_message(
        group_id="reco_forward_open_log",
        # hdfs_path="viewfs:///home/ds/dw/ks_data_science.db/reco_comment_zone_model_sample_data_train_test/data_type=0327_8_11Train",
        hdfs_path="viewfs:///home/reco_algorithm/dw/reco_algorithm.db/comment_model_data_zt08_dev1/datatype=0424_18_19_2hTrain",
        # hdfs_path="viewfs:///home/reco_algorithm/dw/reco_algorithm.db/comment_model_data_zt08_dev1/datatype=0421_12_1hTest",
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
            dict(column_index=16, column_name="age_segment", type="int"),
            dict(column_index=17, column_name="gender", type="int"),
            dict(column_index=18, column_name="device_id", type="string"),
            dict(column_index=19, column_name="photo_id", type="int"),
            dict(column_index=20, column_name="photo_author_id", type="int"),
            dict(column_index=28, column_name="user_id", type="int")
            # dict(column_index=i, column_name=v[0], type=v[1]) for (i, v) in enumerate(value_dict.items()) if v[0] in common_attr_set
        ],
        item_attrs=[
            dict(column_index=0, column_name="comment_id", type="int"),
            dict(column_index=1, column_name="author_id", type="int"),
            dict(column_index=2, column_name="like_cnt", type="int"),
            dict(column_index=3, column_name="reply_cnt", type="int"),
            dict(column_index=4, column_name="dislike_cnt", type="int"),
            dict(column_index=5, column_name="realshow_cnt", type="int"),
            dict(column_index=6, column_name="minute_diff", type="float"),
            dict(column_index=7, column_name="ltr", type="float"),
            dict(column_index=8, column_name="rtr", type="float"),
            dict(column_index=9, column_name="showaction", type="int"),
            dict(column_index=10, column_name="replyaction", type="int"),
            dict(column_index=11, column_name="likeaction", type="int"),
            dict(column_index=12, column_name="expandaction", type="int"),
            dict(column_index=13, column_name="audienceaction", type="int"),
            dict(column_index=14, column_name="reportaction", type="int"),
            dict(column_index=15, column_name="comment_content_segs", type="string_list"),
            dict(column_index=21, column_name="comment_genre", type="int"),
            dict(column_index=22, column_name="risk_insult_tag", type="int"),
            dict(column_index=23, column_name="risk_negative_tag", type="int"),
            dict(column_index=24, column_name="risk_inactive_tag", type="int"),
            dict(column_index=25, column_name="mmu_category_tag", type="int"),
            dict(column_index=26, column_name="mmu_emotion_tag", type="int"),
            dict(column_index=27, column_name="mmu_entity_list", type="string_list"),
            dict(column_index=29, column_name="predict_reply_score", type="float"),
            dict(column_index=30, column_name="quality_v2_score", type="float"),
            dict(column_index=31, column_name="predict_like_score", type="float")
            # dict(column_index=i, column_name=v[0], type=v[1]) for (i, v) in enumerate(value_dict.items()) if v[0] not in common_attr_set
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
    .count_reco_result(
        save_count_to="retrieve_num"
    ) \
    .if_("retrieve_num <= 0") \
        .return_(0) \
    .end_() \
    .set_attr_default_value(
        item_attrs=[
            {
                "name": "expandaction",
                "type": "int",
                "value": 0
            },
            {
                "name": "replyaction",
                "type": "int",
                "value": 0
            },
            {
                "name": "likeaction",
                "type": "int",
                "value": 0
            },
            {
                "name": "audienceaction",
                "type": "int",
                "value": 0
            },
            {
                "name": "reportaction",
                "type": "int",
                "value": 0
            },
            {
                "name": "showaction",
                "type": "int",
                "value": 0
            }
        ]
    ) \
    .enrich_attr_by_lua(
        import_item_attr=["showaction", "expandaction", "replyaction", "likeaction", "audienceaction", "reportaction"],
        export_item_attr=["sample_weight"],
        function_for_item="cal_sample_weight",
        lua_script="""
            function cal_sample_weight()
                local weight = showaction * 1.0
                if (expandaction or 0) > 0 then
                  weight = weight + 3.0
                end
                if (replyaction or 0) > 0 then
                  weight = weight + 5.0
                end
                if (likeaction or 0) > 0 then
                  weight = weight + 3.0
                end
                if (audienceaction or 0) > 0 then
                  weight = weight + 3.0
                end
                if (reportaction or 0) > 0 then
                  weight = weight + 8.0
                end
                return weight
            end
        """
    ) \
    .enrich_attr_by_lua(
        import_item_attr=["like_cnt", "reply_cnt"],
        export_item_attr=["like_reply_tr"],
        function_for_item="cal_like_reply_tr",
        lua_script="""
            function cal_like_reply_tr()
                return like_cnt * 1.0 / (reply_cnt + 1.0)
            end
        """
    ) \
    .log_debug_info(
        for_debug_request_only = False,
        respect_sample_logging = True,
        print_all_common_attrs = True,
        print_all_item_attrs = True
    ) \
    # .get_remote_embedding_lite(
    #     # mmu提供的content embedding
    #     kess_service="grpc_mmuCommentContentEmb",
    #     id_converter={"type_name": "mioEmbeddingIdConverter"},
    #     query_source_type="item_attr",
    #     input_attr_name="comment_id",
    #     output_attr_name="mmu_comment_content_emb",
    #     timeout_ms=10,
    #     slot=100,
    #     size=256,
    #     shard_num=4,
    #     client_side_shard=True,
    # ) \
    # .enrich_attr_by_lua(
    #     import_item_attr=["realshow_cnt", "ltr", "rtr"],
    #     export_item_attr=["ltr", "rtr"],
    #     function_for_item="reset_unbelievable_cmt_xtr",
    #     lua_script="""
    #       function reset_unbelievable_cmt_xtr()
    #         local ltr = ltr
    #         local rtr = rtr
    #         if (realshow_cnt < 80) then
    #             ltr = -1.0
    #             rtr = -1.0
    #         end
    #         return ltr, rtr
    #       end
    #     """
    # ) \
    # .log_debug_info(
    #     for_debug_request_only = False,
    #     respect_sample_logging = True,
    #     item_attrs=["comment_id", "realshow_cnt", "ltr", "rtr"],
    #     select_item={
    #         "attr_name": "realshow_cnt",
    #         "compare_to": 80,
    #         "select_if": "<",
    #     }
    # ) \
    # .perflog_attr_value(
    #     check_point="read_data_valid.perflog",
    #     item_attrs=["comment_id", "realshow_cnt", "ltr", "rtr"],
    #     select_item={
    #         "attr_name": "realshow_cnt",
    #         "compare_to": 80,
    #         "select_if": "<",
    #     }
    # )


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
        export_item_attr=["sample_weight"],
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
    .log_debug_info(
        common_attrs=common_attrs,
        item_attrs=item_attrs,
        for_debug_request_only=False,
        respect_sample_logging=True
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

fetch_profile = DataReaderFlow(name="fetch_profile") \
    .fetch_kgnn_neighbors(
        id_from_common_attr="user_id",
        # save_weight_to="interact_comment_weights",  # like + reply
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
        # save_weight_to="write_comment_weights",  # like + reply
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
    .split_string(
        input_item_attr="comment_content_segs",
        output_item_attr="comment_content_segs",
        delimiters="_",
    ) \
    .split_string(
        input_item_attr="mmu_entity_list",
        output_item_attr="mmu_entity_list",
        delimiters="_",
    ) \
    .perflog_attr_value(
        check_point="comment.fea",
        common_attrs=["gender", "age_segment"],
    ) \
    .enrich_attr_by_lua(
        import_item_attr=["ltr", "rtr", "like_cnt", "reply_cnt",
                          "predict_reply_score", "quality_v2_score", "predict_like_score"],
        function_for_item="fre_group",
        export_item_attr=["fre_grouped_ltr", "fre_grouped_rtr", "fre_grouped_like_cnt", "fre_grouped_reply_cnt",
                          "fre_grouped_predict_reply_score", "fre_grouped_quality_v2_score", "fre_grouped_predict_like_score"],
        lua_script_file="./lua_scripts/fre_dis_100.lua"
      ) \
    .set_attr_value(
        item_attrs=[
            {
                "name": "field_emb",
                "type": "int",
                "value": 1
            }
        ]
    )\
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

            **id_config("field_emb", 404), # Autodis分桶的 meta embedding

            # 对realshow_cnt进行等距分桶
            **discreate_config("realshow_cnt", 405, [8000, 0, 100000, 1, 0]),


            **discreate_config("like_reply_tr", 406, [0.001, 0, 10000, 1, -1]),

            # 等频MMU提供的3个分数
            **id_config("fre_grouped_predict_reply_score", 407),
            **id_config("fre_grouped_quality_v2_score", 408),
            **id_config("fre_grouped_predict_like_score", 409),


            **discreate_config("like_cnt", 203, [5, 0, 100000, 1, 0]),
            **discreate_config("reply_cnt", 204, [5, 0, 100000, 1, 0]),
            # 等频分桶like_cnt,reply_cnt
            # **id_config("fre_grouped_like_cnt", 203),
            # **id_config("fre_grouped_reply_cnt", 204),

            **discreate_config("minute_diff", 205, [36, 0, 336, 1, 0]),

            **discreate_config("ltr", 206, [0.001, 0, 1000, 1, -1]),
            **discreate_config("rtr", 207, [0.001, 0, 1000, 1, -1]),
            **discreate_config("like_reply_tr", 406, [0.001, 0, 10000, 1, -1]),

            # 等频分桶ltr，rtr
            # **id_config("fre_grouped_ltr", 206),
            # **id_config("fre_grouped_rtr", 207),

            # **id_config("ltr", 206),  # Autodis id类特征仅能 int类型
            # **id_config("rtr", 207),

            **id_config("showaction", 208),
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
        },
        is_common_attr=False,
        slots_output="comment_item_slots",
        parameters_output="comment_item_signs",
    )
    

send_mio = DataReaderFlow(name="send_mio") \
    .count_reco_result(save_count_to="send_sample_cnt") \
    .enrich_attr_by_lua(
        import_item_attr=["expandaction", "likeaction", "replyaction"],
        export_item_attr=["expandaction_v", "likeaction_v", "replyaction_v"],
        function_for_item="trans",
        lua_script="""
            function trans() 
                return expandaction * 1.0, likeaction * 1.0, replyaction * 1.0
            end
        """
    ) \
    .perflog_attr_value(
        check_point="comment.mio.record",
        item_attrs=["sample_weight", "expandaction_v", "likeaction_v", "replyaction_v", "expandaction", "likeaction", "replyaction"],
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
                    "mmu_comment_content_emb",
                    "predict_reply_score", "quality_v2_score", "predict_like_score",
                    "fre_grouped_predict_reply_score", "fre_grouped_quality_v2_score", "fre_grouped_predict_like_score"
                    ],
        common_attrs=["user_hash", "photo_hash"]
    ) \
    .send_to_mio_learner(     # wht: 这是啥？
        # 在这里就会丢失photo hash的int64精度，mio不支持int64
        attrs=["sample_weight",
               "expandaction_v", "likeaction_v", "replyaction_v",
               "expandaction", "likeaction", "replyaction",
               "like_cnt", "reply_cnt", "ltr", "rtr", "like_reply_tr",
               "mmu_comment_content_emb"],
        slots_attrs=["comment_common_slots", "comment_item_slots"],
        signs_attrs=["comment_common_signs", "comment_item_signs"],
        lineid_attr="user_id",
        time_ms_attr="time_ms",
        label_attr="likeaction_v",
        user_hash_attr="photo_hash"
    ) \
    .log_debug_info(
      print_all_common_attrs=True,
      print_all_item_attrs=True,
      for_debug_request_only=False,
      respect_sample_logging=True,
    )

pipelines = [read_data_valid, extract_fea, send_mio]
runner = OfflineRunner("comment_profile")
runner.IGNORE_UNUSED_ATTR=['llsid']
runner.add_leaf_flows(leaf_flows=pipelines)
runner.build(output_file=os.path.join(current_dir, "cofea_reader.json"))
