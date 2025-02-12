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
              "mod", "page_type_str", 'request_time']

labels=["expandAction_first", "replyAction_first", "likeAction_first", "audienceAction_first",
        "expandAction_second", "replyAction_second", "likeAction_second", "audienceAction_second",
        "copyAction", "copyAction_first", "copyAction_second", "shareAction", "shareAction_first", "shareAction_second",
        "cancelHateAction_first", "cancelHateAction_second", 'cancelLikeAction_first', 'cancelLikeAction_second',
        'hateAction_first', 'hateAction_second', 'replyTaskAction_first', 'replyTaskAction_second', 
        'subAtAction_first', 'subAtAction_second', "stayDurationMs", "stayDurationMs_second", "subShowCntAction",
        "reportAction_first", "reportAction_second"]

item_attrs=["comment_id", "author_id", "like_cnt", "reply_cnt", "minute_diff", "realshow_cnt", "dislike_cnt",
            "showAction", "recall_type",
            "comment_genre", 'content_length']

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


hive_table_column_schema = {
    'comment_id': (0, 'int'),
    'author_id': (1, 'int'),
    'like_cnt': (2, 'int'),
    'reply_cnt': (3, 'int'),
    'dislike_cnt': (4, 'int'),
    'realshow_cnt': (5, 'int'),
    'ltr': (6, 'float'),
    'rtr': (7, 'float'),
    'minute_diff': (8, 'float'),
    'showAction': (9, 'int'),
    'replyaction': (10, 'int'),
    'likeaction': (11, 'int'),
    'expandaction': (12, 'int'),
    'audienceaction': (13, 'int'),
    'reportaction': (14, 'int'),
    'mmu_category_tag': (15, 'int'),
    'mmu_emotion_tag': (16, 'int'),
    'quality_score': (17, 'float'),
    'related_score': (18, 'float'),
    'photo_id': (19, 'int'),
    'photo_author_id': (20, 'int'),
    'user_id': (21, 'int'),
    'device_id': (22, 'string'),
    'gender': (23, 'int'),
    'age_segment': (24, 'int'),
    'quality_v2_score': (25, 'float'),
    'predict_like_score': (26, 'float'),
    'predict_reply_score': (27, 'float'),
    'expandAction_first': (28, 'int'),
    'likeAction_first': (29, 'int'),
    'likeAction_second': (30, 'int'),
    'replyAction_first': (31, 'int'),
    'replyAction_second': (32, 'int'),
    'copyAction_first': (33, 'int'),
    'copyAction_second': (34, 'int'),
    'shareAction_first': (35, 'int'),
    'shareAction_second': (36, 'int'),
    'expandAction_second': (37, 'int'),
    'audienceAction_first': (38, 'int'),
    'audienceAction_second': (39, 'int'),
    'reportAction_first': (40, 'int'),
    'reportAction_second': (41, 'int'),
    'comment_genre': (42, 'int'),
    'has_pic': (43, 'int'),
    'has_emoji': (44, 'int'),
    'is_text_pic': (45, 'int'),
    'is_text_emoji': (46, 'int'),
    'is_comment_contain_at': (47, 'int'),
    'is_ai_play': (48, 'int'),
    'is_ai_kwai_wonderful_rely': (49, 'int'),
    'risk_negative_tag': (50, 'int'),
    'risk_inactive_tag': (51, 'int'),
    'risk_insult_tag': (52, 'int'),
    'like_cnt_weekly': (53, 'int'),
    'reply_cnt_weekly': (54, 'int'),
    'show_cnt_weekly': (55, 'int'),
    'first_level_like_cnt': (56, 'int'),
    'sub_like_cnt': (57, 'int'),
    'auto_expand': (58, 'int'),
    'first_like_cnt': (59, 'int'),
    'inform_cnt': (60, 'int'),
    'copy_cnt': (61, 'int'),
    'content_segment_num': (62, 'int'),
    'content_length': (63, 'int'),
    'is_political': (64, 'int'),
    'page_type_str': (65, 'string'),
    'mod': (66, 'string'),
    'llsid': (67, 'string'),
    'recall_type': (68, 'int'),

    'product_name': (69, 'string'),
    'hateAction_first': (70, 'int'),
    'hateAction_second': (71, 'int'),
    'cancelHateAction_first': (72, 'int'),
    'cancelHateAction_second': (73, 'int'),
    'cancelLikeAction_first': (74, 'int'),
    'cancelLikeAction_second': (75, 'int'),
    'photo_upload_time': (76, 'int'),
    'request_time': (77, 'int'),
    'city_name': (78, 'string'),

    'comment_content_segs': (79, 'string_list'),

    'subAtAction_first': (80, 'int'),
    'subAtAction_second': (81, 'int'),
    'replyTaskAction_first': (82, 'int'),
    'replyTaskAction_second': (83, 'int'),
    'stayDurationMs': (84, 'int'),
    'staydurationms_second': (85, 'int'),
    'comment_content': (86, 'string')
}

read_data = DataReaderFlow(name="read_data") \
    .fetch_message(
        group_id="reco_forward_open_log",
        hdfs_path="viewfs:///home/reco_algorithm/dw/reco_algorithm.db/comment_model_data_wht/data_set=20250120_21_22_22h",
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
            dict(column_index=c_idx, column_name=c_name, type=c_type) for c_name, (c_idx, c_type) in hive_table_column_schema.items() if c_name in common_attrs
        ],
        item_attrs=[
            dict(column_index=c_idx, column_name=c_name, type=c_type) for c_name, (c_idx, c_type) in hive_table_column_schema.items() if c_name not in common_attrs
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
    .set_attr_value(
        item_attrs=[
            {
            "name": "showAction",
            "type": "int",
            "value": 1
            }
        ]
    ) \


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
        import_common_attr=["user_id", "device_id", "photo_id"],
        function_for_common="get_user_photo_hash",
        export_common_attr=["user_photo_hash"],
        lua_script="""
            function get_user_photo_hash()
                if device_id == nil or device_id == "" then
                    local up_hash_str = tostring(user_id) .. tostring(photo_id)
                    return util.CityHash64(up_hash_str)
                end
                local dp_hash_str = tostring(device_id) .. tostring(photo_id)
                return util.CityHash64(dp_hash_str)
            end
        """
    )

send_mio = DataReaderFlow(name="send_mio") \
    .count_reco_result(save_count_to="send_sample_cnt") \
    .perflog_attr_value(
        check_point="send.mio",
        item_attrs=labels,
        common_attrs=["send_sample_cnt"]
    ) \
    .send_to_mio_learner(
        attrs=labels + ["recall_type", "comment_genre", "hate_label"],
        slots_attrs=["comment_common_slots", "comment_item_slots"],
        signs_attrs=["comment_common_signs", "comment_item_signs"],
        lineid_attr="user_id",
        time_ms_attr="time_ms",
        label_attr="likeAction_first",
        user_hash_attr="user_photo_hash"
    )

flows = [read_data, gen_feature, send_mio]
runner = OfflineRunner("comment_profile")
runner.CHECK_UNUSED_ATTR=[]        # 这个不要设置为False，否则不便于发现特征缺失
runner.add_leaf_flows(leaf_flows=flows)
runner.build(output_file=os.path.join(current_dir, "cofea_reader.json"))
