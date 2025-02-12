"""
注意：eval时showAction置为1
新增单评论停留时长标签。
"""

from dragonfly.ext.cofea.cofea_api_mixin import CofeaApiMixin
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin
from dragonfly.ext.kgnn.kgnn_api_mixin import KgnnApiMixin
from dragonfly.ext.embedding.embedding_api_mixin import EmbeddingApiMixin
from dragonfly.common_leaf_dsl import LeafFlow, OfflineRunner
import os
import argparse


current_dir = os.path.dirname(__file__)


class DataReaderFlow(LeafFlow, MioApiMixin, OfflineApiMixin, CofeaApiMixin, KuibaApiMixin, KgnnApiMixin, EmbeddingApiMixin):
    def clean_all(self, reason, **kwargs):
        return self.limit(0, name="clean_all_for_" + reason, **kwargs)


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default='')    # viewfs:///home/reco_algorithm/dw/reco_algorithm.db/comment_model_data_wht/data_set=20240421_9_12_4h
# parser.add_argument("--mode", type=str, default='train', help='train | eval') 
args = parser.parse_args()


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
}

common_attrs=["user_id", "device_id", "photo_id", "photo_author_id", "gender", "age_segment", "request_time",
              "mod", "page_type_str", 
              # "llsid", "city_name",  "photo_upload_time", 
            #   "is_political", "product_name",
              ]
labels = [
    'likeAction_first',
    'likeAction_second',
    'replyAction_first',
    'replyAction_second',
    'expandAction_first',
    'expandAction_second',
    'copyAction_first',
    'copyAction_second',
    'shareAction_first',
    'shareAction_second',
    'audienceAction_first',
    'audienceAction_second',
    'reportAction_first',
    'reportAction_second',

    'hateAction_first',
    'hateAction_second',
    'cancelHateAction_first',
    'cancelHateAction_second',
    'cancelLikeAction_first',
    'cancelLikeAction_second',
    'subAtAction_first',
    'subAtAction_second',
    'replyTaskAction_first',
    'replyTaskAction_second',
    'stayDurationMs'
]
item_attrs=['comment_id', 'author_id', 'like_cnt', 'reply_cnt', 'dislike_cnt', 'realshow_cnt', 'minute_diff', 
            'comment_genre', 'content_length', 
            # 'comment_content_segs', 'mmu_category_tag', 'mmu_emotion_tag',  'related_score',
             # 'risk_insult_tag', 'risk_negative_tag', 'risk_inactive_tag', 
            # 'quality_v2_score', 'quality_score', 
            # 'show_cnt_weekly', 'like_cnt_weekly', 'reply_cnt_weekly', 'auto_expand', 'first_like_cnt',
            # 'sub_like_cnt', 'first_level_like_cnt', 
            # 'content_segment_num', 'inform_cnt', 'copy_cnt',
            # 'has_pic', 'has_emoji', 'is_text_pic', 'is_text_emoji', 'is_ai_play', 'is_ai_kwai_wonderful_rely', 'is_comment_contain_at',
            'recall_type'   # recall_type用于区分爬评热评
            ]
item_attrs = item_attrs + labels

set_default_value_int_attrs = []
set_default_value_double_attrs = []
set_default_value_string_attrs = []
set_default_value_string_list_attrs = []

for feat, v in hive_table_column_schema.items():
    if feat in common_attrs+item_attrs:
        if v[1] == 'int':
            set_default_value_int_attrs.append(feat)
        elif v[1] =='float':
            set_default_value_double_attrs.append(feat)
        elif v[1] == 'string':
            set_default_value_string_attrs.append(feat)
        elif v[1] == 'string_list':
            set_default_value_string_list_attrs.append(feat)

default_value_map_list_for_item = []
default_value_map_list_for_common = []
for name in set_default_value_int_attrs:
    if name in common_attrs:
        default_value_map_list_for_common.append({"name": name, "type": "int", "value": 0})
    elif name in item_attrs:
        default_value_map_list_for_item.append({"name": name, "type": "int", "value": 0})
for name in set_default_value_double_attrs:
    if name in common_attrs:
        default_value_map_list_for_common.append({"name": name, "type": "double", "value": 0.0})
    elif name in item_attrs:  
        default_value_map_list_for_item.append({"name": name, "type": "double", "value": 0.0})
for name in set_default_value_string_attrs:
    if name in common_attrs:
        default_value_map_list_for_common.append({"name": name, "type": "string", "value": ''})
    elif name in item_attrs:
        default_value_map_list_for_item.append({"name": name, "type": "string", "value": ''})

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
        group_id="wht",
        hdfs_path=args.data_path,
        hdfs_format="raw_text",
        output_attr="csv_sample_data",
    ) \
    .convert_csv_to_tf_sequence_example(
        from_extra_var="csv_sample_data",
        common_attrs=[
            dict(column_index=c_idx, column_name=c_name, type=c_type) for c_name, (c_idx, c_type) in hive_table_column_schema.items() if c_name in common_attrs
        ],
        item_attrs=[
            dict(column_index=c_idx, column_name=c_name, type=c_type) for c_name, (c_idx, c_type) in hive_table_column_schema.items() if c_name in item_attrs
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

    

gen_new_feature = DataReaderFlow(name="gen_new_feature") \
    .fetch_remote_embedding(    # 调用迁移的新emb服务
        protocol=1,
        id_converter = {"type_name": "mioEmbeddingIdConverter"},
        colossusdb_embd_service_name="hot_comment_xtr_v3_c26",
        colossusdb_embd_table_name="emb_slide_multi_task_with_global",
        query_source_type="common_attr",
        input_attr_name="user_id",
        output_attr_name ="uid_emb",
        is_raw_data=True, 
        raw_data_type="scale_int8",
        slot=38,
        size=64,
        enable_smaller_size=True,
        max_signs_per_request=500,
        timeout_ms=20,
    ) \
    .fetch_remote_embedding(    # 调用迁移的新emb服务
        protocol=1,
        id_converter = {"type_name": "mioEmbeddingIdConverter"},
        colossusdb_embd_service_name="hot_comment_xtr_v3_c26",
        colossusdb_embd_table_name="emb_slide_multi_task_with_global",
        query_source_type="common_attr",
        input_attr_name="photo_id",
        output_attr_name ="pid_emb",
        is_raw_data=True, 
        raw_data_type="scale_int8",
        slot=26,
        size=64,
        max_signs_per_request=500,
        timeout_ms=20,
    ) \
    .enrich_attr_by_lua(
        import_common_attr=["uid_emb", "pid_emb"],
        function_for_common="check_exist",
        export_common_attr=["uid_emb_exist", 'pid_emb_exist'],
        lua_script=f"""
            function check_exist()
                local uid_emb_exist=1
                local pid_emb_exist=1
                if uid_emb == nil then
                    uid_emb_exist=0
                end
                if pid_emb == nil then
                    pid_emb_exist=0
                end
                return uid_emb_exist, pid_emb_exist
            end
        """
    )

    

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
    .extract_kuiba_parameter(
        config={
            **id_config("gender", 101),     # dim=4
            **id_config("age_segment", 102),
        
            **id_config("photo_id", 103),   # dim=64
            **id_config_slot("photo_author_id", 104, 202),
            **id_config_slot("user_id", 105, 202),
            **id_config("device_id", 106),

            # new feature
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
        lua_script=f"""
            function get_user_photo_hash()
                if device_id == nil or device_id == "" then
                    local up_hash_str = tostring(user_id) .. tostring(photo_id)
                    return util.CityHash64(up_hash_str)
                end
                local dp_hash_str = tostring(device_id) .. tostring(photo_id)
                return util.CityHash64(dp_hash_str)
            end
        """
    ) \



send_mio = DataReaderFlow(name="send_mio") \
    .count_reco_result(save_count_to="send_sample_cnt") \
    .perflog_attr_value(
        check_point="wht_test.offline_test.cofea_reader_offline_eval2",
        common_attrs=['send_sample_cnt', ],
        item_attrs=labels + ['comment_genre', 'content_length']
    ) \
    .log_debug_info(
        for_debug_request_only=False,
        respect_sample_logging=True,
        print_all_common_attrs=True,
        print_all_item_attrs=True
    ) \
    .send_to_mio_learner(
        attrs=labels + ['recall_type', 'sample_weight',],
        slots_attrs=["comment_common_slots", "comment_item_slots"],
        signs_attrs=["comment_common_signs", "comment_item_signs"],
        lineid_attr="user_id",
        time_ms_attr="request_time",
        label_attr="likeAction_first",
        user_hash_attr="user_photo_hash"
    )

flows = [read_data, gen_feature, send_mio]
runner = OfflineRunner("comment_model")
# runner.IGNORE_UNUSED_ATTR=['comment_weights_int']  + ['comment_mmu_categories']
runner.add_leaf_flows(leaf_flows=flows)
runner.build(output_file=os.path.join(current_dir, "cofea_reader.json"))
