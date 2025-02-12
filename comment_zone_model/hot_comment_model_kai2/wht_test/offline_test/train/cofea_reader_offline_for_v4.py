"""
注意：eval时showAction置为1
"""
""" 
TODO
- 作者是否赞过
- 看评人与写评人的社交关系
"""

from dragonfly.ext.cofea.cofea_api_mixin import CofeaApiMixin
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin
from dragonfly.ext.kgnn.kgnn_api_mixin import KgnnApiMixin
from dragonfly.ext.kgnn.node_attr_schema import NodeAttrSchema
from dragonfly.common_leaf_dsl import LeafFlow, OfflineRunner
import os
import argparse


current_dir = os.path.dirname(__file__)


class DataReaderFlow(LeafFlow, MioApiMixin, OfflineApiMixin, CofeaApiMixin, KuibaApiMixin, KgnnApiMixin):
    def clean_all(self, reason, **kwargs):
        return self.limit(0, name="clean_all_for_" + reason, **kwargs)


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default='')    # viewfs:///home/reco_algorithm/dw/reco_algorithm.db/comment_model_data_wht/data_set=20240421_9_12_4h
# parser.add_argument("--mode", type=str, default='train', help='train | eval') 
args = parser.parse_args()


common_attrs=["user_id", "device_id", "photo_id", "photo_author_id", "gender", "age_segment", "llsid",
            "mod", "page_type_str", "is_political", "product_name", "photo_upload_time", "request_time", "city_name"]
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

    'hateaction_first',
    'hateaction_second',
    'cancelhateaction_first',
    'cancelhateaction_second',
    'cancellikeaction_first',
    'cancellikeaction_second',
]
item_attrs=['comment_id', 'author_id', 'like_cnt', 'reply_cnt', 'dislike_cnt', 'realshow_cnt', 'minute_diff', 
            'comment_genre', 'risk_insult_tag', 'risk_negative_tag', 'risk_inactive_tag', 'mmu_category_tag', 
            'mmu_emotion_tag', 'quality_v2_score', 'quality_score', 'related_score',
            'show_cnt_weekly', 'like_cnt_weekly', 'reply_cnt_weekly', 'auto_expand', 'first_like_cnt',
            'sub_like_cnt', 'first_level_like_cnt', 
            'content_length', 'content_segment_num', 'inform_cnt', 'copy_cnt',
            'has_pic', 'has_emoji', 'is_text_pic', 'is_text_emoji', 'is_ai_play', 'is_ai_kwai_wonderful_rely', 'is_comment_contain_at',
            'recall_type'   # recall_type用于区分爬评热评
            ]
item_attrs = item_attrs + labels

unused_attrs=[]
item_attrs = [attr for attr in item_attrs if attr not in unused_attrs]

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
    'hateaction_first': (70, 'int'),
    'hateaction_second': (71, 'int'),
    'cancelhateaction_first': (72, 'int'),
    'cancelhateaction_second': (73, 'int'),
    'cancellikeaction_first': (74, 'int'),
    'cancellikeaction_second': (75, 'int'),
    'photo_upload_time': (76, 'int'),
    'request_time': (77, 'int'),
    'city_name': (78, 'string'),
    
}
read_data = DataReaderFlow(name="read_data") \
    .fetch_message(
        group_id="reco_forward_open_log",
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
            dict(column_index=c_idx, column_name=c_name, type=c_type) for c_name, (c_idx, c_type) in hive_table_column_schema.items() if c_name not in common_attrs
        ],
        column_separator="|",
        item_separator=",",
        list_separator=" ",
        save_result_to="tf_sequence_example"
    ) \
    .log_debug_info(
        common_attrs = ["csv_sample_data", "tf_sequence_example"],
        for_debug_request_only=False,
        respect_sample_logging=True,
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



gen_kgnn_fea = DataReaderFlow(name="gen_kgnn_fea") \
    .fetch_kgnn_neighbors(
        id_from_common_attr="user_id",
        save_weight_to="comment_weights",  # like + reply
        save_neighbors_to="comment_ids",
        edge_attr_schema=NodeAttrSchema(1, 0).add_int64_list_attr("comment_mmu_categories", 1),
        kess_service="grpc_kgnn_user_interact_comment_info-U2I",
        relation_name='U2I',
        shard_num=4,
        sample_num=20,
        timeout_ms=10,
        sample_type="topn",
        padding_type="zero",
    ) \
    .cast_attr_type(
        attr_type_cast_configs=[
            {
                "to_type": "int",
                "from_common_attr": "comment_weights",
                "to_common_attr": "comment_weights_int"
            }
        ]
    ) \
    .log_debug_info(
        common_attrs=["user_id", "comment_ids", "comment_mmu_categories", "comment_weights", "comment_weights_int"],
        for_debug_request_only=False,
        respect_sample_logging=True,
    )


set_default_value_int_attrs = []
set_default_value_double_attrs = []
set_default_value_string_attrs = []
for feat, v in hive_table_column_schema.items():
    if v[1] == 'int':
        set_default_value_int_attrs.append(feat)
    elif v[1] =='float':
        set_default_value_double_attrs.append(feat)
    elif v[1] == 'string':
        set_default_value_string_attrs.append(feat)

default_value_map_list_for_item = []
default_value_map_list_for_common = []
for name in set_default_value_int_attrs:
    if name in common_attrs:
        default_value_map_list_for_common.append({"name": name, "type": "int", "value": 0})
    else:
        default_value_map_list_for_item.append({"name": name, "type": "int", "value": 0})
for name in set_default_value_double_attrs:
    if name in common_attrs:
        default_value_map_list_for_common.append({"name": name, "type": "double", "value": 0.0})
    else:  
        default_value_map_list_for_item.append({"name": name, "type": "double", "value": 0.0})
for name in set_default_value_string_attrs:
    if name in common_attrs:
        default_value_map_list_for_common.append({"name": name, "type": "string", "value": ''})
    else:
        default_value_map_list_for_item.append({"name": name, "type": "string", "value": ''})


gen_feature = DataReaderFlow(name="gen_feature") \
    .set_attr_value(
        common_attrs=default_value_map_list_for_common,
        item_attrs=default_value_map_list_for_item,
        no_overwrite=True
    ) \
    .enrich_attr_by_lua(
        import_common_attr = ['request_time'],
        function_for_common = "calc_hour_day",
        export_common_attr = ["request_hour", 'request_day'],
        lua_script = """
            function calc_hour_day()
                if request_time == nil then
                    return -1, -1
                end
                local SECONDS_PER_DAY = 24 * 60 * 60
                local SECONDS_PER_HOUR = 60 * 60
                local UTC_OFFSET = 8 * SECONDS_PER_HOUR     -- utc+8时区

                -- 将毫秒转换为秒
                local timestamp_s = math.floor(request_time / 1000) + UTC_OFFSET
                
                -- 计算当天已经过去的秒数
                local seconds_today = timestamp_s % (24 * 60 * 60)
                
                -- 计算小时数
                local hour = math.floor(seconds_today / 3600)
                
                -- 计算自1970年1月1日以来的天数
                -- 1970年1月1日是星期四，所以加4
                local days_since_epoch = math.floor(timestamp_s / (24 * 60 * 60)) + 4
                
                -- 0-6，0是星期日，1是星期一
                local day_of_week = days_since_epoch % 7

                return hour, day_of_week
            end
        """
    ) \
    .enrich_attr_by_lua(
        import_item_attr=["like_cnt", "reply_cnt", "dislike_cnt", "realshow_cnt", 'first_like_cnt', 'sub_like_cnt', 
                          'first_level_like_cnt', "show_cnt_weekly", "like_cnt_weekly", "reply_cnt_weekly",
                          "copy_cnt", "minute_diff"],
        export_item_attr=["ltr", "rtr", "dtr", "sqrt_like_cnt", "sqrt_reply_cnt", 'sqrt_dislike_cnt', 
                          'sqrt_hour_diff', 'sqrt_sub_like_cnt', 'sqrt_first_level_like_cnt', 'sqrt_first_like_cnt',
                          'dislike_like_ratio', 'sub_root_like_ratio', 'ltr_weekly', 'rtr_weekly', 'sqrt_copy_cnt',
                          "ltr_copy", "rtr_copy"],
        function_for_item="cal_xtr",
        lua_script="""
            function cal_xtr()
                local vv = realshow_cnt or 0.0
                local ltr = like_cnt / (vv + 1.0)
                local rtr = reply_cnt / (vv + 1.0)
                local dtr = dislike_cnt / (vv + 1.0)
                local sqrt_like_cnt = math.sqrt(like_cnt)
                local sqrt_reply_cnt = math.sqrt(reply_cnt)
                local sqrt_dislike_cnt = math.sqrt(dislike_cnt)
                local sqrt_hour_diff = math.sqrt(minute_diff / 60)
                local sqrt_sub_like_cnt = math.sqrt(sub_like_cnt)
                local sqrt_first_level_like_cnt = math.sqrt(first_level_like_cnt)
                local sqrt_first_like_cnt = math.sqrt(first_like_cnt)
                local dislike_like_ratio = dislike_cnt / (like_cnt + 1.0)
                local sub_root_like_ratio = first_like_cnt / (first_level_like_cnt + 1.0)
                local ltr_weekly = like_cnt_weekly / (show_cnt_weekly + 1.0)
                local rtr_weekly = reply_cnt_weekly / (show_cnt_weekly + 1.0)
                local sqrt_copy_cnt = math.sqrt(copy_cnt)
                return ltr, rtr, dtr, sqrt_like_cnt, sqrt_reply_cnt, sqrt_dislike_cnt, sqrt_hour_diff, sqrt_sub_like_cnt, sqrt_first_level_like_cnt, sqrt_first_like_cnt, dislike_like_ratio, sub_root_like_ratio, ltr_weekly, rtr_weekly, sqrt_copy_cnt, ltr, rtr
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
        
            # new_feature
            **id_config("photo_id", 103),   # dim=64
            **id_config_slot("photo_author_id", 104, 202),
            **id_config_slot("user_id", 105, 202),
            **id_config("device_id", 106),
            **id_config("mod", 110),  
            **id_config("page_type_str", 111),  
            **id_config("city_name", 114),   

            **id_config("is_political", 112),     # dim=4
            **id_config("product_name", 113),     # dim=4也够了
            
            **id_config("request_hour", 115),       # dim=8
            **id_config("request_day", 116),       
            
            
            # # 序列特征
            # **list_config("comment_ids", 300, 201, 20),
            # "comment_weights_int": {"attrs": [{"key_type": 301, "attr": ["comment_weights_int"], **kuiba_list_converter_config_list_limit(20)}]},
            # **list_config("comment_mmu_categories", 302, 255, 20),
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

            # 以下是新特征  
            **id_config("comment_genre", 250),      # dim=8
            **id_config("risk_insult_tag", 251),    
            **id_config("risk_inactive_tag", 252),
            **id_config("risk_negative_tag", 253),
            **id_config("mmu_emotion_tag", 254),
            **id_config("mmu_category_tag", 255),

            # denominator, smooth, max_val, buckets, min_val
            **discreate_config("sqrt_like_cnt", 271, [1, 0, 1000, 1, 0]),   # dim=12
            **discreate_config("sqrt_reply_cnt", 272, [1, 0, 1000, 1, 0]),
            **discreate_config("sqrt_dislike_cnt", 273, [1, 0, 1000, 1, 0]),
            **discreate_config("sqrt_hour_diff", 274, [1, 0, 1000, 1, 0]),
            **discreate_config("sqrt_sub_like_cnt", 275, [1, 0, 1000, 1, 0]),
            **discreate_config("sqrt_first_level_like_cnt", 276, [1, 0, 1000, 1, 0]),
            **discreate_config("sqrt_first_like_cnt", 277, [1, 0, 1000, 1, 0]),

            **discreate_config("quality_score", 278, [0.01, 0, 100, 1, 0]), # dim=8
            **discreate_config("related_score", 279, [0.01, 0, 100, 1, 0]),
            **discreate_config("quality_v2_score", 280, [0.01, 0, 100, 1, 0]),
            **discreate_config("ltr_copy", 281, [0.01, 0, 100, 1, 0]),
            **discreate_config("rtr_copy", 282, [0.01, 0, 100, 1, 0]),
            **discreate_config("dtr", 283, [0.01, 0, 100, 1, 0]),
            **discreate_config("ltr_weekly", 284, [0.01, 0, 100, 1, 0]),
            **discreate_config("rtr_weekly", 285, [0.01, 0, 100, 1, 0]),

            **discreate_config("dislike_like_ratio", 286, [0.01, 0, 1000, 1, 0]),   # dim=12
            **discreate_config("sub_root_like_ratio", 287, [0.01, 0, 1000, 1, 0]),
            **discreate_config("content_length", 288, [5, 0, 1000, 1, 0]),      
            **discreate_config("content_segment_num", 289, [1, 0, 1000, 1, 0]), 
            **discreate_config("inform_cnt", 290, [1, 0, 1000, 1, 0]),         
            **discreate_config("sqrt_copy_cnt", 291, [1, 0, 1000, 1, 0]),       

            **id_config("auto_expand", 270),   # dim=4
            **id_config("has_pic", 292),       
            **id_config("has_emoji", 293),   
            **id_config("is_text_pic", 294),   
            **id_config("is_text_emoji", 295),   
            **id_config("is_ai_play", 296),   
            **id_config("is_ai_kwai_wonderful_rely", 297),   
            **id_config("is_comment_contain_at", 298),

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
        check_point="wht_test.offline_test.cofea_reader_online_for_v4",
        common_attrs=['request_hour', 'request_day'],
        item_attrs=set_default_value_int_attrs + set_default_value_double_attrs + ["ltr", "rtr", "dtr", "sqrt_like_cnt", "sqrt_reply_cnt", 'sqrt_dislike_cnt', 
                          'sqrt_hour_diff', 'sqrt_sub_like_cnt', 'sqrt_first_level_like_cnt', 'sqrt_first_like_cnt',
                          'dislike_like_ratio', 'sub_root_like_ratio', 'ltr_weekly', 'rtr_weekly', 'sqrt_copy_cnt',
                          "request_hour", 'request_day'],
        aggregator="max",
    )

    


send_mio = DataReaderFlow(name="send_mio") \
    .count_reco_result(save_count_to="send_sample_cnt") \
    .perflog_attr_value(
        check_point="wht_test.offline_test.cofea_reader_online_for_v4",
        common_attrs=['send_sample_cnt'],
    ) \
    .log_debug_info(
        for_debug_request_only=False,
        respect_sample_logging=True,
        print_all_common_attrs=True,
        print_all_item_attrs=True
    ) \
    .send_to_mio_learner(
        # 在这里就会丢失photo hash的int64精度，mio不支持int64
        attrs=labels + ["comment_genre", "recall_type", "sample_weight",
                        ],
        slots_attrs=["comment_common_slots", "comment_item_slots"],
        signs_attrs=["comment_common_signs", "comment_item_signs"],
        lineid_attr="user_id",
        time_ms_attr="time_ms",
        label_attr="likeAction_first",
        user_hash_attr="user_hash"
    )

flows = [read_data, gen_feature, send_mio]
runner = OfflineRunner("comment_model")
runner.IGNORE_UNUSED_ATTR=['llsid']  + ['photo_upload_time', 'photo_hash']
runner.add_leaf_flows(leaf_flows=flows)
runner.build(output_file=os.path.join(current_dir, "cofea_reader.json"))
