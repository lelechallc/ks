"""
从hive表取训练数据，同时标记需要dump的数据，然后在model.py中对标记的数据导出
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

selected_cids=[891420674333, 891593752581, 891615628127, 891532606507, 891490719586, 891495754923, 891650705571, 891404774889, 891468935138, 891523960760, 891620275108, 891568762432, 891423541405, 891490941264, 891617265510, 891623393610, 891397721846, 891601523735, 891547178370, 891520050304, 891568089311]

common_attrs=["user_id", "device_id", "photo_id", "photo_author_id", "gender", "age_segment", "llsid",
            # "mod", "page_type_str", "is_political", "product_name", "photo_upload_time", "request_time", "city_name"
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

    'hateaction_first',
    'hateaction_second',
    'cancelhateaction_first',
    'cancelhateaction_second',
    'cancellikeaction_first',
    'cancellikeaction_second',
]
item_attrs=['comment_id', 'author_id', 'like_cnt', 'reply_cnt', 'dislike_cnt', 'realshow_cnt', 'minute_diff', 
            # 'comment_genre', 'risk_insult_tag', 'risk_negative_tag', 'risk_inactive_tag', 'mmu_category_tag', 
            # 'mmu_emotion_tag', 'quality_v2_score', 'quality_score', 'related_score',
            # 'show_cnt_weekly', 'like_cnt_weekly', 'reply_cnt_weekly', 'auto_expand', 'first_like_cnt',
            # 'sub_like_cnt', 'first_level_like_cnt', 
            # 'content_length', 'content_segment_num', 'inform_cnt', 'copy_cnt',
            # 'has_pic', 'has_emoji', 'is_text_pic', 'is_text_emoji', 'is_ai_play', 'is_ai_kwai_wonderful_rely', 'is_comment_contain_at',
            # 'recall_type'   # recall_type用于区分爬评热评
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
    'comment_content_segs': (79, 'string_list'),
}
read_data = DataReaderFlow(name="read_data") \
    .fetch_message(
        group_id="wht001",
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
        for_debug_request_only=True,
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
    elif name in item_attrs+labels:
        default_value_map_list_for_item.append({"name": name, "type": "int", "value": 0})
for name in set_default_value_double_attrs:
    if name in common_attrs:
        default_value_map_list_for_common.append({"name": name, "type": "double", "value": 0.0})
    elif name in item_attrs+labels:  
        default_value_map_list_for_item.append({"name": name, "type": "double", "value": 0.0})
for name in set_default_value_string_attrs:
    if name in common_attrs:
        default_value_map_list_for_common.append({"name": name, "type": "string", "value": ''})
    elif name in item_attrs+labels:
        default_value_map_list_for_item.append({"name": name, "type": "string", "value": ''})


gen_feature = DataReaderFlow(name="gen_feature") \
    .set_attr_value(
        common_attrs=default_value_map_list_for_common,
        item_attrs=default_value_map_list_for_item,
        no_overwrite=True
    ) \
    .count_item_attr(
        counters = [{
            "check_attr_name": "comment_id",
            "output_attr_name": "is_select_cid",
            "check_values": selected_cids,
            "max_count": 1
        }]
    ) \
    .enrich_attr_by_lua(
        import_item_attr=["like_cnt", "reply_cnt", "realshow_cnt", "comment_id"],
        export_item_attr=["ltr", "rtr", "dummy_cid"],
        function_for_item="cal_xtr",
        lua_script="""
            function cal_xtr()
            local ltr = like_cnt / (realshow_cnt + 1.0)
            local rtr = reply_cnt / (realshow_cnt + 1.0)
            local dummy_cid = comment_id % 10000
            return ltr, rtr, dummy_cid
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
    .perflog_attr_value(
        check_point="wht_test.dump_data.cofea_reader_online",
        item_attrs=['is_select_cid'],
        # aggregator="max",
        # select_item = {
        #     "attr_name": "comment_id",
        #     "compare_to": selected_cids,
        #     "select_if": "in",
        #     "select_if_attr_missing": False
        # }
    )
    # .enrich_attr_by_lua(
    #     import_common_attr=["photo_id", "age_segment", "gender"],
    #     function_for_common="get_photo_hash",
    #     export_common_attr=["photo_hash"],
    #     lua_script="""
    #         function get_photo_hash()
    #             local photo_hash_str = tostring(photo_id) .. tostring(age_segment) .. tostring(gender)
    #             return util.CityHash64(photo_hash_str)
    #         end
    #     """
    # ) \
    

    


send_mio = DataReaderFlow(name="send_mio") \
    .count_reco_result(save_count_to="send_sample_cnt") \
    .perflog_attr_value(
        check_point="wht_test.dump_data.cofea_reader_online",
        common_attrs=['send_sample_cnt'],
    ) \
    .log_debug_info(
        for_debug_request_only=False,
        respect_sample_logging=True,
        print_all_common_attrs=False,
        print_all_item_attrs=True,
        trace_item_keys=selected_cids,
    ) \
    .send_to_mio_learner(
        # 在这里就会丢失photo hash的int64精度，mio不支持int64
        attrs=labels + ["sample_weight", 'dummy_cid', 'is_select_cid', 'comment_id'],
        slots_attrs=["comment_common_slots", "comment_item_slots"],
        signs_attrs=["comment_common_signs", "comment_item_signs"],
        lineid_attr="user_id",
        time_ms_attr="time_ms",
        label_attr="likeAction_first",
        user_hash_attr="user_photo_hash"
    )

flows = [read_data, gen_feature, send_mio]
runner = OfflineRunner("comment_model")
runner.IGNORE_UNUSED_ATTR=['llsid']
runner.add_leaf_flows(leaf_flows=flows)
runner.build(output_file=os.path.join(current_dir, "cofea_reader.json"))
