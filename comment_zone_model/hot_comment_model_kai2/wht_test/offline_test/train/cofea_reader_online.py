""" 
和 cofea_reader.py 的区别在于，是从 kafka 中读取数据，而不是从离线文件中读取
kafka数据源：universe_feature/.../comment/xtr/pipeline.py (send_topic)
如果要增加特征，需要修改universe_feature

TODO
- 增加 地理特征：country_name\province_name\city_name TODO
- 作者是否赞过
- 看评人与写评人的社交关系
- 行为序列
- 来源页
"""

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


# 要从 kafka 中读取的特征
common_attrs=["user_id", "device_id", "photo_id", "photo_author_id", "gender", "age_segment", "llsid", "mod",
              "page_type_str", "is_political", "product_name"]
labels = [
    "showAction", "expandAction", "replyAction", "likeAction", "audienceAction", "reportAction",
    "expandAction_first", "replyAction_first", "likeAction_first", "audienceAction_first", "reportAction_first",
    "expandAction_second", "replyAction_second", "likeAction_second", "audienceAction_second", "reportAction_second",
    "copyAction", "copyAction_first", "copyAction_second",
    "shareAction", "shareAction_first", "shareAction_second",

    "cancelLikeAction_first", "cancelLikeAction_second",
    "hateAction_first", "hateAction_second", "cancelHateAction_first", "cancelHateAction_second",
    "replyTaskAction_first", "replyTaskAction_second", "subAtAction_first", "subAtAction_second"
]
item_attrs=['comment_id', 'author_id', 'like_cnt', 'reply_cnt', 'dislike_cnt', 'realshow_cnt', 'minute_diff', 
            'comment_genre', 'risk_insult_tag', 'risk_negative_tag', 'risk_inactive_tag', 'mmu_category_tag', 
            'mmu_emotion_tag', 'quality_v2_score', 'quality_score', 'related_score',
            'show_cnt_weekly', 'like_cnt_weekly', 'reply_cnt_weekly', 'auto_expand', 'first_like_cnt',
            'sub_like_cnt', 'first_level_like_cnt', 
            # 'comment_content_segs_v',  'mmu_entity_list_v', 
            'content_length', 'content_segment_num', 'inform_cnt', 'copy_cnt',
            'has_pic', 'has_emoji', 'is_text_pic', 'is_text_emoji', 'is_ai_play', 'is_ai_kwai_wonderful_rely', 'is_comment_contain_at',
            'recall_type'   # recall_type用于区分爬评热评
            ]

# 没有发过来的item attrs：
# mmu_entity_list_v（发的是mmu_entity_list）
# comment_content_segs_v（发的是comment_content_segs）,
# recall_type(特征不需要)


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
    


set_default_value_int_attrs_for_item = labels + ['comment_id', 'author_id', 'like_cnt', 'reply_cnt', 'dislike_cnt', 'realshow_cnt',
            'comment_genre', 'risk_insult_tag', 'risk_negative_tag', 'risk_inactive_tag', 'mmu_category_tag', 
            'mmu_emotion_tag', 'quality_v2_score', 'quality_score', 'related_score',
            'show_cnt_weekly', 'like_cnt_weekly', 'reply_cnt_weekly', 'auto_expand', 'first_like_cnt',
            'sub_like_cnt', 'first_level_like_cnt', 'content_length', 'content_segment_num', 'inform_cnt', 'copy_cnt',
            'has_pic', 'has_emoji', 'is_text_pic', 'is_text_emoji', 'is_ai_play', 'is_ai_kwai_wonderful_rely', 'is_comment_contain_at',
            ]
set_default_value_double_attrs_for_item = ['minute_diff', 'related_score', 'quality_score', 'quality_v2_score']
set_default_value_int_list_attrs_for_item = []
set_default_value_str_list_attrs_for_item = []

default_value_map_list_for_item = []
for name in set_default_value_int_attrs_for_item:
    default_value_map_list_for_item.append({"name": name, "type": "int", "value": 0})
for name in set_default_value_double_attrs_for_item:
    default_value_map_list_for_item.append({"name": name, "type": "double", "value": 0.0})
for name in set_default_value_int_list_attrs_for_item:
    default_value_map_list_for_item.append({"name": name, "type": "int_list", "value": []})
for name in set_default_value_str_list_attrs_for_item:
    default_value_map_list_for_item.append({"name": name, "type": "str_list", "value": []})


set_default_value_int_attrs_for_common = ['user_id', 'photo_id', 'photo_author_id', 'gender', 'age_segment', 'is_political', ]
set_default_value_double_attrs_for_common = []
set_default_value_str_attrs_for_common = ['device_id', 'llsid', 'mod', 'page_type_str', 'product_name']

default_value_map_list_for_common = []
for name in set_default_value_int_attrs_for_common:
    default_value_map_list_for_common.append({"name": name, "type": "int", "value": 0})
for name in set_default_value_double_attrs_for_common:
    default_value_map_list_for_common.append({"name": name, "type": "double", "value": 0.0})
for name in set_default_value_str_attrs_for_common:
    default_value_map_list_for_common.append({"name": name, "type": "string", "value": ''})



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
    # .log_debug_info(
    #     common_attrs=["user_id", "comment_ids", "comment_mmu_categories", "comment_weights", "comment_weights_int"],
    #     for_debug_request_only=False,
    #     respect_sample_logging=True,
    # )


gen_feature = DataReaderFlow(name="gen_feature") \
    .enrich_attr_by_lua(
        import_common_attr=["request_time"],
        export_common_attr=["time_ms"],
        function_for_common="cal",
        lua_script="""
            function cal()
                local time_ms = request_time // 1000
                return time_ms
            end
        """
    ) \
    .set_attr_value(
        no_overwrite=True,
        common_attrs=default_value_map_list_for_common,
        item_attrs=default_value_map_list_for_item
    ) \
    .enrich_attr_by_lua(
        import_item_attr=["like_cnt", "reply_cnt", "dislike_cnt", "realshow_cnt", 'first_like_cnt', 'sub_like_cnt', 
                          'first_level_like_cnt', "show_cnt_weekly", "like_cnt_weekly", "reply_cnt_weekly",
                          "copy_cnt", "minute_diff"],
        export_item_attr=["ltr", "rtr", "dtr", "sqrt_like_cnt", "sqrt_reply_cnt", 'sqrt_dislike_cnt', 
                          'sqrt_hour_diff', 'sqrt_sub_like_cnt', 'sqrt_first_level_like_cnt', 'sqrt_first_like_cnt',
                          'dislike_like_ratio', 'sub_root_like_ratio', 'ltr_weekly', 'rtr_weekly', 'sqrt_copy_cnt',
                          'ltr_copy', 'rtr_copy'],
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
            **id_config("product_name", 113),   # dim=4也够了
            
            **id_config("is_political", 112),     # dim=4
            
            # 序列特征
            **list_config("comment_ids", 300, 201, 20),
                "comment_weights_int": {"attrs": [{"key_type": 301, "attr": ["comment_weights_int"], **kuiba_list_converter_config_list_limit(20)}]},
            **list_config("comment_mmu_categories", 302, 255, 20),
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
    .log_debug_info(
        print_all_common_attrs = True,
        print_all_item_attrs = True,
        for_debug_request_only = False,
        respect_sample_logging = True,
    ) \
    .perflog_attr_value(
        check_point="wht_test.add_feature.cofea_reader_online",
        item_attrs=set_default_value_int_attrs_for_item + set_default_value_double_attrs_for_item + ["ltr", "rtr", "dtr", "sqrt_like_cnt", "sqrt_reply_cnt", 'sqrt_dislike_cnt', 
                          'sqrt_hour_diff', 'sqrt_sub_like_cnt', 'sqrt_first_level_like_cnt', 'sqrt_first_like_cnt',
                          'dislike_like_ratio', 'sub_root_like_ratio', 'ltr_weekly', 'rtr_weekly', 'sqrt_copy_cnt'],
        aggregator="max",
    )

    

send_mio = DataReaderFlow(name="send_mio") \
    .count_reco_result(save_count_to="send_sample_cnt") \
    .perflog_attr_value(
        check_point="wht_test.add_feature.cofea_reader_online",
        common_attrs=["send_sample_cnt", "user_hash", "photo_hash"]
    ) \
    .send_to_mio_learner(
        # 在这里就会丢失photo hash的int64精度，mio不支持int64
        attrs=labels + ["sample_weight", "realshow_cnt", "comment_genre", "recall_type", 
                        # "time_ms", 'user_id'
                        ],
        slots_attrs=["comment_common_slots", "comment_item_slots"],
        signs_attrs=["comment_common_signs", "comment_item_signs"],
        lineid_attr="user_id",
        time_ms_attr="time_ms",
        label_attr="likeAction_first",
        user_hash_attr="user_hash"
    )

flows = [read_data, gen_kgnn_fea, gen_feature, send_mio]
runner = OfflineRunner("comment_profile")
runner.IGNORE_UNUSED_ATTR=['llsid']
runner.add_leaf_flows(leaf_flows=flows)
runner.build(output_file=os.path.join(current_dir, "cofea_reader.json"))
