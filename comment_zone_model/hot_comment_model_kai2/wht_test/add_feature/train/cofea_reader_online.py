""" copy自 beyyyyyy_add_feature/train/cofea_reader_online_no_seq.py
    一点一点往上加特征！
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

# 务必逐项确认这些特征确实不需要
unused_common_attrs=['client_request_info', 'llsid', 'city_name',
                     'photo_author_high_score', 'photo_author_low_score', 'photo_click_count', 'photo_comment_count', 
                     'photo_comment_stay_time_sum_ms', 'photo_duration_ms', 'photo_effective_play_count', 'photo_follow_count', 
                     'photo_forward_count', 'photo_hetuV1_level1_tag_id', 'photo_like_count', 'photo_long_play_count', 
                     'photo_real_show', 'photo_recommend_count', 'photo_short_play_count', 'photo_upload_time', 
                     'photo_upload_type', 'photo_view_length_sum']
unused_item_attrs=['before_rerank_seq', 'cancelHateAction_first', 'cancelHateAction_second', 'cancelLikeAction_first', 
                   'cancelLikeAction_second', 'expand_cnt', 
                   'hateAction_first', 'hateAction_second', 
                   'mmu_entity_list_v', 'sample_weight',
                   'predict_like_score', 'predict_reply_score', 'quality_score', 'quality_v2_score', 'recall_type', 
                   'replyTaskAction_first', 'replyTaskAction_second', 'reportAction_second', 
                   'risk_inactive_tag', 'risk_insult_tag', 'risk_negative_tag', 'subAtAction_first', 
                   'subAtAction_second', 'subShowCntAction',
                   'audienceAction', 'expandAction', 'likeAction', 'replyAction', 'reportAction']

common_attrs = [attr for attr in common_attrs if attr not in unused_common_attrs]
item_attrs = [attr for attr in item_attrs if attr not in unused_item_attrs]


set_default_value_int_attrs_for_item = labels + ['like_cnt', 'reply_cnt', 'dislike_cnt', 'realshow_cnt',
            'show_cnt_weekly', 'like_cnt_weekly', 'reply_cnt_weekly', 'auto_expand', 'first_like_cnt',
            'sub_like_cnt', 'first_level_like_cnt', 'content_length', 'content_segment_num', 'inform_cnt', 'copy_cnt',
            'has_pic', 'has_emoji', 'is_text_pic', 'is_text_emoji', 'is_ai_play', 'is_ai_kwai_wonderful_rely', 'is_comment_contain_at',
            ]
set_default_value_double_attrs_for_item = ['minute_diff', 'related_score']
set_default_value_int_list_attrs_for_item = []
set_default_value_str_list_attrs_for_item = []
set_default_value_str_attrs_for_item = ['comment_content_segs_v']


default_value_map_list_for_item = []
for name in set_default_value_int_attrs_for_item:
    default_value_map_list_for_item.append({"name": name, "type": "int", "value": 0})
for name in set_default_value_double_attrs_for_item:
    default_value_map_list_for_item.append({"name": name, "type": "double", "value": 0.0})
for name in set_default_value_int_list_attrs_for_item:
    default_value_map_list_for_item.append({"name": name, "type": "int_list", "value": []})
for name in set_default_value_str_list_attrs_for_item:
    default_value_map_list_for_item.append({"name": name, "type": "str_list", "value": []})
for name in set_default_value_str_attrs_for_item:
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
    .set_attr_value(
        no_overwrite=True,
        item_attrs=default_value_map_list_for_item
    ) \
    

gen_new_feature = DataReaderFlow(name="gen_new_feature") \
    .split_string(
        input_item_attr="comment_content_segs_v",
        output_item_attr="comment_content_segs",
        delimiters="_",
    ) \
    .enrich_attr_by_lua(
        import_item_attr=["comment_content_segs"],
        export_item_attr=["comment_content_segs"],
        function_for_item="add_cls",
        lua_script="""
            function add_cls()
                local added = {}
                added[1] = "[cls]"
                for i = 1, #(comment_content_segs or {}) do  
                    added[i+1] = comment_content_segs[i]
                end
                return added
            end
        """
    ) \
    .enrich_attr_by_lua(
        import_item_attr=["comment_content_segs"],
        export_item_attr=["mask_pack"],
        function_for_item="seg_num",
        lua_script=f"""
            function seg_num()
                local final_seg_num = math.min({SEQ_LEN}, #(comment_content_segs or {{}}))
                
                local mask_pack = {{}}
                for i=1, final_seg_num do
                    mask_pack[i] = 1
                end
                while #mask_pack < {SEQ_LEN} do
                    table.insert(mask_pack, 0)
                end
                return mask_pack
            end
        """
    ) \
    .set_attr_value(
        item_attrs=[
            {
            "name": "pos_ids",
            "type": "int_list",
            "value": list(range(SEQ_LEN))
            },
        ]
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
    

gen_feature = DataReaderFlow(name="gen_feature") \
    .extract_kuiba_parameter(
        config={
            **id_config("gender", 101),     # dim=4
            **id_config("age_segment", 102),
        
            **id_config("photo_id", 103),   # dim=64
            **id_config_slot("photo_author_id", 104, 202),
            **id_config_slot("user_id", 105, 202),
            **id_config("device_id", 106),

            ## new feature
            **id_config("mod", 110),        # dim=32
            **id_config("page_type_str", 111),  
            **id_config("is_political", 112),       # dim=4
            **id_config("product_name", 113),  

            # **id_config("city_name", 114),          # dim=32
            # **id_config("request_hour", 115),       # dim=8
            # **id_config("request_day", 116),    

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

            ## new feature
            # 以下是新特征  
            **id_config("comment_genre", 250),      # dim=8
            # **id_config("risk_insult_tag", 251),    
            # **id_config("risk_inactive_tag", 252),
            # **id_config("risk_negative_tag", 253),
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

            # **discreate_config("quality_score", 278, [0.01, 0, 100, 1, 0]), # dim=8
            **discreate_config("related_score", 279, [0.01, 0, 100, 1, 0]),
            # **discreate_config("quality_v2_score", 280, [0.01, 0, 100, 1, 0]),
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

            **list_config("comment_content_segs", 300, 300, SEQ_LEN),    
            **list_config("pos_ids", 301, 301, SEQ_LEN),

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

    

send_mio = DataReaderFlow(name="send_mio") \
    .count_reco_result(save_count_to="send_sample_cnt") \
    .perflog_attr_value(
        check_point="send.mio",
        item_attrs=["new_sample_weight"] + labels,
        common_attrs=["send_sample_cnt"]
    ) \
    .send_to_mio_learner(
        attrs=labels + ['mask_pack', 'new_sample_weight'],
        slots_attrs=["comment_common_slots", "comment_item_slots"],
        signs_attrs=["comment_common_signs", "comment_item_signs"],
        lineid_attr="user_id",
        time_ms_attr="time_ms",
        label_attr="likeAction_first",
        user_hash_attr="user_hash"
    )

flows = [read_data, gen_new_feature, gen_feature, send_mio]
runner = OfflineRunner("comment_profile")
# runner.CHECK_UNUSED_ATTR=False        # 这个不要设置为False，否则不便于发现特征缺失
runner.add_leaf_flows(leaf_flows=flows)
runner.build(output_file=os.path.join(current_dir, "cofea_reader.json"))
