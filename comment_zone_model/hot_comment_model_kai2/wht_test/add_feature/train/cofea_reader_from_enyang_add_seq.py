""" 继承自 beyyyyyy_add_feature/train/cofea_reader_online_no_seq.py
    一点一点往上加特征！
    目前已添加：request_time/city/content_segments
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

SEQ_LEN = 20

# 要从 kafka 中读取的特征
common_attrs=["user_id", "device_id", "photo_id", "photo_author_id", "gender", "age_segment", "llsid",
              "mod", "client_request_info", "page_type_str", "is_political", "product_name", "photo_upload_time",
            #   "city_name", 
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
            # "comment_content_segs_v", "new_sample_weight", 
            "expandAction_first", "replyAction_first", "likeAction_first", "audienceAction_first", "reportAction_first",
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
unused_common_attrs=['client_request_info', 'is_political', 'llsid', 'mod', 'page_type_str', 
                     'photo_author_high_score', 'photo_author_low_score', 'photo_click_count', 'photo_comment_count', 
                     'photo_comment_stay_time_sum_ms', 'photo_duration_ms', 'photo_effective_play_count', 'photo_follow_count', 
                     'photo_forward_count', 'photo_hetuV1_level1_tag_id', 'photo_like_count', 'photo_long_play_count', 
                     'photo_real_show', 'photo_recommend_count', 'photo_short_play_count', 'photo_upload_time', 
                     'photo_upload_type', 'photo_view_length_sum', 'product_name']
unused_item_attrs=['auto_expand', 'before_rerank_seq', 'cancelHateAction_first', 'cancelHateAction_second', 'cancelLikeAction_first', 
                   'cancelLikeAction_second', 'comment_genre', 'content_length', 'content_segment_num', 'copy_cnt', 'expand_cnt', 
                   'first_level_like_cnt', 'first_like_cnt', 'has_emoji', 'has_pic', 'hateAction_first', 'hateAction_second', 
                   'inform_cnt', 'is_ai_kwai_wonderful_rely', 'is_ai_play', 'is_comment_contain_at', 'is_text_emoji', 
                   'is_text_pic', 'like_cnt_weekly', 'mmu_category_tag', 'mmu_emotion_tag', 'mmu_entity_list_v', 
                   'predict_like_score', 'predict_reply_score', 'quality_score', 'quality_v2_score', 'recall_type', 
                   'related_score', 'replyTaskAction_first', 'replyTaskAction_second', 'reply_cnt_weekly', 'reportAction_second', 
                   'risk_inactive_tag', 'risk_insult_tag', 'risk_negative_tag', 'show_cnt_weekly', 'subAtAction_first', 
                   'subAtAction_second', 'subShowCntAction', 'sub_like_cnt']

common_attrs = [attr for attr in common_attrs if attr not in unused_common_attrs]
item_attrs = [attr for attr in item_attrs if attr not in unused_item_attrs]



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
    

# gen_new_feature = DataReaderFlow(name="gen_new_feature") \
#     .enrich_attr_by_lua(
#         import_common_attr = ['request_time'],
#         function_for_common = "calc_hour_day",
#         export_common_attr = ["request_hour", 'request_day'],
#         lua_script = """
#             function calc_hour_day()
#                 if request_time == nil then
#                     return -1, -1
#                 end
#                 local SECONDS_PER_DAY = 24 * 60 * 60
#                 local SECONDS_PER_HOUR = 60 * 60
#                 local UTC_OFFSET = 8 * SECONDS_PER_HOUR     -- utc+8时区

#                 -- 将毫秒转换为秒
#                 local timestamp_s = math.floor(request_time / 1000) + UTC_OFFSET
                
#                 -- 计算当天已经过去的秒数
#                 local seconds_today = timestamp_s % (24 * 60 * 60)
                
#                 -- 计算小时数
#                 local hour = math.floor(seconds_today / 3600)
                
#                 -- 计算自1970年1月1日以来的天数
#                 -- 1970年1月1日是星期四，所以加4
#                 local days_since_epoch = math.floor(timestamp_s / (24 * 60 * 60)) + 4
                
#                 -- 0-6，0是星期日，1是星期一
#                 local day_of_week = days_since_epoch % 7

#                 return hour, day_of_week
#             end
#         """
#     ) \
#     .split_string(
#         input_item_attr="comment_content_segs_v",
#         output_item_attr="comment_content_segs",
#         delimiters="_",
#     ) \
#     .enrich_attr_by_lua(
#         import_item_attr=["comment_content_segs"],
#         export_item_attr=["comment_content_segs"],
#         function_for_item="add_cls",
#         lua_script="""
#             function add_cls()
#                 local added = {}
#                 added[1] = "[cls]"
#                 for i = 1, #(comment_content_segs or {}) do  
#                     added[i+1] = comment_content_segs[i]
#                 end
#                 return added
#             end
#         """
#     ) \
#     .enrich_attr_by_lua(
#         import_item_attr=["comment_content_segs"],
#         export_item_attr=["mask_pack"],
#         function_for_item="seg_num",
#         lua_script=f"""
#             function seg_num()
#                 local final_seg_num = math.min({SEQ_LEN}, #(comment_content_segs or {{}}))
                
#                 local mask_pack = {{}}
#                 for i=1, final_seg_num do
#                     mask_pack[i] = 1
#                 end
#                 while #mask_pack < {SEQ_LEN} do
#                     table.insert(mask_pack, 0)
#                 end
#                 return mask_pack
#             end
#         """
#     ) \
#     .set_attr_value(
#         item_attrs=[
#             {
#             "name": "pos_ids",
#             "type": "int_list",
#             "value": list(range(SEQ_LEN))
#             },
#         ]
#     ) \


gen_kgnn_fea = DataReaderFlow(name="gen_kgnn_fea") \
    .fetch_kgnn_neighbors(
        id_from_common_attr="user_id",
        save_weight_to="comment_weights",  # like + reply
        save_neighbors_to="comment_ids",
        edge_attr_schema=NodeAttrSchema(1, 0).add_int64_list_attr("comment_mmu_categories", 1),
        kess_service="grpc_kgnn_user_interact_comment_info-U2I",
        relation_name='U2I',
        shard_num=4,
        sample_num=SEQ_LEN,
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
    .enrich_attr_by_lua(
        import_common_attr=["comment_ids"],
        export_common_attr=["mask_pack", "user_seq_len", "user_seq_coverage"],
        function_for_common="gen_user_seq_mask",
        lua_script=f"""
            function gen_user_seq_mask()
                local final_seg_num = math.min({SEQ_LEN}, #(comment_ids or {{}}))

                local mask_pack = {{}}
                for i=1, final_seg_num do
                    mask_pack[i] = 1
                end
                while #mask_pack < {SEQ_LEN} do
                    table.insert(mask_pack, 0)
                end

                local user_seq_coverage=0
                if #(comment_ids or {{}}) > 0 then
                    user_seq_coverage=1
                end
                return mask_pack, #(comment_ids or {{}}), user_seq_coverage
            end
        """
    ) \
    # .set_attr_value(
    #     item_attrs=[
    #         {
    #         "name": "pos_ids",
    #         "type": "int_list",
    #         "value": list(range(SEQ_LEN+1))
    #         },
    #     ]
    # ) \
    

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

            # ## new feature
            # **id_config("city_name", 114),          # dim=32
            # **id_config("request_hour", 115),       # dim=8
            # **id_config("request_day", 116),    

            # 序列特征
            **list_config("comment_ids", 300, 201, SEQ_LEN),
            # **list_config("pos_ids", 301, 301, SEQ_LEN+1),
            # "comment_weights_int": {"attrs": [{"key_type": 301, "attr": ["comment_weights_int"], **kuiba_list_converter_config_list_limit(14)}]},
            # **list_config("comment_mmu_categories", 302, 255, 14),

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
            # **list_config("comment_content_segs", 300, 300, SEQ_LEN),    
            # **list_config("pos_ids", 301, 301, SEQ_LEN),

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
        item_attrs=["sample_weight"] + labels,
        common_attrs=["send_sample_cnt", "user_seq_len", 'user_seq_coverage']
    ) \
    .send_to_mio_learner(
        attrs=labels + ['mask_pack'],
        slots_attrs=["comment_common_slots", "comment_item_slots"],
        signs_attrs=["comment_common_signs", "comment_item_signs"],
        lineid_attr="user_id",
        time_ms_attr="time_ms",
        label_attr="likeAction_first",
        user_hash_attr="user_hash"
    )

flows = [read_data, gen_kgnn_fea, gen_feature, send_mio]
runner = OfflineRunner("comment_profile")
# runner.CHECK_UNUSED_ATTR=False        # 这个不要设置为False，否则不便于发现特征缺失
runner.IGNORE_UNUSED_ATTR=['comment_mmu_categories']  + ['comment_weights_int']
runner.add_leaf_flows(leaf_flows=flows)
runner.build(output_file=os.path.join(current_dir, "cofea_reader.json"))
