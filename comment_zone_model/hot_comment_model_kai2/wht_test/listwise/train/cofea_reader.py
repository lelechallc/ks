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

SEQ_SIZE = 180

class DataReaderFlow(LeafFlow, MioApiMixin, OfflineApiMixin, CofeaApiMixin, KuibaApiMixin, KgnnApiMixin):
    def clean_all(self, reason, **kwargs):
        return self.limit(0, name="clean_all_for_" + reason, **kwargs)


# 要从 kafka 中读取的特征
common_attrs=["user_id", "device_id", "photo_id", "photo_author_id", "gender", "age_segment", "llsid", 
              # "mod", "page_type_str", "is_political", "product_name",
              ]
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
# available_item_attrs=['comment_id', 'author_id', 'like_cnt', 'reply_cnt', 'dislike_cnt', 'realshow_cnt', 'minute_diff', 
#             'comment_genre', 'risk_insult_tag', 'risk_negative_tag', 'risk_inactive_tag', 'mmu_category_tag', 
#             'mmu_emotion_tag', 'quality_v2_score', 'quality_score', 'related_score',
#             'show_cnt_weekly', 'like_cnt_weekly', 'reply_cnt_weekly', 'auto_expand', 'first_like_cnt',
#             'sub_like_cnt', 'first_level_like_cnt', 
#             # 'comment_content_segs_v',  'mmu_entity_list_v', 
#             'content_length', 'content_segment_num', 'inform_cnt', 'copy_cnt',
#             'has_pic', 'has_emoji', 'is_text_pic', 'is_text_emoji', 'is_ai_play', 'is_ai_kwai_wonderful_rely', 'is_comment_contain_at',
#             'recall_type'   # recall_type用于区分爬评热评
#             ]

# 没有发过来的item attrs：
# mmu_entity_list_v（发的是mmu_entity_list）
# comment_content_segs_v（发的是comment_content_segs）,
# recall_type(特征不需要)


used_item_attrs=['comment_id', 'author_id', 'like_cnt', 'reply_cnt', 'dislike_cnt', 'realshow_cnt', 'minute_diff', 
                'comment_genre', 'risk_insult_tag', 'risk_negative_tag', 'risk_inactive_tag', 'mmu_category_tag', 'mmu_emotion_tag',
                ]   
item_attrs = used_item_attrs + labels


set_default_value_int_attrs_for_item = labels + ['comment_id', 'author_id', 'like_cnt', 'reply_cnt', 'dislike_cnt', 'realshow_cnt',
            'comment_genre', 'risk_insult_tag', 'risk_negative_tag', 'risk_inactive_tag', 'mmu_category_tag', 'mmu_emotion_tag', 
            # 'quality_v2_score', 'quality_score', 'related_score',
            # 'show_cnt_weekly', 'like_cnt_weekly', 'reply_cnt_weekly', 'auto_expand', 'first_like_cnt',
            # 'sub_like_cnt', 'first_level_like_cnt', 'content_length', 'content_segment_num', 'inform_cnt', 'copy_cnt',
            # 'has_pic', 'has_emoji', 'is_text_pic', 'is_text_emoji', 'is_ai_play', 'is_ai_kwai_wonderful_rely', 'is_comment_contain_at',
            ]
set_default_value_double_attrs_for_item = ['minute_diff', 
                                        #    'related_score', 'quality_score', 'quality_v2_score',
                                        ]
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

set_default_value_int_attrs_for_common = ['user_id', 'photo_id', 'photo_author_id', 'gender', 'age_segment', ]
set_default_value_double_attrs_for_common = []
set_default_value_str_attrs_for_common = ['device_id', 'llsid', ]

default_value_map_list_for_common = []
for name in set_default_value_int_attrs_for_common:
    default_value_map_list_for_common.append({"name": name, "type": "int", "value": 0})
for name in set_default_value_double_attrs_for_common:
    default_value_map_list_for_common.append({"name": name, "type": "double", "value": 0.0})
for name in set_default_value_str_attrs_for_common:
    default_value_map_list_for_common.append({"name": name, "type": "string", "value": ''})


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
        save_count_to="retrieve_item_num"
    ) \
    .filter_by_attr(
        attr_name="showAction",
        remove_if="<=",
        compare_to=0,
        remove_if_attr_missing=True,
    ) \
    .count_reco_result(save_count_to="show_item_num") \
    .limit(SEQ_SIZE) \
    .count_reco_result(save_count_to="final_seq_len") \
    .perflog_attr_value(
        check_point="wht_test.listwise.cofea_reader",
        common_attrs=["retrieve_item_num", "show_item_num", "final_seq_len"],
    ) \
    .if_("final_seq_len <= 0") \
        .return_(0) \
    .end_() \



gen_feature = DataReaderFlow(name="gen_feature") \
    .set_attr_value(
        no_overwrite=True,
        common_attrs=default_value_map_list_for_common,
        item_attrs=default_value_map_list_for_item
    ) \
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
    .enrich_attr_by_lua(
        import_common_attr=["final_seq_len"],
        export_common_attr=["mask_pack"],
        function_for_common="gen_mask_pack",
        lua_script=f'''
            function gen_mask_pack()
                local mask_pack = {{}}
                for i=1, final_seq_len do
                    table.insert(mask_pack, 1)
                end
                while #mask_pack < {SEQ_SIZE} do
                    table.insert(mask_pack, 0)
                end
                return mask_pack
            end
        '''
    ) \
    .enrich_attr_by_lua(
        import_item_attr=["like_cnt", "reply_cnt", "dislike_cnt", "realshow_cnt", "minute_diff",
                        #   'first_like_cnt', 'sub_like_cnt', 'first_level_like_cnt', 
                        #   "show_cnt_weekly", "like_cnt_weekly", "reply_cnt_weekly",
                        #   "copy_cnt", 
                          ],
        export_item_attr=["ltr", "rtr", "dtr", "sqrt_like_cnt", "sqrt_reply_cnt", 'sqrt_dislike_cnt', 'sqrt_hour_diff', 
                        #   'sqrt_sub_like_cnt', 'sqrt_first_level_like_cnt', 'sqrt_first_like_cnt',
                        #   'dislike_like_ratio', 'sub_root_like_ratio', 'ltr_weekly', 'rtr_weekly', 'sqrt_copy_cnt',
                        #   'ltr_copy', 'rtr_copy'
                          ],
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
                return ltr, rtr, dtr, sqrt_like_cnt, sqrt_reply_cnt, sqrt_dislike_cnt, sqrt_hour_diff
            end
        """
    ) \
    .enrich_attr_by_lua(       
        import_item_attr=["like_cnt", "reply_cnt", "dislike_cnt", "minute_diff", "ltr", "rtr", "dtr", "sqrt_like_cnt", "sqrt_reply_cnt", 'sqrt_dislike_cnt', 'sqrt_hour_diff'],
        export_item_attr=["like_cnt_dis", "reply_cnt_dis", "dislike_cnt_dis", "minute_diff_dis", "ltr_dis", "rtr_dis", "dtr_dis",
                          "sqrt_like_cnt", "sqrt_reply_cnt", 'sqrt_dislike_cnt', 'sqrt_hour_diff',
                          ],
        function_for_item="discreate",
        lua_script='''
            function discreate()
                local like_cnt_dis = math.max(math.min(like_cnt // 5, 100000), 0) * 1
                local reply_cnt_dis = math.max(math.min(reply_cnt // 5, 100000), 0) * 1
                local dislike_cnt_dis = math.max(math.min(dislike_cnt // 3, 100000), 0) * 1
                local minute_diff_dis = math.max(math.min(minute_diff // 3, 14400), 0) * 1
                local ltr_dis = math.max(math.min(ltr * 1000, 100000), 0) * 137
                local rtr_dis = math.max(math.min(rtr * 1000, 100000), 0) * 137
                local dtr_dis = math.max(math.min(dtr * 100, 100), 0)
                return math.floor(like_cnt_dis), math.floor(reply_cnt_dis), math.floor(dislike_cnt_dis), math.floor(minute_diff_dis),
                       math.floor(ltr_dis), math.floor(rtr_dis), math.floor(dtr_dis), math.floor(sqrt_like_cnt), math.floor(sqrt_reply_cnt), math.floor(sqrt_dislike_cnt), math.floor(sqrt_hour_diff)
            end
        '''
    ) \
    .enrich_attr_by_lua(       
        import_item_attr=labels,
        export_item_attr=['like_label', 'reply_label', 'expand_label', 'continuous_expand_label', 'copy_label', 'share_label', 'audience_label'],
        function_for_item="generate_label",
        lua_script='''
            function generate_label()
                local like_label, reply_label, expand_label, continuous_expand_label, copy_label, share_label, audience_label = 0, 0, 0, 0, 0, 0, 0
                if likeAction_first>0 or likeAction_second>0 then
                    like_label = 1
                end
                if replyAction_first>0 or replyAction_second>0 then
                    reply_label = 1
                end
                if expandAction_first>0 then
                    expand_label = 1
                end
                if expandAction_first>1 then
                    continuous_expand_label = 1
                end
                if copyAction_first>0 or copyAction_second>0 then
                    copy_label = 1
                end
                if shareAction_first>0 or shareAction_second>0 then
                    share_label = 1
                end
                if audienceAction_first>0 or audienceAction_second>0 then
                    audience_label = 1
                end

                return like_label, reply_label, expand_label, continuous_expand_label, copy_label, share_label, audience_label
            end
        '''
    ) \
    .pack_item_attr(
        item_source={
            "reco_results": True,
        },
        mappings=[
            {
                "from_item_attr": "comment_id",
                "to_common_attr": "comment_id_list",
            },
            {
                "from_item_attr": "author_id",
                "to_common_attr": "author_id_list",
            },
            {
                "from_item_attr": "like_cnt_dis",
                "to_common_attr": "like_cnt_dis_list",
            },
            {
                "from_item_attr": "reply_cnt_dis",
                "to_common_attr": "reply_cnt_dis_list",
            },
            {
                "from_item_attr": "dislike_cnt_dis",
                "to_common_attr": "dislike_cnt_dis_list",
            },
            {
                "from_item_attr": "minute_diff_dis",
                "to_common_attr": "minute_diff_dis_list",
            },
            {
                "from_item_attr": "ltr_dis",
                "to_common_attr": "ltr_dis_list",
            },
            {
                "from_item_attr": "rtr_dis",
                "to_common_attr": "rtr_dis_list",
            },
            {
                "from_item_attr": "dtr_dis",
                "to_common_attr": "dtr_dis_list",
            },
            {
                "from_item_attr": "sqrt_like_cnt",
                "to_common_attr": "sqrt_like_cnt_list",
            },
            {
                "from_item_attr": "sqrt_reply_cnt",
                "to_common_attr": "sqrt_reply_cnt_list",
            },
            {
                "from_item_attr": "sqrt_dislike_cnt",
                "to_common_attr": "sqrt_dislike_cnt_list",
            },
            {
                "from_item_attr": "sqrt_hour_diff",
                "to_common_attr": "sqrt_hour_diff_list",
            },
            {
                "from_item_attr": "comment_genre",
                "to_common_attr": "comment_genre_list",
            },
            {
                "from_item_attr": "risk_insult_tag",
                "to_common_attr": "risk_insult_tag_list",
            },
            {
                "from_item_attr": "risk_negative_tag",
                "to_common_attr": "risk_negative_tag_list",
            },
            {
                "from_item_attr": "risk_inactive_tag",
                "to_common_attr": "risk_inactive_tag_list",
            },
            {
                "from_item_attr": "mmu_category_tag",
                "to_common_attr": "mmu_category_tag_list",
            },
            {
                "from_item_attr": "mmu_emotion_tag",
                "to_common_attr": "mmu_emotion_tag_list",
            },
            {
                "from_item_attr": "like_label",
                "to_common_attr": "like_label_list",
            },
            {
                "from_item_attr": "reply_label",
                "to_common_attr": "reply_label_list",
            },
            {
                "from_item_attr": "expand_label",
                "to_common_attr": "expand_label_list",
            },
            {
                "from_item_attr": "continuous_expand_label",
                "to_common_attr": "continuous_expand_label_list",
            },
            {
                "from_item_attr": "copy_label",
                "to_common_attr": "copy_label_list",
            },
            {
                "from_item_attr": "share_label",
                "to_common_attr": "share_label_list",
            },
            {
                "from_item_attr": "audience_label",
                "to_common_attr": "audience_label_list",
            },
        ]
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
            # **id_config("mod", 110),  
            # **id_config("page_type_str", 111),  
            # **id_config("product_name", 113),     # dim=4
            # **id_config("is_political", 112),     # dim=4

            **list_config("comment_id_list", 301, 301, SEQ_SIZE),
            **list_config("author_id_list", 302, 302, SEQ_SIZE),

            **list_config("like_cnt_dis_list", 303, 303, SEQ_SIZE),
            **list_config("reply_cnt_dis_list", 304, 304, SEQ_SIZE),
            **list_config("minute_diff_dis_list", 305, 305, SEQ_SIZE),
            **list_config("ltr_dis_list", 306, 306, SEQ_SIZE),
            **list_config("rtr_dis_list", 307, 307, SEQ_SIZE),
            **list_config("dtr_dis_list", 308, 308, SEQ_SIZE),
            **list_config("dislike_cnt_dis_list", 309, 309, SEQ_SIZE),
            **list_config("sqrt_like_cnt_list", 310, 310, SEQ_SIZE),
            **list_config("sqrt_reply_cnt_list", 311, 311, SEQ_SIZE),
            **list_config("sqrt_dislike_cnt_list", 312, 312, SEQ_SIZE),
            **list_config("sqrt_hour_diff_list", 313, 313, SEQ_SIZE),

            **list_config("comment_genre_list", 250, 250, SEQ_SIZE),
            **list_config("risk_insult_tag_list", 251, 251, SEQ_SIZE),
            **list_config("risk_negative_tag_list", 252, 252, SEQ_SIZE),
            **list_config("risk_inactive_tag_list", 253, 253, SEQ_SIZE),
            **list_config("mmu_category_tag_list", 254, 254, SEQ_SIZE),
            **list_config("mmu_emotion_tag_list", 255, 255, SEQ_SIZE),
        },
        is_common_attr=True,
        slots_output="comment_common_slots",
        parameters_output="comment_common_signs",
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
        check_point="wht_test.listwise.cofea_reader",
        item_attrs=set_default_value_int_attrs_for_item + set_default_value_double_attrs_for_item 
                        + ["ltr", "rtr", "dtr", "sqrt_like_cnt", "sqrt_reply_cnt", 'sqrt_dislike_cnt', 'sqrt_hour_diff', 
                            "like_cnt_dis", "reply_cnt_dis", "dislike_cnt_dis", "minute_diff_dis", "ltr_dis", "rtr_dis", "dtr_dis"]
                        +['like_label', 'reply_label', 'expand_label', 'continuous_expand_label', 'copy_label', 'share_label', 'audience_label'],
        aggregator="max",
    ) \
    .perflog_attr_value(
        check_point="wht_test.listwise.cofea_reader",
        common_attrs=["gender", "age_segment"],
        aggregator="count",
    ) \

    

send_mio = DataReaderFlow(name="send_mio") \
    .count_reco_result(save_count_to="send_sample_cnt") \
    .perflog_attr_value(
        check_point="wht_test.listwise.cofea_reader",
        common_attrs=["send_sample_cnt", "user_hash", "photo_hash"]
    ) \
    .send_to_mio_learner(
        # 在这里就会丢失photo hash的int64精度，mio不支持int64
        attrs=['like_label_list', 'reply_label_list', 'expand_label_list', 'continuous_expand_label_list', 'copy_label_list', 
               'share_label_list', 'audience_label_list', 'mask_pack', 'comment_id_list', 'photo_id'],
        slots_attrs=["comment_common_slots"],
        signs_attrs=["comment_common_signs"],
        lineid_attr="user_id",
        time_ms_attr="time_ms",
        label_attr="like_label",
        user_hash_attr="user_hash"
    )

flows = [read_data, gen_feature, send_mio]
runner = OfflineRunner("comment_profile")
runner.IGNORE_UNUSED_ATTR=['llsid']
runner.add_leaf_flows(leaf_flows=flows)
runner.build(output_file=os.path.join(current_dir, "cofea_reader.json"))
