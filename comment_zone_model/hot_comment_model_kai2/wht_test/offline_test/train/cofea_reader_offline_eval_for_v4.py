"""
注意：eval时showAction置为1
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

SEQ_LEN=20

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
              # "llsid", "city_name",  "photo_upload_time", 
            #   "mod", "page_type_str", "is_political", "product_name",
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
            # 'comment_content_segs', 'comment_genre', 'mmu_category_tag', 'mmu_emotion_tag',  'related_score',
             # 'risk_insult_tag', 'risk_negative_tag', 'risk_inactive_tag', 
            # 'quality_v2_score', 'quality_score', 
            # 'show_cnt_weekly', 'like_cnt_weekly', 'reply_cnt_weekly', 'auto_expand', 'first_like_cnt',
            # 'sub_like_cnt', 'first_level_like_cnt', 
            # 'content_length', 'content_segment_num', 'inform_cnt', 'copy_cnt',
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



# gen_kgnn_fea = DataReaderFlow(name="gen_kgnn_fea") \
#     .fetch_kgnn_neighbors(
#         id_from_common_attr="user_id",
#         save_weight_to="comment_weights",  # like + reply
#         save_neighbors_to="comment_ids",
#         edge_attr_schema=NodeAttrSchema(1, 0).add_int64_list_attr("comment_mmu_categories", 1),
#         kess_service="grpc_kgnn_user_interact_comment_info-U2I",
#         relation_name='U2I',
#         shard_num=4,
#         sample_num=SEQ_LEN,
#         timeout_ms=10,
#         sample_type="topn",
#         padding_type="zero",
#     ) \
#     .cast_attr_type(
#         attr_type_cast_configs=[
#             {
#                 "to_type": "int",
#                 "from_common_attr": "comment_weights",
#                 "to_common_attr": "comment_weights_int"
#             }
#         ]
#     ) \
#     .enrich_attr_by_lua(
#         import_common_attr=["comment_ids"],
#         export_common_attr=["mask_pack", "user_seq_len", "user_seq_coverage"],
#         function_for_common="gen_user_seq_mask",
#         lua_script=f"""
#             function gen_user_seq_mask()
#                 local final_seg_num = math.min({SEQ_LEN}, #(comment_ids or {{}}))

#                 local mask_pack = {{}}
#                 for i=1, final_seg_num do
#                     mask_pack[i] = 1
#                 end
#                 while #mask_pack < {SEQ_LEN} do
#                     table.insert(mask_pack, 0)
#                 end

#                 local user_seq_coverage=0
#                 if #(comment_ids or {{}}) > 0 then
#                     user_seq_coverage=1
#                 end
#                 return mask_pack, #(comment_ids or {{}}), user_seq_coverage
#             end
#         """
#     ) \


# gen_new_feature = DataReaderFlow(name="gen_new_feature") \
#     .set_attr_value(
#         common_attrs=default_value_map_list_for_common,
#         item_attrs=default_value_map_list_for_item,
#         no_overwrite=True
#     ) \
#     .enrich_attr_by_lua(
#         import_item_attr=["like_cnt", "reply_cnt", "dislike_cnt", "realshow_cnt", 'first_like_cnt', 'sub_like_cnt', 
#                           'first_level_like_cnt', "show_cnt_weekly", "like_cnt_weekly", "reply_cnt_weekly",
#                           "copy_cnt", "minute_diff"],
#         export_item_attr=["ltr", "rtr", "dtr", "sqrt_like_cnt", "sqrt_reply_cnt", 'sqrt_dislike_cnt', 
#                           'sqrt_hour_diff', 'sqrt_sub_like_cnt', 'sqrt_first_level_like_cnt', 'sqrt_first_like_cnt',
#                           'dislike_like_ratio', 'sub_root_like_ratio', 'ltr_weekly', 'rtr_weekly', 'sqrt_copy_cnt',
#                           'ltr_copy', 'rtr_copy'],
#         function_for_item="cal_xtr",
#         lua_script="""
#             function cal_xtr()
#                 local vv = realshow_cnt or 0.0
#                 local ltr = like_cnt / (vv + 1.0)
#                 local rtr = reply_cnt / (vv + 1.0)
#                 local dtr = dislike_cnt / (vv + 1.0)
#                 local sqrt_like_cnt = math.sqrt(like_cnt)
#                 local sqrt_reply_cnt = math.sqrt(reply_cnt)
#                 local sqrt_dislike_cnt = math.sqrt(dislike_cnt)
#                 local sqrt_hour_diff = math.sqrt(minute_diff / 60)
#                 local sqrt_sub_like_cnt = math.sqrt(sub_like_cnt)
#                 local sqrt_first_level_like_cnt = math.sqrt(first_level_like_cnt)
#                 local sqrt_first_like_cnt = math.sqrt(first_like_cnt)
#                 local dislike_like_ratio = dislike_cnt / (like_cnt + 1.0)
#                 local sub_root_like_ratio = first_like_cnt / (first_level_like_cnt + 1.0)
#                 local ltr_weekly = like_cnt_weekly / (show_cnt_weekly + 1.0)
#                 local rtr_weekly = reply_cnt_weekly / (show_cnt_weekly + 1.0)
#                 local sqrt_copy_cnt = math.sqrt(copy_cnt)
#                 return ltr, rtr, dtr, sqrt_like_cnt, sqrt_reply_cnt, sqrt_dislike_cnt, sqrt_hour_diff, sqrt_sub_like_cnt, sqrt_first_level_like_cnt, sqrt_first_like_cnt, dislike_like_ratio, sub_root_like_ratio, ltr_weekly, rtr_weekly, sqrt_copy_cnt, ltr, rtr
#             end
#         """
#     ) \
    # .enrich_attr_by_lua(
    #     import_item_attr=["comment_content_segs"],
    #     export_item_attr=["comment_content_segs"],
    #     function_for_item="add_cls",
    #     lua_script="""
    #         function add_cls()
    #             local added = {}
    #             added[1] = "[cls]"
    #             for i = 1, #(comment_content_segs or {}) do  
    #                 added[i+1] = comment_content_segs[i]
    #             end
    #             return added
    #         end
    #     """
    # ) \
    # .enrich_attr_by_lua(
    #     import_item_attr=["comment_content_segs"],
    #     export_item_attr=["mask_pack"],
    #     function_for_item="seg_num",
    #     lua_script=f"""
    #         function seg_num()
    #             local final_seg_num = math.min({SEQ_LEN}, #(comment_content_segs or {{}}))
                
    #             local mask_pack = {{}}
    #             for i=1, final_seg_num do
    #                 mask_pack[i] = 1
    #             end
    #             while #mask_pack < {SEQ_LEN} do
    #                 table.insert(mask_pack, 0)
    #             end
    #             return mask_pack
    #         end
    #     """
    # ) \
    # .set_attr_value(
    #     item_attrs=[
    #         {
    #         "name": "pos_ids",
    #         "type": "int_list",
    #         "value": list(range(SEQ_LEN))
    #         },
    #     ]
    # ) \
    

gen_new_feature = DataReaderFlow(name="gen_new_feature") \
    .get_remote_embedding_lite(
        kess_service="grpc_mmuCommentContentEmb",
        id_converter={"type_name": "mioEmbeddingIdConverter"},
        query_source_type="item_attr",
        input_attr_name="comment_id",
        output_attr_name="comment_content_emb_v2",
        timeout_ms=50,
        slot=107,
        size=256,
        shard_num=4,
        client_side_shard=True,
    ) \
    .get_remote_embedding_lite(
        kess_service="grpc_mmuCommentContentEmb",
        id_converter={"type_name": "mioEmbeddingIdConverter"},
        query_source_type="item_attr",
        input_attr_name="comment_id",
        output_attr_name="visual_comment_content_emb_vista",
        timeout_ms=50,
        slot=108,
        size=256,
        shard_num=4,
        client_side_shard=True,
    ) \
    .count_reco_result(
        save_count_to="none_content_emb_count",
        select_item = {
          "attr_name": "comment_content_emb_v2",
          "compare_to": [],
          "select_if": "is null",
          "select_if_attr_missing": True,
        },
    ) \
    .count_reco_result(
        save_count_to="none_visual_emb_count",
        select_item = {
          "attr_name": "visual_comment_content_emb_vista",
          "compare_to": [],
          "select_if": "is null",
          "select_if_attr_missing": True,
        },
    ) \
    .count_reco_result(
        save_count_to="item_count",
    ) \
    .enrich_attr_by_lua(
        import_common_attr = ["none_content_emb_count", "none_visual_emb_count", "item_count" ],
        function_for_common = "calculate",
        export_common_attr = ["none_content_emb_prop", "none_visual_emb_prop"],
        lua_script = """
            function calculate()
                return none_content_emb_count * 1.0 / item_count, none_visual_emb_count * 1.0 / item_count
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

            ## new feature
            # **id_config("mod", 110),        # dim=32
            # **id_config("page_type_str", 111),  
            # **id_config("is_political", 112),       # dim=4
            # **id_config("product_name", 113),  

            # **id_config("city_name", 114),          # dim=32
            # **id_config("request_hour", 115),       # dim=8
            # **id_config("request_day", 116),    

            # **list_config("comment_ids", 300, 201, SEQ_LEN),

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
            # **id_config("comment_genre", 250),      # dim=8
            # # **id_config("risk_insult_tag", 251),    
            # # **id_config("risk_inactive_tag", 252),
            # # **id_config("risk_negative_tag", 253),
            # **id_config("mmu_emotion_tag", 254),
            # **id_config("mmu_category_tag", 255),

            # # denominator, smooth, max_val, buckets, min_val
            # **discreate_config("sqrt_like_cnt", 271, [1, 0, 1000, 1, 0]),   # dim=12
            # **discreate_config("sqrt_reply_cnt", 272, [1, 0, 1000, 1, 0]),
            # **discreate_config("sqrt_dislike_cnt", 273, [1, 0, 1000, 1, 0]),
            # **discreate_config("sqrt_hour_diff", 274, [1, 0, 1000, 1, 0]),
            # **discreate_config("sqrt_sub_like_cnt", 275, [1, 0, 1000, 1, 0]),
            # **discreate_config("sqrt_first_level_like_cnt", 276, [1, 0, 1000, 1, 0]),
            # **discreate_config("sqrt_first_like_cnt", 277, [1, 0, 1000, 1, 0]),

            # # **discreate_config("quality_score", 278, [0.01, 0, 100, 1, 0]), # dim=8
            # **discreate_config("related_score", 279, [0.01, 0, 100, 1, 0]),
            # # **discreate_config("quality_v2_score", 280, [0.01, 0, 100, 1, 0]),
            # **discreate_config("ltr_copy", 281, [0.01, 0, 100, 1, 0]),
            # **discreate_config("rtr_copy", 282, [0.01, 0, 100, 1, 0]),
            # **discreate_config("dtr", 283, [0.01, 0, 100, 1, 0]),
            # **discreate_config("ltr_weekly", 284, [0.01, 0, 100, 1, 0]),
            # **discreate_config("rtr_weekly", 285, [0.01, 0, 100, 1, 0]),

            # **discreate_config("dislike_like_ratio", 286, [0.01, 0, 1000, 1, 0]),   # dim=12
            # **discreate_config("sub_root_like_ratio", 287, [0.01, 0, 1000, 1, 0]),
            # **discreate_config("content_length", 288, [5, 0, 1000, 1, 0]),      
            # **discreate_config("content_segment_num", 289, [1, 0, 1000, 1, 0]), 
            # **discreate_config("inform_cnt", 290, [1, 0, 1000, 1, 0]),         
            # **discreate_config("sqrt_copy_cnt", 291, [1, 0, 1000, 1, 0]),       

            # **id_config("auto_expand", 270),   # dim=4
            # **id_config("has_pic", 292),       
            # **id_config("has_emoji", 293),   
            # **id_config("is_text_pic", 294),   
            # **id_config("is_text_emoji", 295),   
            # **id_config("is_ai_play", 296),   
            # **id_config("is_ai_kwai_wonderful_rely", 297),   
            # **id_config("is_comment_contain_at", 298),   

            # **list_config("comment_content_segs", 300, 300, SEQ_LEN),    
            # **list_config("pos_ids", 301, 301, SEQ_LEN),

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
    # .perflog_attr_value(
    #     check_point="wht_test.offline_test.cofea_reader_online_for_v4",
    #     common_attrs=['request_hour', 'request_day'],
    #     aggregator="avg",
    # ) \
    # .perflog_attr_value(
    #     check_point="wht_test.offline_test.cofea_reader_online_for_v4",
    #     common_attrs=['city_name'],
    #     aggregator="count",
    # ) \

    


send_mio = DataReaderFlow(name="send_mio") \
    .count_reco_result(save_count_to="send_sample_cnt") \
    .perflog_attr_value(
        check_point="wht_test.offline_test.cofea_reader_online_for_v4",
        common_attrs=['send_sample_cnt', ] + labels,
    ) \
    .log_debug_info(
        for_debug_request_only=False,
        respect_sample_logging=True,
        print_all_common_attrs=True,
        print_all_item_attrs=True
    ) \
    .send_to_mio_learner(
        attrs=labels + ['recall_type', 'sample_weight'],
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
