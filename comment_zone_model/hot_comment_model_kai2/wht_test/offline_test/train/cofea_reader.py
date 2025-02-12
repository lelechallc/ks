""" 
和 cofea_reader.py 的区别在于，是从 kafka 中读取数据，而不是从离线文件中读取
kafka数据源：universe_feature/.../comment/xtr/pipeline.py (send_topic)
如果要增加特征，需要修改universe_feature
特征改动：
- 增加 sqrt_like_cnt, sqrt_reply_cnt, sqrt_dislike_cnt, sqrt_hour_diff, sqrt_sub_like_cnt, sqrt_first_level_like_cnt, sqrt_first_like_cnt 取整分桶的离散化embedding. 最大值均限制为1000.
- 增加 对 dislike_like_ratio, sub_root_like_ratio(first_like_cnt/first_level_like_cnt) 进行等距分桶的离散化embedding
- 增加 对 quality_score, quality_v2_score, related_score 进行等距分桶的离散化embedding
- 增加 对 ltr, rtr, dtr 进行等距分桶的离散化embedding
- 增加 comment_genre (类别特征) —— 0-纯文本评论；1-包含图的评论
- 增加 risk_insult_tag, risk_inactive_tag, risk_negative_tag, mmu_emotion_tag, mmu_category_tag (类别特征)
- 增加 auto_expand (类别特征，0/1)
- 增加 ltr_weekly, rtr_weekly 等间距分桶的离散化embedding (使用"show_cnt_weekly", "like_cnt_weekly", "reply_cnt_weekly"计算)
- 增加 机型特征 mod (str)
- 增加 评论长度 content_length(avg=19)；评论词数 content_segment_num(avg=6) (TODO check 数据是否正常)
- 增加 inform_cnt (avg=1), sqrt_copy_cnt(avg=27).      expand_cnt avg=51k, 故舍弃。
- 增加 'has_pic', 'has_emoji', 'is_text_pic', 'is_text_emoji', 'is_ai_play'(AI玩评), 'is_ai_kwai_wonderful_rely'（被AI小快正常回复，非兜底）, 
        'is_comment_contain_at' 等离散特征   TODO check 数据是否正常


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
args = parser.parse_args()


common_attrs=["user_id", "device_id", "photo_id", "photo_author_id", "gender", "age_segment", "llsid",
              "mod", "page_type_str", "is_political"]
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
    'audienceaction_first',
    'audienceaction_second',
    'reportaction_first',
    'reportaction_second',
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
    'showaction': (9, 'int'),
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
    'audienceaction_first': (38, 'int'),
    'audienceaction_second': (39, 'int'),
    'reportaction_first': (40, 'int'),
    'reportaction_second': (41, 'int'),
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
}
read_data = DataReaderFlow(name="read_data") \
    .fetch_message(
        group_id="reco_forward_open_log",
        hdfs_path=args.data_path,
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
    .end_()


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

default_value_map_list = []
for name in set_default_value_int_attrs:
    default_value_map_list.append({"name": name, "type": "int", "value": 0})
for name in set_default_value_double_attrs:
    default_value_map_list.append({"name": name, "type": "double", "value": 0.0})
for name in set_default_value_string_attrs:
    default_value_map_list.append({"name": name, "type": "string", "value": ''})

gen_feature = DataReaderFlow(name="gen_feature") \
    .set_attr_default_value(
        item_attrs=default_value_map_list
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
            
            **id_config("is_political", 112),     # dim=4
            
            # "comment_ids": {"attrs": [{"key_type": 300, "attr": ["comment_ids"], **kuiba_list_converter_config_list_limit(20)}]},
            # "comment_weights": {"attrs": [{"key_type": 301, "attr": ["comment_weights"], **kuiba_list_converter_config_list_limit(20)}]},
            # "comment_mmu_categories": {"attrs": [{"key_type": 302, "attr": ["comment_mmu_categories"], **kuiba_list_converter_config_list_limit(20)}]},

            # 序列特征
            # **list_config("comment_ids", 300, 201, 20),
            # **list_config("comment_weights", 301, 301, 20),
            # **list_config("comment_mmu_categories", 302, 255, 20)
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
            
            **id_config("showaction", 208),         # dim=8

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
    # .get_remote_embedding_lite(
    #     kess_service="grpc_mmuCommentContentEmb",
    #     id_converter={"type_name": "mioEmbeddingIdConverter"},
    #     query_source_type="item_attr",
    #     input_attr_name="comment_id",
    #     output_attr_name="mmu_hetu_content_emb",
    #     timeout_ms=50,
    #     slot=101,
    #     size=128,
    #     shard_num=4,
    #     client_side_shard=True,
    # ) \
    # .get_remote_embedding_lite(
    #     kess_service="grpc_mmuCommentContentEmb",
    #     id_converter={"type_name": "mioEmbeddingIdConverter"},
    #     query_source_type="item_attr",
    #     input_attr_name="comment_id",
    #     output_attr_name="mmu_clip_content_emb",
    #     timeout_ms=50,
    #     slot=104,
    #     size=1024,
    #     shard_num=4,
    #     client_side_shard=True,
    # ) \
    # .get_remote_embedding_lite(
    #     kess_service="grpc_mmuCommentContentEmb",
    #     id_converter={"type_name": "mioEmbeddingIdConverter"},
    #     query_source_type="item_attr",
    #     input_attr_name="comment_id",
    #     output_attr_name="mmu_bert_content_emb",
    #     timeout_ms=50,
    #     slot=103,
    #     size=256,
    #     shard_num=4,
    #     client_side_shard=True,
    # ) \
    # .set_attr_default_value(
    #     item_attrs=[
    #         {
    #             "name": "mmu_hetu_content_emb",
    #             "type": "double_list",
    #             "value": [0.0] * 128,
    #         },
    #         {
    #             "name": "mmu_clip_content_emb",
    #             "type": "double_list",
    #             "value": [0.0] * 1024,
    #         },
    #         {
    #             "name": "mmu_bert_content_emb",
    #             "type": "double_list",
    #             "value": [0.0] * 256,
    #         }
    #     ]
    # )
# .fetch_kgnn_neighbors(
    #     id_from_common_attr="user_id",
    #     save_weight_to="comment_weights",  # like + reply
    #     save_neighbors_to="comment_ids",
    #     edge_attr_schema=NodeAttrSchema(1, 0).add_int64_list_attr("comment_mmu_categories", 1),
    #     kess_service="grpc_kgnn_user_interact_comment_info-U2I",
    #     relation_name='U2I',
    #     shard_num=4,
    #     sample_num=20,
    #     timeout_ms=100,
    #     sample_type="topn",
    #     padding_type="zero",
    # ) \
    # .log_debug_info(
    #     common_attrs=["user_id", "comment_ids", "comment_mmu_categories", "comment_weights"],
    #     for_debug_request_only=False,
    #     respect_sample_logging=False,
    # ) \


send_mio = DataReaderFlow(name="send_mio") \
    .count_reco_result(save_count_to="send_sample_cnt") \
    .perflog_attr_value(
        check_point="send_mio.before",
        item_attrs=labels + ['sample_weight', 'recall_type', 'comment_genre'],
        common_attrs=["send_sample_cnt", "user_hash", "photo_hash"]
    ) \
    .log_debug_info(
        for_debug_request_only=False,
        respect_sample_logging=True,
        print_all_common_attrs=True,
        print_all_item_attrs=True
    ) \
    .send_to_mio_learner(
        # 在这里就会丢失photo hash的int64精度，mio不支持int64
        attrs=labels + ["comment_genre", "recall_type", "sample_weight"
                        # "mmu_hetu_content_emb", "mmu_clip_content_emb", "mmu_bert_content_emb"
                        ],
        slots_attrs=["comment_common_slots", "comment_item_slots"],
        signs_attrs=["comment_common_signs", "comment_item_signs"],
        lineid_attr="user_id",
        # time_ms_attr="time_ms",
        label_attr="like_action",
        user_hash_attr="user_hash"
    )

flows = [read_data, gen_feature, send_mio]
runner = OfflineRunner("comment_model")
runner.IGNORE_UNUSED_ATTR=['llsid'] + ['age_segment', 'audienceaction', 'device_id', 'expandaction', 'gender', 'is_political', 'likeaction', 'mod', 'page_type_str', 'photo_author_id', 'photo_id', 'predict_like_score', 'predict_reply_score', 'replyaction', 'reportaction', 'user_id']
runner.add_leaf_flows(leaf_flows=flows)
runner.build(output_file=os.path.join(current_dir, "cofea_reader.json"))
