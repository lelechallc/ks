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
              "mod", "page_type_str", 
            #   "llsid",
            #   "client_request_info", , "is_political", "product_name", 
            #   "city_name", "photo_upload_time",
            #   "photo_real_show", "photo_click_count", "photo_like_count", "photo_follow_count", "photo_forward_count",
            #   "photo_long_play_count", "photo_short_play_count", "photo_comment_count", "photo_view_length_sum",
            #   "photo_effective_play_count", "photo_comment_stay_time_sum_ms", "photo_recommend_count",
            #   "photo_upload_type", "photo_duration_ms", "photo_author_low_score", "photo_author_high_score",
            #   "photo_hetuV1_level1_tag_id"
]

labels=[
    "expandAction_first", "replyAction_first", "likeAction_first", "audienceAction_first", 
        "expandAction_second", 
        "copyAction_first", "shareAction_first", 
        # "cancelHateAction_first", "cancelHateAction_second", 'cancelLikeAction_first', 'cancelLikeAction_second',
        # 'hateAction_first', 'hateAction_second', 'replyTaskAction_first', 'replyTaskAction_second', 
        # 'subAtAction_first', 'subAtAction_second', "stayDurationMs", "subShowCntAction",
        # "audienceAction_second", "copyAction",  "copyAction_second", "shareAction_second",
        # "likeAction_second", 'replyAction_second', 'reportAction_first', 'shareAction', 'showAction'
]

item_attrs=["comment_id", "author_id", "like_cnt", "reply_cnt", "minute_diff", "dislike_cnt",
            "comment_genre", 'content_length',
            "risk_insult_tag", "risk_negative_tag", "risk_inactive_tag",
            "mmu_category_tag", "mmu_emotion_tag", 
            # "realshow_cnt", "recall_type", "showAction", 
            # "comment_content_segs_v", "mmu_entity_list_v",  "sample_weight",
            # "related_score", "quality_score", "predict_like_score", "predict_reply_score", "quality_v2_score", 
            # 'expand_cnt', 'inform_cnt', 'copy_cnt', 'sub_like_cnt', 'first_level_like_cnt',
            # "show_cnt_weekly", "like_cnt_weekly", "reply_cnt_weekly", "auto_expand", "first_like_cnt",
            # 'has_pic', 'has_emoji', 'is_text_pic', 'is_text_emoji', 'is_ai_play', 'is_ai_kwai_wonderful_rely',
            #  'content_segment_num', 'is_comment_contain_at', 'before_rerank_seq'
]



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
    
    

gen_feature = DataReaderFlow(name="gen_feature") \
    .extract_kuiba_parameter(
      config={
          # **id_config("gender", 101),
          # **id_config("age_segment", 102),
          **id_config("photo_id", 103),
          **id_config("photo_author_id", 104),
          **id_config("user_id", 105),
          **id_config("device_id", 106),
          **id_config("product", 107),
          **id_config("page_type_str", 108),
          **id_config("age_segment", 109),
          **id_config("gender", 110),
          **id_config("city_name", 111),
          **id_config("mod", 112),
          **id_config("is_political", 113),
      },
      is_common_attr=True,
      slots_output="comment_common_slots",
      parameters_output="comment_common_signs",
    ) \
    .extract_kuiba_parameter(
      config={
        **id_config("comment_id", 201),
        **id_config("author_id", 202),
        **discreate_config("like_cnt", 203, [5, 0, 100000, 1, 0]),
        **discreate_config("reply_cnt", 204, [5, 0, 100000, 1, 0]),
        **discreate_config("minute_diff", 205, [36, 0, 336, 1, 0]),
        # **discreate_config("ltr", 206, [0.001, 0, 1000, 1, 0]),
        # **discreate_config("rtr", 207, [0.001, 0, 1000, 1, 0]),
        # **id_config("showAction", 208),
        **discreate_config("dislike_cnt", 209, [3, 0, 100000, 1, 0]),

        # new_feature
        # **discreate_config("ltr_copy", 210, [0.001, 0, 1000, 137, 0]),
        # **discreate_config("rtr_copy", 211, [0.001, 0, 1000, 137, 0]),
        # **list_config("comment_content_segs", 212, 212, 16),

        # risk and mmu comment content label
        **id_config("comment_genre", 213),
        **id_config("risk_insult_tag", 214),
        **id_config("risk_negative_tag", 215),
        **id_config("risk_inactive_tag", 216),
        **id_config("mmu_category_tag", 217),
        **id_config("mmu_emotion_tag", 218),
        # **list_config("mmu_entity_list", 219, 219, 10),
        **id_config("content_length", 220),

        # 分
        **discreate_config("quality_score", 230, [0.001, 0, 1000, 1, 0]),
        **discreate_config("quality_v2_score", 231, [0.001, 0, 1000, 1, 0]),
        **discreate_config("related_score", 232, [0.001, 0, 1000, 1, 0]),
        **id_config("is_ai_play", 233),
        **id_config("is_comment_contain_at", 234),
        **id_config("is_text_emoji", 235),
        **id_config("is_ai_kwai_wonderful_rely", 236),
        **id_config("has_pic", 237),
        **id_config("is_text_pic", 238),
      },
      is_common_attr=False,
      slots_output="comment_item_slots",
      parameters_output="comment_item_signs",
    ) \
    .enrich_attr_by_lua(
      import_item_attr=["expandAction_first", "likeAction_first", "replyAction_first", "audienceAction_first", "copyAction_first", "shareAction_first", "expandAction_second"],
      export_item_attr=["expandAction_v", "likeAction_v", "replyAction_v", "audienceAction_v", "copyAction_v", "shareAction_v", "expandActionSecond_v"],
      function_for_item="trans",
      lua_script="""
        function trans()
          local expandAction_v = 0
          local likeAction_v = 0
          local replyAction_v = 0
          local audienceAction_v = 0
          local copyAction_v = 0
          local shareAction_v = 0
          local expandActionSecond_v = 0

          if expandAction_first ~= nil and expandAction_first > 0 then
            expandAction_v = expandAction_first
          end
          if likeAction_first ~= nil and likeAction_first > 0 then
            likeAction_v = 1
          end
          if replyAction_first ~= nil and replyAction_first > 0 then
            replyAction_v = 1
          end
          if audienceAction_first ~= nil and audienceAction_first > 0 then
            audienceAction_v = 1
          end
          if copyAction_first ~= nil and copyAction_first > 0 then
            copyAction_v = 1
          end
          if shareAction_first ~= nil and shareAction_first > 0 then
            shareAction_v = 1
          end
          if expandAction_second ~= nil and expandAction_second > 0 then
            expandActionSecond_v = 1
          end
          return expandAction_v, likeAction_v, replyAction_v, audienceAction_v, copyAction_v, shareAction_v, expandActionSecond_v
        end
      """
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
        check_point="send.mio",
        item_attrs=["expandAction_v", "likeAction_v", "replyAction_v"],
        common_attrs=["send_sample_cnt"]
    ) \
    .send_to_mio_learner(
        attrs=["expandAction_v", "likeAction_v", "replyAction_v", "audienceAction_v", "copyAction_v", "shareAction_v", "expandActionSecond_v"],
        slots_attrs=["comment_common_slots", "comment_item_slots"],
        signs_attrs=["comment_common_signs", "comment_item_signs"],
        lineid_attr="user_id",
        time_ms_attr="time_ms",
        label_attr="likeAction_v",
        user_hash_attr="user_photo_hash"
    )

flows = [read_data, gen_feature, send_mio]
runner = OfflineRunner("comment_profile")
# runner.CHECK_UNUSED_ATTR=False        # 这个不要设置为False，否则不便于发现特征缺失
runner.add_leaf_flows(leaf_flows=flows)
runner.build(output_file=os.path.join(current_dir, "cofea_reader.json"))
