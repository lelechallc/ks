from dragonfly.ext.cofea.cofea_api_mixin import CofeaApiMixin
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin
from dragonfly.common_leaf_dsl import LeafFlow, OfflineRunner
import os
import sys

current_dir = os.path.dirname(__file__)


class DataReaderFlow(LeafFlow, MioApiMixin, OfflineApiMixin, CofeaApiMixin, KuibaApiMixin):
    def clean_all(self, reason, **kwargs):
        return self.limit(0, name="clean_all_for_" + reason, **kwargs)


common_attrs=["user_id", "device_id", "photo_id", "photo_author_id", "gender", "age_segment", "llsid"]
item_attrs=["comment_id", "author_id", "like_cnt", "reply_cnt", "minute_diff", "realshow_cnt", "dislike_cnt",
            "showAction", "expandAction", "replyAction", "likeAction", "audienceAction", "reportAction",
            "comment_genre", "risk_insult_tag", "risk_negative_tag", "risk_inactive_tag", "mmu_category_tag", "mmu_emotion_tag", "mmu_entity_list",
            "sample_weight"]

SEQ_SIZE = 16

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
              + [dict(name=f"{item_attr}_list_common_retrieve", path="sample.attr", sample_attr_name=item_attr) for item_attr in item_attrs]
              + [dict(name="request_time", path="timestamp")]

    ) \
    .if_("#(comment_id_list_common_retrieve or {}) == 0") \
        .return_(0) \
    .end_() \
    .enrich_attr_by_lua(
        import_common_attr=["request_time"],
        export_common_attr=["time_ms", "sample_minute_diff"],
        function_for_common="cal",
        lua_script="""
            function cal()
                local time_ms = request_time // 1000
                local sample_minute_diff = (util.GetTimestamp() - request_time) / 1000 / 1000 / 60.0
                return time_ms, sample_minute_diff
            end
        """
    ) \
    .retrieve_by_common_attr(
        attr="comment_id_list_common_retrieve",
        reason=999
    ) \
    .dispatch_common_attr(
        dispatch_config=[
            dict(from_common_attr=f"{attr}_list_common_retrieve", to_item_attr=attr) for attr in item_attrs
        ]
    ) \
    .count_reco_result(save_count_to="retrieve_item_cnt") \
    .filter_by_attr(
        attr_name="showAction",
        remove_if="<=",
        compare_to=0,
        remove_if_attr_missing=True,
    ) \
    .count_reco_result(save_count_to="show_item_cnt") \
    .perflog_attr_value(
        check_point="comment.listwise",
        common_attrs=["retrieve_item_cnt", "show_item_cnt"],
    ) \
    .if_("show_item_cnt <= 0") \
        .return_(0) \
    .end_() \
    .enrich_attr_by_lua(
        import_item_attr=["expandAction", "likeAction", "replyAction"],
        export_item_attr=["expand", "like", "reply", "expand_v", "like_v", "reply_v"],
        function_for_item="transform",
        lua_script='''
            function transform()
                local expand = math.min(math.max((expandAction or 0), 0), 1)
                local like = math.min(math.max((likeAction or 0), 0), 1)
                local reply = math.min(math.max((replyAction or 0), 0), 1)
                return expand, like, reply,
                    expand * 1.0, like * 1.0, reply * 1.0
            end
        '''
    ) \
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
      export_item_attr=["calc_weight"],
      function_for_item="calc",
      lua_script="""
        function calc()
          local weight = 1.0
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
    .limit(SEQ_SIZE)

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


extract_fea = DataReaderFlow(name="extract_fea") \
    .enrich_attr_by_lua(
        import_item_attr=["like_cnt", "reply_cnt", "minute_diff", "ltr", "rtr", "dislike_cnt"],
        export_item_attr=["like_cnt_dis", "reply_cnt_dis", "minute_diff_dis", "ltr_dis", "rtr_dis", "dislike_cnt_dis"],
        function_for_item="discreate",
        lua_script='''
            function discreate()
                local like_cnt_dis = math.max(math.min(like_cnt // 5, 100000), 0) * 1
                local reply_cnt_dis = math.max(math.min(reply_cnt // 5, 100000), 0) * 1
                local minute_diff_dis = math.max(math.min(minute_diff // 3, 14400), 0) * 1
                local ltr_dis = math.max(math.min(ltr * 1000, 100000), 0) * 137
                local rtr_dis = math.max(math.min(rtr * 1000, 100000), 0) * 137
                local dislike_cnt_dis = math.max(math.min(dislike_cnt // 3, 100000), 0) * 1
                return math.ceil(like_cnt_dis), math.ceil(reply_cnt_dis), math.ceil(minute_diff_dis),
                       math.ceil(ltr_dis), math.ceil(rtr_dis), math.ceil(dislike_cnt_dis)
            end
        '''
    ) \
    .count_reco_result(save_count_to="final_seq_cnt") \
    .enrich_attr_by_lua(
        import_common_attr=["final_seq_cnt"],
        export_common_attr=["mask_pack"],
        function_for_common="gen_mask_pack",
        lua_script=f'''
            function gen_mask_pack()
                local mask_pack = {{}}
                for i=1, final_seq_cnt do
                    table.insert(mask_pack, 1)
                end
                while #mask_pack < {SEQ_SIZE} do
                    table.insert(mask_pack, 0)
                end
                return mask_pack
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
                "from_item_attr": "dislike_cnt_dis",
                "to_common_attr": "dislike_cnt_dis_list",
            },
            {
                "from_item_attr": "sample_weight",
                "to_common_attr": "sample_weight_pack",
            },
            {
                "from_item_attr": "expand_v",
                "to_common_attr": "expand_v_pack",
            },
            {
                "from_item_attr": "like_v",
                "to_common_attr": "like_v_pack",
            },
            {
                "from_item_attr": "reply_v",
                "to_common_attr": "reply_v_pack",
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
        ]
    ) \
    .extract_kuiba_parameter(
      config={
          **id_config("gender", 101),
          **id_config("age_segment", 102),
          **id_config("photo_id", 103),
          **id_config_slot("photo_author_id", 104, 202),
          **list_config("comment_id_list", 201, 201, SEQ_SIZE),
          **list_config("author_id_list", 202, 202, SEQ_SIZE),
          **list_config("like_cnt_dis_list", 203, 203, SEQ_SIZE),
          **list_config("reply_cnt_dis_list", 204, 204, SEQ_SIZE),
          **list_config("minute_diff_dis_list", 205, 205, SEQ_SIZE),
          **list_config("ltr_dis_list", 206, 206, SEQ_SIZE),
          **list_config("rtr_dis_list", 207, 207, SEQ_SIZE),
          **list_config("dislike_cnt_dis_list", 208, 208, SEQ_SIZE),

          **list_config("comment_genre_list", 209, 209, SEQ_SIZE),
          **list_config("risk_insult_tag_list", 210, 210, SEQ_SIZE),
          **list_config("risk_negative_tag_list", 211, 211, SEQ_SIZE),
          **list_config("risk_inactive_tag_list", 212, 212, SEQ_SIZE),
          **list_config("mmu_category_tag_list", 213, 213, SEQ_SIZE),
          **list_config("mmu_emotion_tag_list", 214, 214, SEQ_SIZE),
      },
      is_common_attr=True,
      slots_output="comment_common_slots",
      parameters_output="comment_common_signs",
    )

pack_list = DataReaderFlow(name="pack_list") \
    .log_debug_info(
        for_debug_request_only=False,
        respect_sample_logging=True,
        common_attrs=["llsid", "user_id", "device_id", "time_ms", "gender", "age_segment", "photo_id", "photo_author_id", "context_info", "comment_common_slots", "comment_common_signs", "mask_pack"],
        item_attrs=["sample_weight", "expandAction", "likeAction", "replyAction", "expandAction_v", "likeAction_v", "replyAction_v"]
    ) \
    .perflog_attr_value(
      check_point="comment.listwise",
      common_attrs=["sample_minute_diff", "retrieve_item", "show_cnt", "final_seq_cnt",
                      "sample_weight_pack", "expand_v_pack", "like_v_pack", "reply_v_pack", "mask_pack"],
      item_attrs=["showAction", "expandAction", "replyAction", "likeAction", "audienceAction", "reportAction",
                  "expand", "like", "reply", "expand_v", "like_v", "reply_v"
                  "sample_weight", "calc_weight"],
    ) \
    .limit(0) \
    .fake_retrieve(item_keys=[111], reason=888) \
    

send_mio = DataReaderFlow(name="send_mio") \
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
        end"""
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
    .send_to_mio_learner(
        attrs=["sample_weight_pack", "expand_v_pack", "like_v_pack", "reply_v_pack", "mask_pack"],
        slots_attrs=["comment_common_slots"],
        signs_attrs=["comment_common_signs"],
        lineid_attr="user_id",
        time_ms_attr="time_ms",
        label_attr="like_v",
        user_hash_attr="user_hash"
    )

pipelines = [read_data, extract_fea, pack_list, send_mio]
runner = OfflineRunner("comment_listwise")
runner.IGNORE_UNUSED_ATTR=['mmu_entity_list', "photo_hash"]
runner.add_leaf_flows(leaf_flows=pipelines)
runner.build(output_file=os.path.join(current_dir, "cofea_reader.json"))
