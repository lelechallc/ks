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


common_attrs=["user_id", "device_id", "photo_id", "photo_author_id", "gender", "age_segment", "llsid", "request_time"]
int_item_attrs=["comment_id", "author_id", "like_cnt", "reply_cnt", "realshow_cnt", "dislike_cnt",
            "showAction", "expandAction", "replyAction", "likeAction", "audienceAction", "reportAction",
            "comment_genre", "risk_insult_tag", "risk_negative_tag", "risk_inactive_tag", "mmu_category_tag", "mmu_emotion_tag",
            "subShowCntAction", "expandAction_first", "likeAction_first", "likeAction_second", "replyAction_first",
            "replyAction_second", "copyAction_first", "copyAction_second", "shareAction_first",
            "shareAction_second", "audienceAction_first", "audienceAction_second",
            "stayDurationMs", "stayDurationMs_second",
            "comment_stay_time"]
int_list_item_attrs=["mmu_entity_list"]
double_item_attrs=["minute_diff", "like_pxtr", "reply_pxtr", "expand_pxtr", "continuous_expand_pxtr",
            "copy_pxtr", "share_pxtr", "audience_pxtr",]

item_attrs = int_item_attrs + int_list_item_attrs + double_item_attrs

SEQ_SIZE = 20
read_data = DataReaderFlow(name="read_data") \
    .fetch_message(
      group_id="hot_comment_rerank_evaluator",
      kafka_topic="reco_all_comment_join_listwise_sample",
      output_attr="raw_context_str",
    ) \
    .parse_context(
      parse_from_attr="raw_context_str",
      extract_common_attrs=common_attrs,
      extract_item_results=True,
      extract_item_attrs=item_attrs,
    ) \
    .copy_item_meta_info(
        save_item_key_to_attr="comment_id",
    ) \
    .set_attr_default_value(
        item_attrs=[
            {
                "name": name,
                "type": "int",
                "value": 0
            } for name in int_item_attrs
        ] +
        [
            {
                "name": name,
                "type": "int_list",
                "value": []
            } for name in int_list_item_attrs
        ] +
        [
            {
                "name": name,
                "type": "double",
                "value": 0.0
            } for name in double_item_attrs
        ]
    ) \
    .perflog_attr_value(
        check_point="comment.listwise",
        common_attrs=common_attrs,
        item_attrs=item_attrs,
    ) \
    .log_debug_info(
        for_debug_request_only=False,
        respect_sample_logging=True,
        common_attrs=common_attrs,
        item_attrs=item_attrs
    ) \
    .count_reco_result(
        select_item = {
          "attr_name": "showAction",
          "compare_to": 0,
          "select_if": ">",
          "select_if_attr_missing": False
        },
        save_count_to="show_item_cnt") \
    .if_("show_item_cnt <= 0") \
        .return_(0) \
    .end_() \
    .perflog_attr_value(
        check_point="comment.listwise",
        common_attrs=["show_item_cnt"],
    ) \
    .enrich_attr_by_lua(
        import_common_attr=["request_time"],
        export_common_attr=["time_ms", "sample_minute_diff"],
        function_for_common="cal",
        lua_script="""
            function cal()
                local time_ms = request_time
                local sample_minute_diff = (util.GetTimestamp() / 1000 - request_time) / 1000 / 60.0
                return time_ms, sample_minute_diff
            end
        """
    ) \
    .enrich_attr_by_lua(
        import_item_attr=["showAction"],
        export_item_attr=["show"],
        function_for_item="check_show",
        lua_script='''
            function check_show()
                local show = (showAction or 0) > 0
                return show
            end
        '''
    ) \
    .enrich_attr_by_lua(
        import_item_attr=["expandAction", "likeAction", "replyAction",
                          "expandAction_first", "likeAction_first", "likeAction_second", "replyAction_first",
            "replyAction_second", "copyAction_first", "copyAction_second", "shareAction_first",
            "shareAction_second", "audienceAction_first", "audienceAction_second",
            "stayDurationMs", "stayDurationMs_second", "show"],
        export_item_attr=["expand", "like", "reply", "copy", "share", "audience", "continuous", "staytime",
                          "expand_v", "like_v", "reply_v", "copy_v", "share_v", "audience_v", "continuous_v", "staytime_v", "show_v"],
        function_for_item="transform",
        lua_script='''
            function transform()
                local expand = (expandAction_first > 0) and 1 or 0
                local continuous = (expandAction_first > 1) and 1 or 0
                local like = (likeAction_first > 0 or likeAction_second > 0) and 1 or 0
                local reply = (replyAction_first > 0 or replyAction_second > 0) and 1 or 0
                local copy = (copyAction_first > 0 or copyAction_second > 0) and 1 or 0
                local share = (shareAction_first > 0 or shareAction_second > 0) and 1 or 0
                local audience = (audienceAction_first > 0 or audienceAction_second > 0) and 1 or 0
                local staytime = stayDurationMs + stayDurationMs_second

                return expand, like, reply, copy, share, audience, continuous, staytime,
                    expand * 1.0, like * 1.0, reply * 1.0, copy * 1.0, share * 1.0, audience * 1.0, continuous * 1.0, staytime * 1.0,
                    show * 1.0
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
      import_item_attr=["showAction", "expandAction", "replyAction", "likeAction", "audienceAction", "reportAction",
                    "expand", "like", "reply", "copy", "share", "audience", "continuous", "staytime"],
      export_item_attr=["calc_weight"],
      function_for_item="calc",
      lua_script="""
        function calc()
          local weight = 1.0
          if (expand or 0) > 0 then
            weight = weight + 3.0
          end
          if (reply or 0) > 0 then
            weight = weight + 5.0
          end
          if (like or 0) > 0 then
            weight = weight + 3.0
          end
          if (copy or 0) > 0 then
            weight = weight + 3.0
          end
          if (share or 0) > 0 then
            weight = weight + 3.0
          end
          if (audience or 0) > 0 then
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
    .enrich_attr_by_lua(
        import_item_attr=["like_pxtr", "reply_pxtr", "expand_pxtr", "continuous_expand_pxtr",
                          "copy_pxtr", "share_pxtr", "audience_pxtr"],
        export_item_attr=["like_pxtr_dis", "reply_pxtr_dis", "expand_pxtr_dis", "continuous_expand_pxtr_dis",
                          "copy_pxtr_dis", "share_pxtr_dis", "audience_pxtr_dis"],
        function_for_item="get_pxtr_discreate",
        lua_script='''
            function get_pxtr_discreate()

                local like_pxtr_dis = math.max(math.min(like_pxtr // 0.001, 1000), 0) * 1
                local reply_pxtr_dis = math.max(math.min(reply_pxtr // 0.001, 1000), 0) * 1
                local expand_pxtr_dis = math.max(math.min(expand_pxtr // 0.001, 1000), 0) * 1
                local continuous_expand_pxtr_dis = math.max(math.min(continuous_expand_pxtr // 0.001, 1000), 0) * 1
                local copy_pxtr_dis = math.max(math.min(copy_pxtr // 0.001, 1000), 0) * 1
                local share_pxtr_dis = math.max(math.min(share_pxtr // 0.001, 1000), 0) * 1
                local audience_pxtr_dis = math.max(math.min(audience_pxtr // 0.001, 1000), 0) * 1

                return math.ceil(like_pxtr_dis), math.ceil(reply_pxtr_dis), math.ceil(expand_pxtr_dis),
                       math.ceil(continuous_expand_pxtr_dis), math.ceil(copy_pxtr_dis), math.ceil(share_pxtr_dis),
                       math.ceil(audience_pxtr_dis)
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
                "from_item_attr": "copy_v",
                "to_common_attr": "copy_v_pack",
            },
            {
                "from_item_attr": "share_v",
                "to_common_attr": "share_v_pack",
            },
            {
                "from_item_attr": "audience_v",
                "to_common_attr": "audience_v_pack",
            },
            {
                "from_item_attr": "continuous_v",
                "to_common_attr": "continuous_v_pack",
            },
            {
                "from_item_attr": "staytime_v",
                "to_common_attr": "staytime_v_pack",
            },
            {
                "from_item_attr": "show_v",
                "to_common_attr": "show_v_pack",
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
                "from_item_attr": "like_pxtr_dis",
                "to_common_attr": "like_pxtr_dis_list",
            },
            {
                "from_item_attr": "reply_pxtr_dis",
                "to_common_attr": "reply_pxtr_dis_list",
            },
            {
                "from_item_attr": "expand_pxtr_dis",
                "to_common_attr": "expand_pxtr_dis_list",
            },
            {
                "from_item_attr": "continuous_expand_pxtr_dis",
                "to_common_attr": "continuous_expand_pxtr_dis_list",
            },
            {
                "from_item_attr": "copy_pxtr_dis",
                "to_common_attr": "copy_pxtr_dis_list",
            },
            {
                "from_item_attr": "share_pxtr_dis",
                "to_common_attr": "share_pxtr_dis_list",
            },
            {
                "from_item_attr": "audience_pxtr_dis",
                "to_common_attr": "audience_pxtr_dis_list",
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

          **list_config("like_pxtr_dis_list", 241, 241, SEQ_SIZE),
          **list_config("reply_pxtr_dis_list", 242, 242, SEQ_SIZE),
          **list_config("expand_pxtr_dis_list", 243, 243, SEQ_SIZE),
          **list_config("continuous_expand_pxtr_dis_list", 244, 244, SEQ_SIZE),
          **list_config("copy_pxtr_dis_list", 245, 245, SEQ_SIZE),
          **list_config("share_pxtr_dis_list", 246, 246, SEQ_SIZE),
          **list_config("audience_pxtr_dis_list", 247, 247, SEQ_SIZE),
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
        attrs=["sample_weight_pack", "expand_v_pack", "like_v_pack", "reply_v_pack", 
               "copy_v_pack", "share_v_pack", "audience_v_pack", "continuous_v_pack", "staytime_v_pack",
               "show_v_pack", "mask_pack"],
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
