from dragonfly.ext.cofea.cofea_api_mixin import CofeaApiMixin
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin
from dragonfly.ext.kgnn.kgnn_api_mixin import KgnnApiMixin
from dragonfly.common_leaf_dsl import LeafFlow, OfflineRunner
from dragonfly.ext.embedding.embedding_api_mixin import EmbeddingApiMixin
from dragonfly.ext.gsu.gsu_api_mixin import GsuApiMixin

import os

current_dir = os.path.dirname(__file__)


class DataReaderFlow(LeafFlow, MioApiMixin, OfflineApiMixin, CofeaApiMixin, KuibaApiMixin, KgnnApiMixin, EmbeddingApiMixin, GsuApiMixin):
    def clean_all(self, reason, **kwargs):
        return self.limit(0, name="clean_all_for_" + reason, **kwargs)


# 要从 kafka 中读取的特征
common_attrs=["user_id", "device_id", "photo_id", "photo_author_id", "gender", "age_segment", 
              "mod", "page_type_str",
]

labels=["expandAction_first", "replyAction_first", "likeAction_first", "audienceAction_first", "reportAction_first",
        "expandAction_second", "replyAction_second", "likeAction_second", "audienceAction_second",
        "copyAction", "copyAction_first", "copyAction_second", "shareAction", "shareAction_first", "shareAction_second",
        "cancelHateAction_first", "cancelHateAction_second", 'cancelLikeAction_first', 'cancelLikeAction_second',
        'hateAction_first', 'hateAction_second', 'replyTaskAction_first', 'replyTaskAction_second', 
        'subAtAction_first', 'subAtAction_second', "stayDurationMs", "subShowCntAction",
]

item_attrs=["comment_id", "author_id", "like_cnt", "reply_cnt", "minute_diff", "realshow_cnt", "dislike_cnt",
            "showAction", "recall_type",
            "comment_genre", 'content_length','comment_content'
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
    .extract_kuiba_parameter(
        config={
            **id_config("gender", 101),     # dim=4
            **id_config("age_segment", 102),
        
            **id_config("photo_id", 103),   # dim=64
            **id_config_slot("photo_author_id", 104, 202),
            **id_config_slot("user_id", 105, 202),
            **id_config("device_id", 106),

            # ## new feature
            **id_config("mod", 110),        # dim=32
            **id_config("page_type_str", 111),  

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
            **id_config("comment_genre", 250),      # dim=8
            **discreate_config("content_length", 251, [5, 0, 1000, 1, 0]),      # dim=32

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
    .enrich_attr_by_lua(
        import_item_attr = ["comment_content"],
        function_for_item="cal",
        export_item_attr = ["comment_content_list"],
        lua_script="""
            function cal()
                local comment_content_list = {}
                -- local comment_content = "Keras是ONEIROS（Open-ended Neuro-Electronic Intelligent Robot Operating System，开放式神经电子智能机器人操作系统）项目研究工作的部分产物[3]，主要作者和维护者是Google工程师François Chollet。"
                table.insert(comment_content_list, comment_content)
                return comment_content_list
            end
        """
    ) \
    .gsu_bert_tokenization(
            sentence_list_attr = "comment_content_list",
            output_id_attr = "token_output_id",
            output_token_attr = "token_output",
            output_mask_attr = "token_output_mask",
            sentence_len_limit = 25,
            list_len_limit = 25,
            is_common_attr = False
        ) \
    .enrich_attr_by_lua(
            import_item_attr = ["comment_content", "token_output_id", "token_output", "token_output_mask"],
            function_for_item="cal",
            export_item_attr = ["token_input_ids", "token_input_mask", "token_sep_ids"],
            lua_script="""
                function cal()
                    local token_output = token_output or {}
                    local token_output_id = token_output_id or {}
                    local token_output_mask = token_output_mask or {}
                    local token_sep_ids = {}

                    -- ids
                    local s_end = 0
                    table.insert(token_output_id, 1, 101)
                    for i = 1, #token_output_id do
                        s_end = s_end + 1
                        if token_output_id[i] == 1 then
                            break  
                        end
                    end
                    if s_end ==  #token_output_id then
                        table.insert(token_output_id, 102)
                    elseif s_end > 0 then
                        table.insert(token_output_id, s_end, 102)
                        for i = s_end + 1, #token_output_id do
                            if token_output_id[i] == 1 then
                                token_output_id[i]= 0
                            end
                        end
                    end

                    -- mask
                    local m_end = 0
                    table.insert(token_output_mask, 1, 1)
                    for i = 1, #token_output_mask do
                        m_end = m_end + 1
                        if token_output_mask[i] == 0 then
                            token_output_mask[i] = 1
                        elseif token_output_mask[i]==-99999 then
                            break
                        end
                    end
                    if m_end ==  #token_output_id then
                        table.insert(token_output_mask, 1)
                    elseif m_end > 0 then
                        table.insert(token_output_mask, m_end, 1)
                        for i = m_end + 1, #token_output_mask do
                            if token_output_mask[i]==-99999 then
                                token_output_mask[i] = 0
                            end
                        end
                    end

                    for i = 1, #token_output_mask do
                        table.insert(token_sep_ids, 0)
                    end
                    -- local token_output_id_str = table.concat(token_output_id, ",")
                    -- local token_output_str = table.concat(token_output, ",")
                    -- local token_output_mask_str = table.concat(token_output_mask, ",")
                    -- local token_sep_ids_str  = table.concat(token_sep_ids, ",")
                    -- if comment_content ~= nil then
                        -- print("content_id_output_mask_sepids:", "\t", comment_content, "\t", token_output_id_str, "\t", token_output_str, "\t", token_output_mask_str, "\t", token_sep_ids_str)
                    -- end
                    return token_output_id, token_output_mask, token_sep_ids
                end
            """
    ) \
    .log_debug_info(
            item_attrs = ['comment_content','token_input_ids', 'token_input_mask', 'token_output', 'token_sep_ids'],
            for_debug_request_only=False,
            respect_sample_logging=False,
    ) \

gen_new_feature = DataReaderFlow(name="gen_new_feature") \
    .fetch_remote_embedding(    
        protocol=1,
        id_converter = {"type_name": "mioEmbeddingIdConverter"},
        colossusdb_embd_service_name="mmu-bert-emb",
        colossusdb_embd_table_name="mmu_bert_emb_table",
        query_source_type="item_attr",
        input_attr_name="comment_id",
        output_attr_name ="bert_first5_layers_emb",
        slot=0,
        size=27*256,
        enable_smaller_size=True,
        max_signs_per_request=500,
        timeout_ms=20,
    ) \
    .log_debug_info(
        item_attrs = ['bert_first5_layers_emb'],
        for_debug_request_only=False,
        respect_sample_logging=False,
    ) \
    

send_mio = DataReaderFlow(name="send_mio") \
    .count_reco_result(save_count_to="send_sample_cnt") \
    .perflog_attr_value(
        check_point="send.mio",
        item_attrs=labels,
        common_attrs=["send_sample_cnt"]
    ) \
    .send_to_mio_learner(
        attrs=labels + ['recall_type', 'comment_genre'] + ["token_input_ids", "token_input_mask", "token_sep_ids", "bert_first5_layers_emb"],
        # attrs=labels + ['recall_type', 'comment_genre'] + ['bert_first5_layers_emb'],
        slots_attrs=["comment_common_slots", "comment_item_slots"],
        signs_attrs=["comment_common_signs", "comment_item_signs"],
        lineid_attr="user_id",
        time_ms_attr="time_ms",
        label_attr="likeAction_first",
        user_hash_attr="user_photo_hash"
    )

flows = [read_data, gen_feature, gen_new_feature, send_mio]
runner = OfflineRunner("comment_profile")
# runner.CHECK_UNUSED_ATTR=False        # 这个不要设置为False，否则不便于发现特征缺失
runner.add_leaf_flows(leaf_flows=flows)
runner.build(output_file=os.path.join(current_dir, "cofea_reader.json"))
