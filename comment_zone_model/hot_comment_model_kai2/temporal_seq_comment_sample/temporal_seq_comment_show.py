# -*- coding: utf-8 -*-
# created by baienyang on 2024/03/20

from dragonfly.common_leaf_dsl import LeafFlow, OfflineRunner
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin


class SendTemporalCommentSample(LeafFlow, OfflineApiMixin):
    """
    发送时序区样本
    """
    def fetch_data(self):
        return self.fetch_message(
            kafka_topic="kscdm_dwd_ks_csm_show_cmt_photo_rt",
            group_id="reco_temporal_comment_show",
            output_attr="show_comment_str",
        ) \
        .parse_protobuf_from_string(
            input_attr="show_comment_str",
            output_attr="show_comment",
            class_name="kuaishou.dp.schema.proto.kscdm.dwdkscsmshowcmtphotort.DwdKsCsmShowCmtPhotoRt",
            use_dynamic_proto=True
        ) \
        .enrich_with_protobuf(
            from_extra_var="show_comment",
            attrs=[
                "user_id",
                "device_id",
                "photo_id",
                "llsid",
                dict(name="photo_author_id", path="author_id"),
                dict(name="common_comment_id", path="comment_id"),
                "comment_user_id",
                "is_second_comment",
                dict(name="common_comment_index", path="comment_index"),
                dict(name="common_recall_type", path="recall_type"),
                "server_timestamp",
                "client_timestamp"
            ]
        ) \
        .perflog_attr_value(
            check_point="comment.attr",
            common_attrs=["is_second_comment", "common_comment_index"],
        ) \
        .if_("is_second_comment == 1") \
            .return_(0) \
        .end_() \
        .perflog_attr_value(
            check_point="comment.recall",
            common_attrs=["common_recall_type"],
            aggregator="count"
        ) \
        .if_("common_recall_type > 0") \
            .return_(0) \
        .end_() \
        .enrich_attr_by_lua(
            import_common_attr=["client_timestamp", "server_timestamp"],
            export_common_attr=["client_minute_diff", "server_minute_diff"],
            function_for_common="calc_minute_diff",
            lua_script="""
                function calc_minute_diff()
                    local cur_ms = util.GetTimestamp() / 1000
                    local client_minute_diff = (cur_ms - client_timestamp) / (60 * 1000.0)
                    local server_minute_diff = (cur_ms - server_timestamp) / (60 * 1000.0)
                    return client_minute_diff, server_minute_diff
                end
            """
        ) \
        .perflog_attr_value(
            check_point="comment.temporal",
            common_attrs=["client_minute_diff", "server_minute_diff"],
        )
    

    def send_data(self):
        return self.retrieve_by_common_attr(
            attr="common_comment_id",
            reason=999,
        ) \
        .copy_item_meta_info(
            save_item_id_to_attr="comment_id"
        ) \
        .copy_attr(
            attrs=[
                {
                    "from_common": "comment_user_id",
                    "to_item": "author_id",
                    "overwrite": True
                },
                {
                    "from_common": "common_recall_type",
                    "to_item": "recall_type",
                    "overwrite": True
                },
                {
                    "from_common": "common_comment_index",
                    "to_item": "comment_index",
                    "overwrite": True
                },
            ]
        ) \
        .get_item_attr_by_distributed_common_index(
            photo_store_kconf_key="cc.knowledgeGraph.hotCommentStoreConfig",
            attrs=[
                {"name": "commentTimestampKV", "as": "timestamp"},
                {"name": "commentRealshowCounterV2", "as": "realshow_cnt"},  # 1h lag
                {"name": "commentContentKV", "as": "comment_content"},
                {"name": "commentContentSplitList", "as": "comment_content_segs"},
                {"name": "commentPunishTagRumaKV", "as": "risk_insult_tag"},
                {"name": "commentPunishTagNegativeKV", "as": "risk_negative_tag"},
                {"name": "commentPunishTagFanNegativeKV", "as": "risk_inactive_tag"},
                {"name": "commentCategoryTagKV", "as": "mmu_category_tag"},
                {"name": "commentEmotionTagKV", "as": "mmu_emotion_tag"},
                {"name": "commentEntityTagList", "as": "mmu_entity_list"},
                {"name": "commentQualityScoreKV", "type": "float", "as": "quality_score"},
                {"name": "commentRelatedScoreKV", "type": "float", "as": "related_score"},
                {"name": "commentPredictLikeScoreKV", "type": "float", "as": "predict_like_score"},
                {"name": "commentPredictReplyScoreKV", "type": "float", "as": "predict_reply_score"},
                {"name": "commentQualityScoreV2KV", "type": "float", "as": "quality_v2_score"},
            ],
        ) \
        .filter_by_rule(
            name="risk_filter",
            rule={
                "join": "or",
                "filters":
                [
                    {
                        "attr_name": "risk_insult_tag",
                        "remove_if": ">",
                        "compare_to": 0
                    },
                    {
                        "attr_name": "risk_negative_tag",
                        "remove_if": ">",
                        "compare_to": 0
                    },
                    {
                        "attr_name": "risk_inactive_tag",
                        "remove_if": ">",
                        "compare_to": 0
                    }
                ]
            }
        ) \
        .count_reco_result(
            save_count_to="risk_filter_cnt",
        ) \
        .enrich_attr_by_lua(
            import_item_attr=["comment_content"],
            export_item_attr=["clean_content"],
            function_for_item="clean",
            lua_script="""
                function clean()

                    local remove_emoji, gsub_n = string.gsub(comment_content or "", "%b[]", "")
                    local remove_at, gsub_n = string.gsub(remove_emoji or "", "@.*%(O%d*%)", "")

                    return remove_at
                end
            """
        ) \
        .log_debug_info(
            item_attrs = [
                "comment_content",
                "clean_content",
            ],
            for_debug_request_only=False,
            respect_sample_logging=True
        ) \
        .filter_by_rule(
            name="content_filter",
            rule={
                "join": "and",
                "filters":
                [
                    {
                        "attr_name": "comment_content",
                        "remove_if": "!=",
                        "compare_to": ""
                    },
                    {
                        "attr_name": "clean_content",
                        "remove_if": "==",
                        "compare_to": ""
                    }
                ]
            }
        ) \
        .count_reco_result(
            save_count_to="content_filter_cnt",
        ) \
        .get_kconf_params(
            kconf_configs=[
                {
                    "kconf_key": "cc.knowledgeGraph.temporalSampleConf",
                    "json_path": "random_filter_threshold",
                    "export_common_attr": "random_filter_threshold",
                    "default_value": 0.0
                }
            ]
        ) \
        .gen_random_item_attr(
            attr_name="sample_random",
            attr_type="double"
        ) \
        .filter_by_rule(
            name="random_filter",
            rule={
                "attr_name": "sample_random",
                "remove_if": "<",
                "compare_to": "{{random_filter_threshold}}"
            }
        ) \
        .count_reco_result(
            save_count_to="random_filter_cnt",
        ) \
        .perflog_attr_value(
            check_point="filter.statis",
            common_attrs=["risk_filter_cnt", "content_filter_cnt", "random_filter_cnt"],
        ) \
        .count_reco_result(
            save_count_to="current_cnt",
        ) \
        .if_("current_cnt <= 0") \
            .return_(0) \
        .end_() \
        .set_attr_value(
            no_overwrite=True,
            item_attrs=[
                {
                    "name": "like_cnt",
                    "type": "int",
                    "value": 0
                },
                {
                    "name": "reply_cnt",
                    "type": "int",
                    "value": 0
                },
                {
                    "name": "dislike_cnt",
                    "type": "int",
                    "value": 0
                },
                {
                    "name": "timestamp",
                    "type": "int",
                    "value": 0
                },
                {
                    "name": "realshow_cnt",
                    "type": "int",
                    "value": 0
                },
                {
                    "name": "comment_content_segs",
                    "type": "string_list",
                    "value": []
                },
                {

                    "name": "comment_genre",
                    "type": "int",
                    "value": 0
                },
                {
                    "name": "risk_insult_tag",
                    "type": "int",
                    "value": 0
                },
                {
                    "name": "risk_negative_tag",
                    "type": "int",
                    "value": 0
                },
                {
                    "name": "risk_inactive_tag",
                    "type": "int",
                    "value": 0
                },
                {
                    "name": "mmu_category_tag",
                    "type": "int",
                    "value": 0
                },
                {
                    "name": "mmu_emotion_tag",
                    "type": "int",
                    "value": 0
                },
                {
                    "name": "mmu_entity_list",
                    "type": "int_list",
                    "value": []
                }
            ]
        ) \
        .enrich_attr_by_lua(
            import_item_attr=["timestamp"],
            export_item_attr=["minute_diff"],
            function_for_item="calc_minute_diff",
            lua_script="""
                function calc_minute_diff()
                    local cur_ts = util.GetTimestamp() / 1000
                    local minute_diff = (cur_ts - timestamp) / (60 * 1000.0)
                    return math.min(minute_diff, 7 * 24 * 60.0)
                end
            """
        ) \
        .reset_user_meta_info(
            timestamp_attr="client_timestamp",
            time_unit="ms",
            user_id_attr="user_id",
            device_id_attr="device_id"
        ) \
        .log_debug_info(
            common_attrs = [
                "llsid",
                "user_id",
                "device_id",
                "photo_id",
                "photo_author_id"
            ],
            item_attrs = [
                "comment_id",
                "comment_index",
                "like_cnt",
                "reply_cnt",
                "dislike_cnt",
                "realshow_cnt",
                "author_id",
                "minute_diff",
                "comment_content_segs",
                "comment_genre",
                "risk_insult_tag",
                "risk_negative_tag",
                "risk_inactive_tag",
                "mmu_category_tag",
                "mmu_emotion_tag",
                "mmu_entity_list",
                "recall_type",
            ],
            for_debug_request_only=False,
            respect_sample_logging=True
        ) \
        .leaf_show(
            enable_leaf_show=True,
            biz_name="hot_comment_leaf",
            producer_type="kafka",
            kafka_topic="reco_temporal_comment_leaf_show",
            send_one_request=True,
            send_item_base_info=False,
            respect_request_num=False,
            use_device_id=True,
            attrs=[
                "comment_id",
                "comment_index",
                "like_cnt",
                "reply_cnt",
                "dislike_cnt",
                "realshow_cnt",
                "author_id",
                "minute_diff",
                "comment_content_segs",
                "comment_genre",
                "risk_insult_tag",
                "risk_negative_tag",
                "risk_inactive_tag",
                "mmu_category_tag",
                "mmu_emotion_tag",
                "mmu_entity_list",
                "recall_type",
                "related_score",
                "quality_score",
                "predict_like_score",
                "predict_reply_score",
                "quality_v2_score",
            ],
            extra_common_attrs=[
                "llsid",
                "user_id",
                "device_id",
                "photo_id",
                "photo_author_id"
            ],
        ) \


    def get_process_flow(self):
        return self.fetch_data() \
            .send_data()


runner=OfflineRunner("send_temporal_comment_sample")
runner.register_proto("dwd_ks_csm_show_cmt_photo_rt.proto",
        """
            syntax = "proto3";

            option java_package = "com.kuaishou.protobuf.dp.schema.proto.kscdm.dwdkscsmshowcmtphotort";
            option java_multiple_files = true;
            package kuaishou.dp.schema.proto.kscdm.dwdkscsmshowcmtphotort;
            option java_outer_classname="DwdKsCsmShowCmtPhotoRtFileOuterClass";

            message DwdKsCsmShowCmtPhotoRt{
                string product = 1;
                string device_id = 2;
                string global_id = 3;
                int64 client_timestamp = 4;
                int64 server_timestamp = 5;
                int64 user_id = 6;
                int64 photo_id = 7;
                int64 author_id = 8;
                string llsid = 9;
                string exp_tag = 10;
                string content_source_exp = 11;
                string content_source_page_tag = 12;
                int64 is_second_comment = 13;
                string ab_mapping_ids = 14;
                int64 comment_id = 15;
                int64 comment_user_id = 16;
                int64 comment_index = 17;
	            int64 recall_type = 18;
            }

        """
    )

leaf_flows=[
    SendTemporalCommentSample(name="send_temporal_comment_flow").get_process_flow(),
]

runner.add_leaf_flows(leaf_flows=leaf_flows)
runner.build(__file__.replace(".py", "") + ".json")
