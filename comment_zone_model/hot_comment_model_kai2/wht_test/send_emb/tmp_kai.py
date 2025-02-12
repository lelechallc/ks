from dragonfly.ext.kgnn.kgnn_api_mixin import KgnnApiMixin
from dragonfly.ext.gsu.gsu_api_mixin import GsuApiMixin
from dragonfly.ext.arrow.arrow_api_mixin import ArrowApiMixin
from dragonfly.ext.cofea.cofea_api_mixin import CofeaApiMixin
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.ext.common.common_api_mixin import CommonApiMixin
from dragonfly.common_leaf_dsl import LeafFlow, OfflineRunner
import os
import sys
import argparse
import json


current_dir = os.path.dirname(__file__)



class DataReaderFlow(LeafFlow, MioApiMixin, OfflineApiMixin, CofeaApiMixin, KuibaApiMixin, ArrowApiMixin, GsuApiMixin, KgnnApiMixin, CommonApiMixin):
    def clean_all(self, reason, **kwargs):
        return self.limit(0, name="clean_all_for_" + reason, **kwargs)


parser = argparse.ArgumentParser()
# parser.add_argument("--online", dest="online", action="store_true")
# parser.add_argument("--test", dest="test", action="store_true")
parser.add_argument("--data_path", type=str, default='/home/reco_algorithm/dw/reco_algorithm.db/tmp_comment_id_high_vv/p_date=2024062610')    # viewfs://hadoop-lt-cluster/home/reco_algorithm/dw/reco_algorithm.db/tmp_comment_id_high_vv/p_date=2024062610
args = parser.parse_args()



# column_name -> (column_index, column_type)  (column_type 仅支持 int/float/string/int_list/float_list/string_list)
hive_table_column_schema = {
    'comment_id': (0, 'int'),
    'vv': (1, 'int'),
}

id_config = lambda attr_name, slot: {
  attr_name: {"attrs": [{"mio_slot_key_type": slot, "key_type": slot, "attr": [attr_name], "converter": "id"}]},
}



"""
[common_attr]
csv_sample_data (string[17]): 844644656360    980
"""
read_data = DataReaderFlow(name="read_data") \
	.fetch_message(
		group_id="001",
		hdfs_path='viewfs://hadoop-lt-cluster/home/reco_algorithm/dw/reco_algorithm.db/tmp_comment_id_high_vv/p_date=2024070112',
		hdfs_format="raw_text",
		output_attr="csv_sample_data",  
	) \
    .convert_csv_to_tf_sequence_example(
        from_extra_var="csv_sample_data",
        item_attrs=[
            dict(column_index=c_idx, column_name=c_name, type=c_type) for c_name, (c_idx, c_type) in hive_table_column_schema.items()
        ],
        column_separator="\t",
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
        save_count_to="retrieve_cnt"
    ) \
    .perflog_attr_value(
        check_point="wht.data.process",
        common_attrs=["retrieve_cnt"],
    ) \
    .get_remote_embedding_lite(
        kess_service="grpc_HotCommentXtrSevenTargetsEmb",
        id_converter={"type_name": "mioEmbeddingIdConverter"},
        query_source_type="item_attr",
        input_attr_name="comment_id",
        output_attr_name="cmt_emb",
        timeout_ms=50,
        slot=201,
        size=64,
        shard_num=1,
        client_side_shard=True,
    ) \
    .set_attr_default_value(
        item_attrs=[
            {
                'name': 'like_action',
                'type': 'int',
                'value': 0
            },
        ]
    ) \
    .filter_by_attr(
        attr_name="cmt_emb",
        remove_if="==",
        compare_to=0,
        remove_if_attr_missing=True,
    ) \
    .extract_kuiba_parameter(
        config={
            **id_config("comment_id", 201),     # dim=64
        },
        is_common_attr=False,
        slots_output="comment_item_slots",
        parameters_output="comment_item_signs",
    ) \
    .export_attr_to_kafka(
        kafka_topic="comment_emb",
        item_attrs=['comment_id', 'cmt_emb'],
        single_json=False
    ) \
    # .write_to_hdfs(
    #     source_attr="comment_id",
    #     is_common=False,
    #     hdfs_path_prefix="viewfs://hadoop-lt-cluster/home/reco_wl/mpi/updown_center/kai2/wuhongtao/tmp_cid",
    # ) \
    # .write_to_hdfs(
    #     source_attr="cmt_emb",
    #     is_common=False,
    #     hdfs_path_prefix="viewfs://hadoop-lt-cluster/home/reco_wl/mpi/updown_center/kai2/wuhongtao/tmp_tmb",
    # ) \
    # .log_debug_info(
    #     print_all_common_attrs=True,
    #     print_all_item_attrs=True,
	# 	for_debug_request_only=False,
	# 	respect_sample_logging=True,
    #     to='file',
    #     to_file_folder='/data/web_server/project/krp_common_leaf_runner/log/',
    #     to_file_name='wht_log',
    #     # append_to_file=True,
	# ) \
    # .write_to_csv(
    #     attrs=["comment_id", "cmt_emb"],
    #     path_prefix="/data/web_server/project/krp_common_leaf_runner/log/files/wht_",
    # ) \
    # .filter_by_attr(
    #     # 过滤掉无曝光的样本，缺少这个字段也会被过滤
    #     attr_name="cmt_emb",
    #     remove_if="==",
    #     compare_to=[],
    #     remove_if_attr_missing=True
    # ) \
    # .write_to_hdfs(
    #     source_attr="comment_id",
    #     is_common=False,
    #     hdfs_path_prefix="/home/reco_wl/mpi/updown_center/kai2/wuhongtao/tmp_cid",
    # ) \
    # .write_to_hdfs(
    #     source_attr="cmt_emb",
    #     is_common=False,
    #     hdfs_path_prefix="/home/reco_wl/mpi/updown_center/kai2/wuhongtao/tmp_tmb",
    # ) \
        
    
    
send_mio = DataReaderFlow(name="send_mio") \
    .send_to_mio_learner(
        # 在这里就会丢失photo hash的int64精度，mio不支持int64
        attrs=["comment_id", "cmt_emb", "like_action", "vv"],
        slots_attrs=["comment_common_slots", "comment_item_slots"],
        signs_attrs=["comment_common_signs", "comment_item_signs"],
        time_ms_attr="time_ms",
        label_attr="like_action",   # 不重要
    ) \
    # .debug_log(print_all_common_attrs=True, print_all_item_attrs=True, to="stdout")\
    
    


    
flows = [read_data, send_mio]
runner = OfflineRunner("comment_model")
runner.IGNORE_UNUSED_ATTR=['cmt_emb']
runner.add_leaf_flows(leaf_flows=flows)
runner.build(output_file=os.path.join(current_dir, "cofea_reader.json"))

