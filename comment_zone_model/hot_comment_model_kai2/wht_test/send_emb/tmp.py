from dragonfly.ext.kgnn.kgnn_api_mixin import KgnnApiMixin
from dragonfly.ext.gsu.gsu_api_mixin import GsuApiMixin
from dragonfly.ext.arrow.arrow_api_mixin import ArrowApiMixin
from dragonfly.ext.cofea.cofea_api_mixin import CofeaApiMixin
from dragonfly.ext.offline.offline_api_mixin import OfflineApiMixin
from dragonfly.ext.kuiba.kuiba_api_mixin import KuibaApiMixin
from dragonfly.ext.mio.mio_api_mixin import MioApiMixin
from dragonfly.common_leaf_dsl import LeafFlow, OfflineRunner
from dragonfly.ext.merchant.merchant_api_mixin import MerchantApiMixin
import os
import sys
import argparse
import json


current_dir = os.path.dirname(__file__)



class DataReaderFlow(LeafFlow, MioApiMixin, OfflineApiMixin, CofeaApiMixin, KuibaApiMixin, ArrowApiMixin, GsuApiMixin, KgnnApiMixin, MerchantApiMixin):
    def clean_all(self, reason, **kwargs):
        return self.limit(0, name="clean_all_for_" + reason, **kwargs)


parser = argparse.ArgumentParser()
# parser.add_argument("--online", dest="online", action="store_true")
# parser.add_argument("--test", dest="test", action="store_true")
parser.add_argument("--data_path", type=str, default='/home/reco_algorithm/dw/reco_algorithm.db/tmp_comment_id_high_vv/p_date=2024070108')    # viewfs://hadoop-lt-cluster/home/reco_algorithm/dw/reco_algorithm.db/tmp_comment_id_high_vv/p_date=2024062001
args = parser.parse_args()



# column_name -> (column_index, column_type)  (column_type 仅支持 int/float/string/int_list/float_list/string_list)
hive_table_column_schema = {
    'comment_id': (0, 'int'),
    'vv': (1, 'int'),
}



"""
[common_attr]
csv_sample_data (string[17]): 844644656360    980
"""
read_data = DataReaderFlow(name="read_data") \
	.fetch_message(
		group_id="001",
		hdfs_path=args.data_path,
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
    .get_remote_embedding_lite(
        kess_service="grpc_HotCommentXtrSevenTargetsEmb",
        id_converter={"type_name": "mioEmbeddingIdConverter"},
        query_source_type="item_attr",
        input_attr_name="comment_id",
        output_attr_name="cmt_emb_vector",
        timeout_ms=50,
        slot=201,
        size=64,
        shard_num=1,
        client_side_shard=True,
    ) \
    .filter_by_attr(
        attr_name="cmt_emb_vector",
        remove_if="==",
        compare_to=0,
        remove_if_attr_missing=True,
    ) \
    .item_list_to_string(
        input_item_attr="cmt_emb_vector",
        output_item_attr="cmt_emb"
    ) \
    .log_debug_info(
        print_all_common_attrs=True,
        print_all_item_attrs=True,
		for_debug_request_only=False,
		respect_sample_logging=True,
        to='file',
        to_file_folder='/data/web_server/project/krp_common_leaf_runner/log/',
        to_file_name='wht_log',
        append_to_file=True,
	) \
    .export_attr_to_kafka(
        kafka_topic="comment_emb",
        item_attrs=['comment_id', 'cmt_emb'],
        single_json=False
    ) \
    # .write_to_csv(
    #     attrs=["comment_id", "cmt_emb"],
    #     path_prefix="/data/web_server/project/krp_common_leaf_runner/output/reco_",
    #     has_header=False
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
        


    
flows = [read_data]
runner = OfflineRunner("comment_model")
runner.IGNORE_UNUSED_ATTR=['cmt_emb']
runner.add_leaf_flows(leaf_flows=flows)
runner.build(output_file=os.path.join(current_dir, "cofea_reader.json"))

