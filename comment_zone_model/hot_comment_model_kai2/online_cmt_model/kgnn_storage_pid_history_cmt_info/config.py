# coding=utf-8

import json
import copy

"""
容器部署的配置示例，注意 py 代码与物理机部署的有所不同
物理机部署时是将 py 代码提供给部署平台, 而容器部署的使用方法是
** 直接在本地运行该脚本，然后将生成的 json 提供给平台(kaiserving) **
如果 part_configs 里有 N 个 part, 则会对每个 part 生成一个 json, 部署时需要部署 N 个服务, 以确保每个服务只有一个 kess_name
开始部署后，每个容器会获得一个平台分发的 SHARD_CONFIG.json 文件，服务会读取该文件以及上述的 json 文件
"""


###############  用户配置项目  ###################

# 实验名称.
experiment_name = "pid_history_cmt_info"
# 线上单台机器最大可用 shm 内存, 一般来说混部容器内存共 248GB, 需要留一些余量给存储之外的各种用途
shm_size = (1 << 30) * 210

# 每个 item 代表一个 storage 的配置.
part_configs = [
    {
        "shard_num": 2,
        "dbs": ["U2I"],
        "checkpoint": {
            "checkpoint_path": "/home/reco/kgnn/" + experiment_name,
            "reserve_days": 1,
            "save_interval_s": 21600,
        }
    }
]

relations = {
    "U2I": {
        "total_memory": (1 << 30) * 400,
        "key_size": 120_000_000,
        "edge_max_num": 1000,
        # 边长度过长后的替换策略. 0=淘汰最老, 1=随机替换, 2=按权重淘汰最小, 3=取消插入
        "oversize_replace_strategy": 0,
        # 节点过期时间, 单位秒, 0 = 不过期.
        "expire_interval": 30 * 86400,
        "elst": "cpt",
        "edge_expire_interval_s": 3600 * 24 * 30,
        # 权重/出度 衰减相关配置
        # 边权重衰减比例, 默认为 1(不衰减)
        "weight_decay_ratio": 1,
        # 衰减间隔(s), 默认为 86400(1天一次)
        "decay_interval_s": 86400,
        # 边的权重衰减到该值时，淘汰掉此边，默认为 0.
        "delete_threshold_weight": 0,
        # 点属性存储, relation 上的点带有属性值则需要配置这个
        # 每个点能携带一个定长的 int64 list 和一个 float list.
        # 注意：如果修改了这项，就不能直接加载老 ckpt，否则内存写乱，你会看到服务在奇怪的地方 check fail
        "edge_attr_op_config": {
            "type_name": "SimpleAttrBlock",
            "int_max_size": 1,    # 存储 mmu 标签
            "float_max_size": 0
        },
    }
}

###############  用户配置项目  ###################


# 以上仅包含部分常用配置, 更多详细可选配置参考：https://git.corp.kuaishou.com/reco-arch/kgnn/-/wikis/Storage-config/Example

# 可调节次级配置.
# 存储的 multi mem kv 的 shard 数, 代表可并行写的并行度.
memkv_shard_num = 64

def error_exit(msg):
  # 如果检查不过, 返回的错误信息.
  backup_debug_msg = {"error_msg": msg + ", 请检查相关配置"}
  print(backup_debug_msg)
  exit()

def translate_db_config(relation_configs):
  for k, v in relation_configs.items():
    v["relation_name"] = k
    # 每个 memkv 的 key 数量
    v["kv_dict_size"] = int(v["key_size"] / memkv_shard_num)
    # 每个 memkv 的内存占用量
    v["kv_size"] = int(v["total_memory"] / memkv_shard_num)
    v["kv_shard_num"] = memkv_shard_num
    v.pop("key_size")
    v.pop("total_memory")
  return relation_configs

def item_name(dbs, shard_id):
  # kess 前缀.
  kess_prefix = "grpc_kgnn_{}".format(experiment_name)
  return "-".join([kess_prefix] + dbs + [str(shard_id)])


def rpc_service_name(dbs):
  # kess 前缀.
  kess_prefix = "grpc_kgnn_{}".format(experiment_name)
  return "-".join([kess_prefix] + dbs)

def update_btq_name(dbs, shard_id):
  # kess 前缀.
  btq_prefix = "btq_kgnn_{}".format(experiment_name)
  return "-".join([btq_prefix] + dbs + [str(shard_id)])

def dump_part_json(item, all_db_config):
  relation_names = item["dbs"]
  global_shard_config = {}
  final_config = {}
  final_config["db_list"] = {}
  for k, v in all_db_config.items():
    if k in relation_names:
      final_config["db_list"][k] = v
  final_config["service_config"] = {"default_rpc_thread_num": 64}

  shard_num = item["shard_num"] if "shard_num" in item else 1
  for shard_id in range(shard_num):
    global_shard_config[shard_id] = item_name(item["dbs"], shard_id)
    cp_item = copy.deepcopy(item)
    cp_item["service_name"] = rpc_service_name(item["dbs"])
    cp_item["exp_name"] = experiment_name
    cp_item["shard_id"] = shard_id
    cp_item["shard_num"] = shard_num
    # stream loader
    if "stream_loader" in cp_item:
      cp_item["stream_loader"]["queue_name"] = update_btq_name(item["dbs"], shard_id)
    final_config["service_config"].update({item_name(item["dbs"], shard_id): cp_item})

  final_config["global_shard_config"] = global_shard_config

  file_name = experiment_name+"-"+"-".join(relation_names)+".json"
  print("Generate file:", file_name)
  with open(file_name, "w") as f:
    json.dump(final_config, f, indent=2)


def main():
  # check 内存分配是否超额度
  for part in part_configs:
    shard = part['shard_num']
    mem = sum([relations[db]["total_memory"] for db in part["dbs"]])
    if mem / shard > shm_size:
      error_exit(f"part with db: {part['dbs']} mem oversize: limit {shm_size}, config: {mem}")

  all_db_config = translate_db_config(relations)
  for part in part_configs:
    dump_part_json(part, all_db_config)


if __name__ == "__main__":
  main()

