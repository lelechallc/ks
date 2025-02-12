kuiba_list_converter_config_list_limit = lambda limit_n:  {
  "converter": "list",
  "type":5,
  "converter_args": {
    "reversed": False,
    "enable_filter": False,
    "limit": limit_n,
  },
 }

kuiba_discrete_converter = lambda denominator, smooth, max_val, buckets, min_val: {
  "converter": "discrete",
  "converter_args": f"{denominator},{smooth},{max_val},{buckets},{min_val}"
}
xtr_config = lambda attr_name, slot: {
  attr_name: {"attrs": [{"mio_slot_key_type": slot, "key_type": slot, "attr": [attr_name], **kuiba_discrete_converter(0.01, 0, 1, 100, -1)}]}
}

cnt_config = lambda attr_name, slot: {
  attr_name: {"attrs": [{"mio_slot_key_type": slot, "key_type": slot, "attr": [attr_name], **kuiba_discrete_converter(1000, 0, 100000, 1, 0)}]},
}

cmt_kgnn_config = lambda attr_name, slot: {
  attr_name: {"attrs": [{"mio_slot_key_type": slot, "key_type": slot, "attr": [attr_name], **kuiba_discrete_converter(5, 0, 100, 1, 0)}]},
}

kuiba_id_converter = {
  "converter": "id"
}

id_config = lambda attr_name, slot: {
  attr_name: {"attrs": [{"mio_slot_key_type": slot, "key_type": slot, "attr": [attr_name], **kuiba_id_converter}]},
}

list_config = lambda attr_name, mio_slot_key_type, key_type, limit: {
  attr_name: {"attrs": [{"mio_slot_key_type": mio_slot_key_type, "key_type": key_type, "attr": [attr_name], **kuiba_list_converter_config_list_limit(limit)}]},
}

pxtr_config = lambda id: {
  f"pctr_{id}": {"attrs": [{"key_type": 1001 + id, "attr": ["pctr"], "converter": "discrete", "converter_args": "1,0,1,200,-1",}]},
  f"pltr_{id}": {"attrs": [{"key_type": 1002 + id, "attr": ["pltr"], "converter": "discrete", "converter_args": "0.2,0,1,400,-1",}]},
  f"pcltr_{id}": {"attrs": [{"key_type": 1003 + id, "attr": ["pcltr"], "converter": "discrete", "converter_args": "0.2,0,1,600,-1",}]},
  f"pftr_{id}": {"attrs": [{"key_type": 1004 + id, "attr": ["pftr"], "converter": "discrete", "converter_args": "0.2,0,1,800,-1",}]},
  f"pwtr_{id}": {"attrs": [{"key_type": 1005 + id, "attr": ["pwtr"], "converter": "discrete", "converter_args": "0.2,0,1,600,-1",}]},
  f"plvtr_{id}": {"attrs": [{"key_type": 1006 + id, "attr": ["plvtr"], "converter": "discrete", "converter_args": "1,0,1,200,-1",}]},
  f"pvtr_{id}": {"attrs": [{"key_type": 1007 + id, "attr": ["pvtr"], "converter": "discrete", "converter_args": "1,0,1,200,-1",}]},
  f"pptr_{id}": {"attrs": [{"key_type": 1008 + id, "attr": ["pptr"], "converter": "discrete", "converter_args": "0.2,0,1,600,-1",}]},
  f"pcmtr_{id}": {"attrs": [{"key_type": 1009 + id, "attr": ["pcmtr"], "converter": "discrete", "converter_args": "0.2,0,1,600,-1",}]},
  f"phtr_{id}": {"attrs": [{"key_type": 1010 + id, "attr": ["phtr"], "converter": "discrete", "converter_args": "0.2,0,1,1000,-1",}]},
  f"pepstr_{id}": {"attrs": [{"key_type": 1011 + id, "attr": ["pepstr"], "converter": "discrete", "converter_args": "0.2,0,1,800,-1",}]},
  f"pcmef_{id}": {"attrs": [{"key_type": 1012 + id, "attr": ["pcmef"], "converter": "discrete", "converter_args": "0.2,0,1,400,-1",}]},
  f"pwtd_{id}": {"attrs": [{"key_type": 1013 + id, "attr": ["pwtd"], "converter": "discrete", "converter_args": "0.5,0,100,1,0",}]},

  f"empirical_ctr_{id}": {"attrs": [{"key_type": 1014 + id, "attr": ["empirical_ctr"], "converter": "discrete", "converter_args": "1,0,1,200,-1",}]},
  f"empirical_ltr_{id}": {"attrs": [{"key_type": 1015 + id, "attr": ["empirical_ltr"], "converter": "discrete", "converter_args": "0.2,0,1,400,-1",}]},
  f"empirical_ftr_{id}": {"attrs": [{"key_type": 1016 + id, "attr": ["empirical_ftr"], "converter": "discrete", "converter_args": "0.2,0,1,800,-1",}]},
  f"empirical_wtr_{id}": {"attrs": [{"key_type": 1017 + id, "attr": ["empirical_wtr"], "converter": "discrete", "converter_args": "0.2,0,1,600,-1",}]},
  f"empirical_ptr_{id}": {"attrs": [{"key_type": 1018 + id, "attr": ["empirical_ptr"], "converter": "discrete", "converter_args": "0.2,0,1,400,-1",}]},
  f"empirical_htr_{id}": {"attrs": [{"key_type": 1019 + id, "attr": ["empirical_htr"], "converter": "discrete", "converter_args": "0.2,0,1,1000,-1",}]},
  f"empirical_cmtr_{id}": {"attrs": [{"key_type": 1020 + id, "attr": ["empirical_cmtr"], "converter": "discrete", "converter_args": "0.2,0,1,600,-1",}]},

  f"cascade_pctr_{id}": {"attrs": [{"key_type": 1021 + id, "attr": ["cascade_pctr"], "converter": "discrete", "converter_args": "1,0,1,200,-1",}]},
  f"cascade_plvtr_{id}": {"attrs": [{"key_type": 1022 + id, "attr": ["cascade_plvtr"], "converter": "discrete", "converter_args": "1,0,1,200,-1",}]},
  f"cascade_psvr_{id}": {"attrs": [{"key_type": 1023 + id, "attr": ["cascade_psvr"], "converter": "discrete", "converter_args": "0.2,0,1,200,-1",}]},
  f"cascade_pltr_{id}": {"attrs": [{"key_type": 1024 + id, "attr": ["cascade_pltr"], "converter": "discrete", "converter_args": "0.2,0,1,400,-1",}]},
  f"cascade_pwtr_{id}": {"attrs": [{"key_type": 1025 + id, "attr": ["cascade_pwtr"], "converter": "discrete", "converter_args": "0.2,0,1,600,-1",}]},
  f"cascade_pftr_{id}": {"attrs": [{"key_type": 1026 + id, "attr": ["cascade_pftr"], "converter": "discrete", "converter_args": "0.2,0,1,800,-1",}]},
  f"cascade_phtr_{id}": {"attrs": [{"key_type": 1027 + id, "attr": ["cascade_phtr"], "converter": "discrete", "converter_args": "0.2,0,1,1000,-1",}]},
  f"cascade_pepstr_{id}": {"attrs": [{"key_type": 1028 + id, "attr": ["cascade_pepstr"], "converter": "discrete", "converter_args": "0.2,0,1,1000,-1",}]},
  f"cascade_pcestr_{id}": {"attrs": [{"key_type": 1029 + id, "attr": ["cascade_pcestr"], "converter": "discrete", "converter_args": "0.2,0,1,600,-1",}]},
}




SEQ_SIZE = 50
COMMON_KUIBA_CONFIG = {
  
  # "click_photo_ids": {"attrs": [{"mio_slot_key_type": 901, "key_type": 901, "attr": ["click_photo_ids"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  # "click_hetu_one": {"attrs": [{"mio_slot_key_type": 902, "key_type": 902, "attr": ["click_hetu_one"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  # "click_hetu_two": {"attrs": [{"mio_slot_key_type": 903, "key_type": 903, "attr": ["click_hetu_two"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "like_photo_ids": {"attrs": [{"key_type": 26, "mio_slot_key_type": 904, "attr": ["like_photo_ids"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "like_hetu_one": {"attrs": [{"key_type": 682, "mio_slot_key_type": 905,  "attr": ["like_hetu_one"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "like_hetu_two": {"attrs": [{"key_type": 683, "mio_slot_key_type": 906,  "attr": ["like_hetu_two"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "follow_photo_ids": {"attrs": [{"key_type": 26, "mio_slot_key_type": 907,  "attr": ["follow_photo_ids"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "follow_hetu_one": {"attrs": [{"key_type": 682, "mio_slot_key_type": 908,  "attr": ["follow_hetu_one"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "follow_hetu_two": {"attrs": [{"key_type": 683, "mio_slot_key_type": 909,  "attr": ["follow_hetu_two"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "forward_photo_ids": {"attrs": [{"key_type": 26, "mio_slot_key_type": 910,  "attr": ["forward_photo_ids"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "forward_hetu_one": {"attrs": [{"key_type": 682, "mio_slot_key_type": 911,  "attr": ["forward_hetu_one"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "forward_hetu_two": {"attrs": [{"key_type": 683,"mio_slot_key_type": 912,  "attr": ["forward_hetu_two"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "comment_photo_ids": {"attrs": [{"key_type": 26, "mio_slot_key_type": 913,  "attr": ["comment_photo_ids"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "comment_hetu_one": {"attrs": [{"key_type": 682, "mio_slot_key_type": 914,  "attr": ["comment_hetu_one"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "comment_hetu_two": {"attrs": [{"key_type": 683,"mio_slot_key_type": 915,  "attr": ["comment_hetu_two"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "collect_photo_ids": {"attrs": [{"key_type": 26, "mio_slot_key_type": 916,  "attr": ["collect_photo_ids"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "collect_hetu_one": {"attrs": [{"key_type": 682, "mio_slot_key_type": 917,  "attr": ["collect_hetu_one"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "collect_hetu_two": {"attrs": [{"key_type": 683,"mio_slot_key_type": 918,  "attr": ["collect_hetu_two"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "profile_enter_photo_ids": {"attrs": [{"key_type": 26, "mio_slot_key_type": 919,  "attr": ["profile_enter_photo_ids"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "profile_enter_hetu_one": {"attrs": [{"key_type": 682, "mio_slot_key_type": 920,  "attr": ["profile_enter_hetu_one"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "profile_enter_hetu_two": {"attrs": [{"key_type": 683,"mio_slot_key_type": 921,  "attr": ["profile_enter_hetu_two"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  
  "user_action_comment_ids": {"attrs": [{"key_type": 26, "mio_slot_key_type": 922,  "attr": ["user_action_comment_ids"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "photo_top_comment_ids": {"attrs": [{"key_type": 26, "mio_slot_key_type": 923,  "attr": ["photo_top_comment_ids"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},

  "human_action": {"attrs": [{"key_type": 711, "attr": ["human_action"], "converter": "id"}]},
  "screen_light": {"attrs": [{"key_type": 712, "attr": ["screen_light"], "converter": "id"}]},
  "net_state": {"attrs": [{"key_type": 713, "attr": ["net_state"], "converter": "id"}]},
  "battery_level": {"attrs": [{"key_type": 714, "attr": ["battery_level"], "converter": "id"}]},
  "battery_charging": {"attrs": [{"key_type": 715, "attr": ["battery_charging"], "converter": "id"}]},
  "headset_state": {"attrs": [{"key_type": 716, "attr": ["headset_state"], "converter": "id"}]},
}

# denominator, smooth, max_val, buckets, min_val
# std::max(std::min(numerator / (denominator + smooth), max_val), min_value) * buckets
ITEM_KUIBA_CONFIG = {
  **id_config("follow_status", 27),
  **pxtr_config(0),
  # "pctr": {"attrs": [{"key_type": 1001, "attr": ["pctr"], "converter": "discrete", "converter_args": "1,0,1,200,-1",}]},
  # "pltr": {"attrs": [{"key_type": 1002, "attr": ["pltr"], "converter": "discrete", "converter_args": "0.2,0,1,400,-1",}]},
  # "pcltr": {"attrs": [{"key_type": 1003, "attr": ["pcltr"], "converter": "discrete", "converter_args": "0.2,0,1,600,-1",}]},
  # "pftr": {"attrs": [{"key_type": 1004, "attr": ["pftr"], "converter": "discrete", "converter_args": "0.2,0,1,800,-1",}]},
  # "pwtr": {"attrs": [{"key_type": 1005, "attr": ["pwtr"], "converter": "discrete", "converter_args": "0.2,0,1,600,-1",}]},
  # "plvtr": {"attrs": [{"key_type": 1006, "attr": ["plvtr"], "converter": "discrete", "converter_args": "1,0,1,200,-1",}]},
  # "pvtr": {"attrs": [{"key_type": 1007, "attr": ["pvtr"], "converter": "discrete", "converter_args": "1,0,1,200,-1",}]},
  # "pptr": {"attrs": [{"key_type": 1008, "attr": ["pptr"], "converter": "discrete", "converter_args": "0.2,0,1,600,-1",}]},
  # "pcmtr": {"attrs": [{"key_type": 1009, "attr": ["pcmtr"], "converter": "discrete", "converter_args": "0.2,0,1,600,-1",}]},
  # "phtr": {"attrs": [{"key_type": 1010, "attr": ["phtr"], "converter": "discrete", "converter_args": "0.2,0,1,1000,-1",}]},
  # "pepstr": {"attrs": [{"key_type": 1011, "attr": ["pepstr"], "converter": "discrete", "converter_args": "0.2,0,1,800,-1",}]},
  # "pcmef": {"attrs": [{"key_type": 1012, "attr": ["pcmef"], "converter": "discrete", "converter_args": "0.2,0,1,400,-1",}]},
  # "pwtd": {"attrs": [{"key_type": 1013, "attr": ["pwtd"], "converter": "discrete", "converter_args": "0.5,0,100,1,0",}]},

  # "empirical_ctr": {"attrs": [{"key_type": 1014, "attr": ["empirical_ctr"], "converter": "discrete", "converter_args": "1,0,1,200,-1",}]},
  # "empirical_ltr": {"attrs": [{"key_type": 1015, "attr": ["empirical_ltr"], "converter": "discrete", "converter_args": "0.2,0,1,400,-1",}]},
  # "empirical_ftr": {"attrs": [{"key_type": 1016, "attr": ["empirical_ftr"], "converter": "discrete", "converter_args": "0.2,0,1,800,-1",}]},
  # "empirical_wtr": {"attrs": [{"key_type": 1017, "attr": ["empirical_wtr"], "converter": "discrete", "converter_args": "0.2,0,1,600,-1",}]},
  # "empirical_ptr": {"attrs": [{"key_type": 1018, "attr": ["empirical_ptr"], "converter": "discrete", "converter_args": "0.2,0,1,400,-1",}]},
  # "empirical_htr": {"attrs": [{"key_type": 1019, "attr": ["empirical_htr"], "converter": "discrete", "converter_args": "0.2,0,1,1000,-1",}]},
  # "empirical_cmtr": {"attrs": [{"key_type": 1020, "attr": ["empirical_cmtr"], "converter": "discrete", "converter_args": "0.2,0,1,600,-1",}]},

  # "cascade_pctr": {"attrs": [{"key_type": 1021, "attr": ["cascade_pctr"], "converter": "discrete", "converter_args": "1,0,1,200,-1",}]},
  # "cascade_plvtr": {"attrs": [{"key_type": 1022, "attr": ["cascade_plvtr"], "converter": "discrete", "converter_args": "1,0,1,200,-1",}]},
  # "cascade_psvr": {"attrs": [{"key_type": 1023, "attr": ["cascade_psvr"], "converter": "discrete", "converter_args": "0.2,0,1,200,-1",}]},
  # "cascade_pltr": {"attrs": [{"key_type": 1024, "attr": ["cascade_pltr"], "converter": "discrete", "converter_args": "0.2,0,1,400,-1",}]},
  # "cascade_pwtr": {"attrs": [{"key_type": 1025, "attr": ["cascade_pwtr"], "converter": "discrete", "converter_args": "0.2,0,1,600,-1",}]},
  # "cascade_pftr": {"attrs": [{"key_type": 1026, "attr": ["cascade_pftr"], "converter": "discrete", "converter_args": "0.2,0,1,800,-1",}]},
  # "cascade_phtr": {"attrs": [{"key_type": 1027, "attr": ["cascade_phtr"], "converter": "discrete", "converter_args": "0.2,0,1,1000,-1",}]},
  # "cascade_pepstr": {"attrs": [{"key_type": 1028, "attr": ["cascade_pepstr"], "converter": "discrete", "converter_args": "0.2,0,1,1000,-1",}]},
  # "cascade_pcestr": {"attrs": [{"key_type": 1029, "attr": ["cascade_pcestr"], "converter": "discrete", "converter_args": "0.2,0,1,600,-1",}]},
}