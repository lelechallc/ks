kuiba_list_converter_config_list_limit = lambda limit_n:  {
  "converter": "list",
  "type":5,
  "converter_args": {
    "reversed": False,
    "enable_filter": False,
    "limit": limit_n,
  },
 }
kuiba_id_converter = {
  "converter": "id"
}

id_config = lambda attr_name, slot: {
  attr_name: {"attrs": [{"mio_slot_key_type": slot, "key_type": slot, "attr": [attr_name], **kuiba_id_converter}]},
}

SEQ_SIZE = 50
COMMON_KUIBA_CONFIG = {
  
  "click_photo_ids": {"attrs": [{"key_type": 901, "attr": ["click_photo_ids"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "click_hetu_one": {"attrs": [{"key_type": 902, "attr": ["click_hetu_one"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "click_hetu_two": {"attrs": [{"key_type": 903, "attr": ["click_hetu_two"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  
  "like_photo_ids": {"attrs": [{"key_type": 904, "attr": ["like_photo_ids"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "like_hetu_one": {"attrs": [{ "key_type": 905, "attr": ["like_hetu_one"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "like_hetu_two": {"attrs": [{"key_type": 906, "attr": ["like_hetu_two"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  
  "follow_photo_ids": {"attrs": [{"key_type": 907, "attr": ["follow_photo_ids"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "follow_hetu_one": {"attrs": [{"key_type": 908, "attr": ["follow_hetu_one"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "follow_hetu_two": {"attrs": [{"key_type": 909, "attr": ["follow_hetu_two"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  
  "forward_photo_ids": {"attrs": [{"key_type": 910, "attr": ["forward_photo_ids"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "forward_hetu_one": {"attrs": [{"key_type": 911, "attr": ["forward_hetu_one"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "forward_hetu_two": {"attrs": [{"key_type": 912, "attr": ["forward_hetu_two"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},

  "comment_photo_ids": {"attrs": [{"key_type": 913, "attr": ["comment_photo_ids"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "comment_hetu_one": {"attrs": [{"key_type": 914, "attr": ["comment_hetu_one"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "comment_hetu_two": {"attrs": [{"key_type": 915, "attr": ["comment_hetu_two"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},

  "collect_photo_ids": {"attrs": [{"key_type": 916, "attr": ["collect_photo_ids"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "collect_hetu_one": {"attrs": [{"key_type": 917, "attr": ["collect_hetu_one"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "collect_hetu_two": {"attrs": [{"key_type": 918, "attr": ["collect_hetu_two"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},

  "download_photo_ids": {"attrs": [{"key_type": 919, "attr": ["download_photo_ids"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "download_hetu_one": {"attrs": [{"key_type": 920, "attr": ["download_hetu_one"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  "download_hetu_two": {"attrs": [{"key_type": 921, "attr": ["download_hetu_two"], **kuiba_list_converter_config_list_limit(SEQ_SIZE)}]},
  
  # "human_action": {"attrs": [{"key_type": 701, "attr": ["human_action"], "converter": "id"}]},
  # "screen_light": {"attrs": [{"key_type": 702, "attr": ["screen_light"], "converter": "id"}]},
  # "net_state": {"attrs": [{"key_type": 703, "attr": ["net_state"], "converter": "id"}]},
  # "battery_level": {"attrs": [{"key_type": 704, "attr": ["battery_level"], "converter": "id"}]},
  # "battery_charging": {"attrs": [{"key_type": 705, "attr": ["battery_charging"], "converter": "id"}]},
  # "headset_state": {"attrs": [{"key_type": 706, "attr": ["headset_state"], "converter": "id"}]},
  # "req_time_hour": {"attrs": [{"key_type": 707, "attr": ["req_time_hour"], "converter": "id"}]},

   "u_mean_pltr": {"attrs": [{"key_type": 950, "attr": ["u_mean_pltr"], "converter": "discrete", "converter_args": "0.2,0,1,400,-1",}]},
  "u_std_pltr": {"attrs": [{"key_type": 951, "attr": ["u_std_pltr"], "converter": "discrete", "converter_args": "0.2,0,1,400, 0",}]},
  "u_mean_pwtr": {"attrs": [{"key_type": 952, "attr": ["u_mean_pwtr"], "converter": "discrete", "converter_args": "0.2,0,1,600,-1",}]},
  "u_std_pwtr": {"attrs": [{"key_type": 953, "attr": ["u_std_pwtr"], "converter": "discrete", "converter_args": "0.2,0,1,600, 0",}]},
  "u_mean_pftr": {"attrs": [{"key_type": 954, "attr": ["u_mean_pftr"], "converter": "discrete", "converter_args": "0.2,0,1,800,-1",}]},
  "u_std_pftr": {"attrs": [{"key_type": 955, "attr": ["u_std_pftr"], "converter": "discrete", "converter_args": "0.2,0,1,800, 0",}]},
  "u_mean_pcmtr": {"attrs": [{"key_type": 956, "attr": ["u_mean_pcmtr"], "converter": "discrete", "converter_args": "0.2,0,1,600,-1",}]},
  "u_std_pcmtr": {"attrs": [{"key_type": 957, "attr": ["u_std_pcmtr"], "converter": "discrete", "converter_args": "0.2,0,1,600, 0",}]},
  "u_mean_pcltr": {"attrs": [{"key_type": 958, "attr": ["u_mean_pcltr"], "converter": "discrete", "converter_args": "0.2,0,1,600,-1",}]},
  "u_std_pcltr": {"attrs": [{"key_type": 959, "attr": ["u_std_pcltr"], "converter": "discrete", "converter_args": "0.2,0,1,600, 0",}]},
  "u_mean_pdtr": {"attrs": [{"key_type": 960, "attr": ["u_mean_pdtr"], "converter": "discrete", "converter_args": "0.2,0,1,600,-1",}]},
  "u_std_pdtr": {"attrs": [{"key_type": 961, "attr": ["u_std_pdtr"], "converter": "discrete", "converter_args": "0.2,0,1,600, 0",}]},
}

# denominator, smooth, max_val, buckets, min_val
# std::max(std::min(numerator / (denominator + smooth), max_val), min_value) * buckets
ITEM_KUIBA_CONFIG = {
  # **id_config("follow_status", 27),
  **id_config("photo_hetu_one", 201),
  **id_config("photo_hetu_two", 202),
  **id_config("photo_hetu_one", 10201),
  **id_config("photo_hetu_two", 10202),

  # "pctr": {"attrs": [{"key_type": 1001, "attr": ["pctr"], "converter": "discrete", "converter_args": "1,0,1,200,-1",}]},
  # "pltr": {"attrs": [{"key_type": 1001, "attr": ["pltr"], "converter": "discrete", "converter_args": "0.2,0,1,400,-1",}]},
  # "pwtr": {"attrs": [{"key_type": 1002, "attr": ["pwtr"], "converter": "discrete", "converter_args": "0.2,0,1,600,-1",}]},
  # "pftr": {"attrs": [{"key_type": 1003, "attr": ["pftr"], "converter": "discrete", "converter_args": "0.2,0,1,800,-1",}]},
  # "pcmtr": {"attrs": [{"key_type": 1004, "attr": ["pcmtr"], "converter": "discrete", "converter_args": "0.2,0,1,600,-1",}]},
  # "pcltr": {"attrs": [{"key_type": 1005, "attr": ["pcltr"], "converter": "discrete", "converter_args": "0.2,0,1,600,-1",}]},
  # "pdtr": {"attrs": [{"key_type": 1006, "attr": ["pdtr"], "converter": "discrete", "converter_args": "0.2,0,1,600,-1",}]},
  
  # "plvtr": {"attrs": [{"key_type": 1006, "attr": ["plvtr"], "converter": "discrete", "converter_args": "1,0,1,200,-1",}]},
  # "pvtr": {"attrs": [{"key_type": 1007, "attr": ["pvtr"], "converter": "discrete", "converter_args": "1,0,1,200,-1",}]},
  # "pptr": {"attrs": [{"key_type": 1008, "attr": ["pptr"], "converter": "discrete", "converter_args": "0.2,0,1,600,-1",}]},
  
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