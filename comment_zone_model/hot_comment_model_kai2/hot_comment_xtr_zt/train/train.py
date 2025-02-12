import os
import sys
import logging
import traceback
import psutil
import kai.tensorflow as kai
from kai.tensorflow.utils import log_helper
from kai.advanced_api.process_utils import get_routine_load_model_path
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
logger = log_helper.get_logger(
    __name__, logging.INFO)

def runtime_main():
    """
    第一步: 导入kai，并初始化kai，从环境变量中拿到分布式各个节点的角色信息
    """
    kai.init()

    """
    第二步: 引入用户定义配置
    """
    kai_config = kai.Config()
    kai_config.deserialize("./kai_v2_config.yaml")

    logger.info("kai-2.0 配置加载成功")

    """
    第三步: 引入用户定义模型结构
    """
    import model

    """
    第四步: 编译，配置检查，并生成dnn_plugin/reader.json等文件
    第五步: 初始化运行期各个模块
    """
    kai.start(kai_config=kai_config,
              init_metrics=False)
    
    kai.load()
    train()

    kai.shutdown()


def train():
    kai.switch_dataset("train")
    kai.set_run_mode(kai.RunMode.TRAIN)

    logger.info("模型训练开始")
    while not kai.data_exhausted():
        kai.run_one_pass_train()
        time_decay()
        model_recycle()
        model_save()
        send_btq()

def model_recycle():
    """
    通用的recycle / shrink 逻辑

    对齐C++中控逻辑，在以下情况时，会触发recycle进行feature的退场
    优先级顺序由上至下，满足条件时同一pass内只触发一次recycle

    1、训练进入了新的一天，会在 00:00 时进行recycle
    2、距离上一次 shrink 过去了 check_shrink_interval_pass，并且feature_num 大于了 max_feature_num

    测试模式不会recycle
    """
    recycle_pass_context = kai.Collector().get_collection(
        kai.GraphKeys.PASS_CONTEXT, "recycle")
    match_condition, condition_dict = recycle_pass_context.meet_condition()

    if match_condition:
        logger.info("model_recycle conditon: {}".format(condition_dict))
        kai.recycle()
    return


def time_decay():
    time_decay_pass_context = kai.Collector().get_collection(
        kai.GraphKeys.PASS_CONTEXT, "time_decay")
    match_condition, condition_dict = time_decay_pass_context.meet_condition()

    if match_condition:
        logger.info("time decay conditon: {}".format(condition_dict))
        kai.time_decay()

    return


def model_save():
    """
    通用的Save model方法

    对齐C++中控的模型保存逻辑,在以下四种情况时，会进行模型的保存
    优先顺序由上至下，即使满足多个条件，一个pass内，仅会保存一次模型

    1、训练进入了新的一天，会在 00:00 时进行模型的保存
    2、距离上一次模型保存过去了 save_model_interval_pass
    3、距离上一次模型保存过去了 save_model_interval_hour
    4、当前训练时间首次位于save_model_hours 中所配置的选项

    测试模式不会保存模型
    """
    save_pass_context = kai.Collector().get_collection(
        kai.GraphKeys.PASS_CONTEXT, "save")
    match_condition, condition_dict = save_pass_context.meet_condition()

    if match_condition:
        logger.info("model_save conditon: {}".format(condition_dict))

        if condition_dict["match_new_day"]:
            kai.change_passwise_option(save_option_path_prefix="model")
        else:
            kai.change_passwise_option(save_option_path_prefix="checkpoint")

        kai.save()

    return


def send_btq():
    """
    通用的BTQ send model方法

    对齐C++中控的模型Send逻辑,在以下五种情况时，会进行模型的send
    优先顺序由上至下，即使满足多个条件，一个pass内，仅会send一次模型

    1、base：训练进入了新的一天，会在 00:00 时进行模型的send
    2、base：当前训练时间首次位于send_base_hours 中所配置的时间
    3、base：距离上一次模型的send过去了至少 send_base_interval_hour
    4、delta：距离上一次send delta 过去了 send_delta_interval_time_ms
    5、delta：距离上一次send delta 过去了 send_delta_interval_pass

    测试模式不会Send模型
    """
    btq_base_pass_context = kai.Collector().get_collection(
        kai.GraphKeys.PASS_CONTEXT, "btq_base")
    match_condition, condition_dict = btq_base_pass_context.meet_condition()
    match_condition = (
        match_condition and kai.Config().btq_save_option.use_btq_save)

    if match_condition:
        logger.info("btq_save_base condition: {}".format(condition_dict))
        kai.change_passwise_option(btq_save_option_mode="base")
        kai.btq_save()
        return

    btq_delta_pass_context = kai.Collector().get_collection(
        kai.GraphKeys.PASS_CONTEXT, "btq_delta")
    match_condition, condition_dict = btq_delta_pass_context.meet_condition()
    match_condition = (
        match_condition and kai.Config().btq_save_option.use_btq_save)
    if match_condition:
        logger.info("btq_save_delta condition: {}".format(condition_dict))
        kai.change_passwise_option(btq_save_option_mode="delta")
        kai.btq_save()
    return


if __name__ == '__main__':
    try:
        runtime_main()
    except Exception as err:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        psutil.Process(os.getpid()).kill()
