import tensorflow as tf
import modeling
import tokenization_zh
import numpy as np
from tqdm import tqdm
import json

def convert_text_to_input(texts, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    input_ids_batch = []
    attention_masks_batch = []
    sep_ids_batch = []
    for (ex_index, text) in enumerate(texts):
        raw_tokens = tokenizer.tokenize(text)

        if len(raw_tokens) > seq_length - 2:
            raw_tokens = raw_tokens[0:(seq_length - 2)]

        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in raw_tokens:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        input_ids_batch.append(input_ids)
        attention_masks_batch.append(input_mask)
        sep_ids_batch.append(input_type_ids)
    return input_ids_batch, attention_masks_batch, sep_ids_batch

with open('./优质4分档标注.txt') as f:
    raw_data = f.readlines()

bert_config_file = "./Robert256/bert_config.json"
bert_checkpoint_file ="./tensorflow_model/bert_model.ckpt"
vocab_file = "./Robert256/vocab.txt"

max_seq_length = 64  # 输入的最大长度
batch_size = 1  # 批量大小

bert_config = modeling.BertConfig.from_json_file(bert_config_file)

input_ids = tf.placeholder(tf.int32, shape=[None, max_seq_length], name="input_ids")
input_mask = tf.placeholder(tf.int32, shape=[None, max_seq_length], name="input_mask")
segment_ids = tf.placeholder(tf.int32, shape=[None, max_seq_length], name="segment_ids")

model = modeling.BertModel(
    config=bert_config,
    is_training=False,
    input_ids=input_ids,
    input_mask=input_mask,
    token_type_ids=segment_ids,
    use_one_hot_embeddings=False
)
tokenizer = tokenization_zh.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=False)

hidden_states = model.get_sequence_output()

init = tf.global_variables_initializer()

tvars = tf.trainable_variables()

reader = tf.train.NewCheckpointReader(bert_checkpoint_file)
ckpt_vars = reader.get_variable_to_shape_map()

available_vars = []
for var in tvars:
    var_name = var.name.split(':')[0]
    if var_name in ckpt_vars:
        available_vars.append(var)
    else:
        print("变量在检查点中未找到，将使用初始化值：", var.name)

output_file = open("tf_bert_verify3.txt", "w")
saver = tf.train.Saver(var_list=available_vars)

with tf.Session() as sess:
    # sess.run(init)
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, bert_checkpoint_file)

    for line in tqdm(raw_data):
        pid, cid, comment_content = line.strip().split('\t')
        convert_input_ids, attention_masks, sep_ids = convert_text_to_input([comment_content], 64, tokenizer)
        input_data = {
            input_ids: convert_input_ids,  # 样例输入
            input_mask: attention_masks,
            segment_ids: sep_ids
        }
        final_hidden_states = sess.run(hidden_states, feed_dict=input_data)
        cls_emb = np.squeeze(final_hidden_states[0, 0, :]).tolist()
        # print(type(cls_emb))
        output_file.write('\t'.join([pid, cid, comment_content, json.dumps(cls_emb)]) + "\n")
output_file.close()