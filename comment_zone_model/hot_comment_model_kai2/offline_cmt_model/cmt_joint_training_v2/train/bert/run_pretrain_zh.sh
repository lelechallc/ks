
# step1: create tf record
python create_pretraining_data_zh.py \
    --input_file=./sample_text.txt \
    --output_file=/tmp/tf_examples.tfrecord \
    --vocab_file=Robert256/vocab.txt \
    --do_lower_case=True \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --masked_lm_prob=0.15 \
    --random_seed=12345 \
    --dupe_factor=5


# step2: run prtrain
python run_pretraining.py \
    --input_file=/tmp/tf_examples.tfrecord \
    --output_dir=/tmp/pretraining_output \
    --do_train=True \
    --do_eval=True \
    --bert_config_file=Robert256/bert_config.json \
    --train_batch_size=32 \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --num_train_steps=200 \
    --num_warmup_steps=10 \
    --learning_rate=2e-5

# step3: 检查模型var name
python -m tensorflow.python.tools.inspect_checkpoint --file_name=/hetu_group/dumeng/projects/MLLM/tensorflow_model/bert_model.ckpt

# step4: 抽取特征
python ./extract_output.py

