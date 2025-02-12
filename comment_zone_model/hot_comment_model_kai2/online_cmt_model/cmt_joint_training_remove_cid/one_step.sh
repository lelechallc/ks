set -e
pwd

python ./train/model.py --mode predict

ls -la ./predict/config

python ./infer/infer_server/infer_cpu_old_emb.py