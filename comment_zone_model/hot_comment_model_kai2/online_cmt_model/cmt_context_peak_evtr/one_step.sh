set -e
pwd

python ./train/model_add_page_rebase_hate.py --mode predict

ls -la ./predict/config

python ./infer/infer_server/infer_gpu.py