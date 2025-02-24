set -e
dir=`pwd`
cd conf/py
#cp $dir/conf/dnn-plugin.yaml .
python3 mio_graph.py train --with_kai #生成计算图
cp -r training/conf/* $dir/conf

python3 pipeline.py #生成 dragonfly pipeline
cp *.json $dir/conf
