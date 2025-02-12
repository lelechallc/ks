set -e
dir=`pwd`
cd conf/py #

python3 tf_graph.py --with_kai train #生成计算图
cp -r training/conf/* $dir/conf

python3 pipeline.py #生成 dragonfly pipeline
cp *.json $dir/conf
