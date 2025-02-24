cd ../py
python3 mio_graph.py predict #生成计算图
cp -r predict/config/* ../fr/

cd ../fr/
python3 infer.py
