#!/bin/bash

wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.bz2
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.t.bz2

bzip2 -d ijcnn1.bz2
bzip2 -d ijcnn1.t.bz2

python PLA_libsvm_dataset.py ijcnn1 ijcnn1.t 300

rm -f ijcnn1 ijcnn1.t

