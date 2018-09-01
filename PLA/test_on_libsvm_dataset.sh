#!/bin/bash

wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/breast-cancer_scale

dataset_size=$(wc breast-cancer_scale | awk '{print $1}')
test_size=$((dataset_size/3))

sort -R breast-cancer_scale > breast-cancer_scale.random
rm -f breast-cancer_scale
tail -n $test_size breast-cancer_scale.random > breast-cancer_scale.t
head -n $((dataset_size-test_size)) breast-cancer_scale.random > breast-cancer_scale
rm -f breast-cancer_scale.random

python PLA_libsvm_dataset.py breast-cancer_scale breast-cancer_scale.t 800

rm -f breast-cancer_scale breast-cancer_scale.t

