#!/bin/bash
#
DATA_PATH="/shared/vsathia2/sp_mlp_predictor/$1"

for l in $(seq 0 31)
do  
    (trap 'kill 0' SIGINT; \
    CUDA_VISIBLE_DEVICES=0 python main_mlp.py --dataset wikitext --lr 0.001 --L ${l} > ${DATA_PATH}/wiki_mlp_out_${l}.txt & \
    wait)
done
