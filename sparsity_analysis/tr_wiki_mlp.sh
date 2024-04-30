#!/bin/bash
#
DATA_PATH="/shared/vsathia2/sp_mlp_predictor/$1"

for l in $(seq 0 31) 
do  
python main_mlp.py --dataset wikitext --lr 0.001 --L ${l} > ${DATA_PATH}/wiki_mlp_out_${l}.txt  
done
