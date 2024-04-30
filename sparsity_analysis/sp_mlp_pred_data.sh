#!/bin/bash

MODEL_PATH="/shared/vsathia2/hf_models/$1"
DATA_SAVE_PATH="/shared/vsathia2/sp_mlp_predictor/$1"
mkdir -p ${DATA_SAVE_PATH}

# ls ${MODEL_PATH}
echo "Model path is ${MODEL_PATH}"
echo "Inference data will be saved at ${DATA_SAVE_PATH}"
seq 100 | awk '{printf "="}'
echo ""
# Collecting training data for model specified at MODEL_PATH
# Data is collected through inference on wikitext
#
ARGS="--model-path ${MODEL_PATH} --data-path ${DATA_SAVE_PATH}"

python collect_data.py $(echo ${ARGS}) --device cuda:0

seq 100 | awk '{printf "="}'
echo ""
echo "Data collection done!"

