#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

REPO=$PWD

LAN=${1:-"en"}
ENCODER_MODEL=${2:-"xlm-roberta-large"}
DATA_DIR=${3:-"$REPO/data/"}
OUT_DIR=${4:-"$REPO/output/"}

model_name="ner_${LAN}"
base_dir=${REPO}/../
test_file=${DATA_DIR}/${LAN}_test.conll
gazetteer_path=${base_dir}/gazetteers/gazetteer_demo/${LAN}

model_file_path=CKPT_MODEL_FILE_PATH

python -m do_predict --test "$test_file" --gazetteer "$gazetteer_path" --out_dir "$OUT_DIR" --model_name "$model_name" --gpus 1 \
                                    --encoder_model "$ENCODER_MODEL" --model "$model_file_path" --batch_size 32
