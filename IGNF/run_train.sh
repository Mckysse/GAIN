#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

REPO=$PWD

LAN=${1:-"en"}
ENCODER_MODEL=${2:-"xlm-roberta-large"}
DATA_DIR=${3:-"$REPO/data/"}
OUT_DIR=${4:-"$REPO/output/"}

base_dir=${REPO}/../
train_file=${DATA_DIR}/${LAN}_train.conll
dev_file=${DATA_DIR}/${LAN}_dev.conll
gazetteer_path=${base_dir}/gazetteers/gazetteer_demo/${LAN}
model_name="ner_${LAN}"

python -m train_model --train "$train_file" --dev "$dev_file" --gazetteer "$gazetteer_path" --out_dir "$OUT_DIR" \
                      --model_name "$model_name" --gpus 1 --epochs 20 --encoder_model "$ENCODER_MODEL" --batch_size 32 \
                      --lr 0.00002
