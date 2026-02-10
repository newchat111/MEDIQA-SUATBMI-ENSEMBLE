#!/bin/bash

# ----------------------------
# Experiment variables
# ----------------------------

BERT="MEDBERT"
METRICS="relevance"
BASE_RUN_FOLDER="exp/bert_run/runs/MEDBERT-AUGMENT-REL"
EPOCH=20
BASE_RUN_ID="MEDBERT-AUGMENT-REL"
ENTITY_NAME="charles2605722943-uc-santa-barbara"
PROJECT_NAME="MEDBERT-AUGMENT-RELEVANCE-V2"
WANDB_GROUP="MEDBERT-AUGMENT-RELEVANCE-V2"
TRAIN_BS=32
EVAL_BATCH_SIZE=32
LR=2e-5
SCHED="cosine"
WARMUP=0.1
TRAIN_DATA_PATH="exp/augment/inputs/fold4/train_augmented.csv"
VAL_DATA_PATH="exp/augment/inputs/fold4/val.csv"
include_query=True

# ----------------------------
# Run Loop (12 times)
# ----------------------------

for i in {1..12}; do
    # Creating unique identifiers for each of the 12 iterations
    RUN_ID="${BASE_RUN_ID}_run${i}"
    RUN_FOLDER="${BASE_RUN_FOLDER}/run${i}"

    echo "------------------------------------------------"
    echo "Executing Run $i / 12 for Relevance Metric"
    echo "ID: $RUN_ID"
    echo "------------------------------------------------"

    python exp/bert_run/scripts/finetuner_v2.py \
        --model_name $BERT \
        --train_data_path $TRAIN_DATA_PATH \
        --val_data_path $VAL_DATA_PATH \
        --metric $METRICS \
        --output_dir $RUN_FOLDER \
        --num_train_epochs $EPOCH \
        --run_id $RUN_ID \
        --wandb_entity $ENTITY_NAME \
        --wandb_project $PROJECT_NAME \
        --wandb_group $WANDB_GROUP \
        --per_device_train_batch_size $TRAIN_BS \
        --per_device_eval_batch_size $EVAL_BATCH_SIZE \
        --metric_for_best_model eval_mean_corr \
        --learning_rate $LR \
        --lr_scheduler_type $SCHED \
        --warmup_ratio $WARMUP \
        --include_query True
done