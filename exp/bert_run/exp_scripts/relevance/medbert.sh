#!/bin/bash

# Fixed Hyperparameters
BERT="MEDBERT"
LR="2e-5"
TRAIN_BS=32
EPOCH=20
SCHED="cosine"
WARMUP=0.1
EVAL_BATCH_SIZE=32
include_query=True
PROJECT_NAME="MEDBERT-REL-HIGHBATCH-V2"
ENTITY_NAME="charles2605722943-uc-santa-barbara"
METRICS="relevance"
DATA_PATH="datasets/mediqa-eval-2026-valid-folded.csv"
FOLDS=(4 3 2 1 0)
OUTPUT_BASE="exp/bert_run/runs/MEDBERT-RELEVANCE-HIGHBATCH-V2"
WANDB_GROUP="lr${LR}_tbs${TRAIN_BS}_ep${EPOCH}_sch${SCHED}_warm${WARMUP}"

for FOLD in "${FOLDS[@]}"; do
    echo "Starting Fold $FOLD"
    
    # Nested loop to run 12 times per fold
    for RUN_IDX in {1..12}; do
        
        RUN_ID="${WANDB_GROUP}_fold${FOLD}_run${RUN_IDX}"
        RUN_FOLDER="${OUTPUT_BASE}/${RUN_ID}"
        mkdir -p "$RUN_FOLDER"

        echo "  → Run $RUN_IDX / 12 for Fold $FOLD"

        python exp/bert_run/scripts/finetuner.py \
            --model_name $BERT \
            --data_path $DATA_PATH \
            --fold $FOLD \
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
            --include_query $include_query
    done
done
