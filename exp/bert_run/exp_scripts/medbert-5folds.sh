#!/bin/bash



BERT="MEDBERT"
PROJECT_NAME="MEDBERT-5folds"
ENTITY_NAME="charles2605722943-uc-santa-barbara"
METRICS="writing-style"
DATA_PATH="datasets/mediqa-eval-2026-valid-folded.csv"
FOLDS=(0 1 2 3 4)
OUTPUT_BASE="exp/bert_run/runs/MEDBERT-5folds"

# Hyperparameter grid
LEARNING_RATE=2e-5
TRAIN_BATCH_SIZE=32
EPOCHS=100
SCHEDULER=cosine
WARMUP_RATIO=0.1

# Fixed evaluation batch size
EVAL_BATCH_SIZE=32

# Optional: W&B group for this HP search
WANDB_GROUP="MEDBERT-5FOLDS"

for FOLD in "${FOLDS[@]}"; do
    # Create a unique run ID and output folder
    RUN_ID="fold${FOLD}"
    RUN_FOLDER="${OUTPUT_BASE}/${RUN_ID}"
    mkdir -p $RUN_FOLDER
    
    echo "Starting run: $RUN_ID"

    python exp/bert_run/scripts/finetuner.py \
        --model_name $BERT \
        --data_path $DATA_PATH \
        --fold $FOLD \
        --metric $METRICS \
        --output_dir $RUN_FOLDER \
        --num_train_epochs $EPOCHS \
        --run_id $RUN_ID \
        --wandb_entity $ENTITY_NAME \
        --wandb_project $PROJECT_NAME \
        --wandb_group $WANDB_GROUP \
        --per_device_train_batch_size $TRAIN_BATCH_SIZE \
        --per_device_eval_batch_size $EVAL_BATCH_SIZE \
        --metric_for_best_model eval_mean_corr \
        --learning_rate $LEARNING_RATE \
        --lr_scheduler_type $SCHEDULER \
        --warmup_ratio $WARMUP_RATIO
done