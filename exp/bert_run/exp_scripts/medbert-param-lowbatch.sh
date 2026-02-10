#!/bin/bash

# ----------------------------
# Hyperparameter search config
# ----------------------------

BERTS="MEDBERT"
PROJECT_NAME="MEDBERT-LOWBATCH"
ENTITY_NAME="charles2605722943-uc-santa-barbara"
METRICS="writing-style"
DATA_PATH="datasets/mediqa-eval-2026-valid-folded.csv"
FOLDS=(4)
OUTPUT_BASE="exp/bert_run/runs/MEDBERT-LOWBATCH"

# Hyperparameter grid
LEARNING_RATES=(2e-5 2e-6 5e-6)
TRAIN_BATCH_SIZES=(4 8 16)
EPOCHS=(15 20)
SCHEDULERS=("linear" "cosine")
WARMUP_RATIOS=(0.0 0.05 0.1)

# Fixed evaluation batch size
EVAL_BATCH_SIZE=32

# Optional: W&B group for this HP search
WANDB_GROUP="MEDBERT-LOWBATCH"

# ----------------------------
# Loop over all combinations
# ----------------------------

for BERT in $BERTS; do
    for FOLD in "${FOLDS[@]}"; do
        for LR in "${LEARNING_RATES[@]}"; do
            for TRAIN_BS in "${TRAIN_BATCH_SIZES[@]}"; do
                for EPOCH in "${EPOCHS[@]}"; do
                    for SCHED in "${SCHEDULERS[@]}"; do
                        for WARMUP in "${WARMUP_RATIOS[@]}"; do
                            
                            # Create a unique run ID and output folder
                            RUN_ID="lr${LR}_tbs${TRAIN_BS}_ep${EPOCH}_sch${SCHED}_warm${WARMUP}_fold${FOLD}"
                            RUN_FOLDER="${OUTPUT_BASE}/${RUN_ID}"
                            mkdir -p $RUN_FOLDER
                            
                            echo "Starting run: $RUN_ID"
                            
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
                                --metric_for_best_model eval_loss \
                                --learning_rate $LR \
                                --lr_scheduler_type $SCHED \
                                --warmup_ratio $WARMUP
                            
                        done
                    done
                done
            done
        done
    done
done
