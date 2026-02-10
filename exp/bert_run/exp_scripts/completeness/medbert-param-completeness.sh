#!/bin/bash

BERTS="MEDBERT"
PROJECT_NAME="MEDBERT-HIGHBATCH-COMPLETENESS"
ENTITY_NAME="charles2605722943-uc-santa-barbara"
METRICS="completeness"
include_query=True
DATA_PATH="datasets/mediqa-eval-2026-valid-folded.csv"
FOLDS=(0 1 2 3 4)
OUTPUT_BASE="exp/bert_run/runs/MEDBERT-HIGHBATCH-COMPLETENESS"

# Hyperparameter grid
LEARNING_RATES=(2e-5 2e-6)
TRAIN_BATCH_SIZES=(32 64)
EPOCHS=(50 100)
SCHEDULERS=("cosine")
WARMUP_RATIOS=(0.1 0.2 0.3)

EVAL_BATCH_SIZE=32

for BERT in $BERTS; do
    for LR in "${LEARNING_RATES[@]}"; do
        for TRAIN_BS in "${TRAIN_BATCH_SIZES[@]}"; do
            for EPOCH in "${EPOCHS[@]}"; do
                for SCHED in "${SCHEDULERS[@]}"; do
                    for WARMUP in "${WARMUP_RATIOS[@]}"; do
                        
                        WANDB_GROUP="lr${LR}_tbs${TRAIN_BS}_ep${EPOCH}_sch${SCHED}_warm${WARMUP}"

                        echo "Starting HP group: $WANDB_GROUP"

                        for FOLD in "${FOLDS[@]}"; do
                            
                            RUN_ID="${WANDB_GROUP}_fold${FOLD}"
                            RUN_FOLDER="${OUTPUT_BASE}/${RUN_ID}"
                            mkdir -p "$RUN_FOLDER"

                            echo "  → Fold $FOLD"

                            python exp/bert_run/scripts/finetuner.py \
                                --model_name $BERT \
                                --data_path $DATA_PATH \
                                --fold $FOLD \
                                --include_query $include_query\
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
                                --warmup_ratio $WARMUP
                        done

                    done
                done
            done
        done
    done
done
