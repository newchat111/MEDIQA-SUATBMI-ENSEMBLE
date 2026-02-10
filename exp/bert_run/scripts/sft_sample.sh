#!/bin/bash

BERTS="MEDBERT"
project_name="BERT"
entity_name="charles2605722943-uc-santa-barbara"
metrics="writing-style"
data_path="datasets/mediqa-eval-2026-valid-folded.csv"
folds=(4)
output_folder="exp/bert_run/runs/run001"
epoch=10

# create the base output folder
mkdir -p $output_folder

for bert in $BERTS; do
    # per model loop
    wandb_group=$bert
    for m in $metrics; do
        # per metric loop
        for f in "${folds[@]}"; do
            echo "model:$bert, metric:$m, fold:$f"

            # create run folder
            run_folder=$output_folder/$bert/$m/fold$f
            mkdir -p $run_folder

            # define run_id
            run_id=$bert-$m-$f

            # call python finetuner
            python exp/bert_run/scripts/finetuner.py \
                --model_name $bert \
                --data_path $data_path \
                --fold $f \
                --metric $m \
                --output_dir $run_folder \
                --num_train_epochs $epoch \
                --run_id $run_id \
                --wandb_entity $entity_name \
                --wandb_project $project_name \
                --wandb_group $wandb_group \
                --per_device_train_batch_size 4 \
                --per_device_eval_batch_size 32 \
                --metric_for_best_model eval_loss \
                --learning_rate 2e-6 \
                --lr_scheduler_type cosine \
                --warmup_ratio 0.05
        done
    done
done
