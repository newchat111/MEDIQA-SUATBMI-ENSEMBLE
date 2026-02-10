#!/bin/bash

# Base configuration
base_run_id="exp-all-metrics"
infer_path="exp/few_shot/datasets/val.csv"
medgemma27b="/data1/xinzhe/microsoft_nlp/workspace/mediqa-competition/models/medgemma1.5-27b"
folder_path=exp/few_shot/runs/$base_run_id

mkdir $folder_path

llm_input_json=$folder_path/input.json
output_path="$folder_path/medgemma.json"
metrics="['disagree_flag','completeness','factual-accuracy','relevance','writing-style','overall']"

python exp/few_shot/scripts/make_shot_main.py \
        --split_from exp/few_shot/datasets/train.csv \
        --infer_path "$infer_path" \
        --output_path "$llm_input_json" \
        --shot_num 7 \
        --metrics "$metrics" \

echo "Running MedGemma-27b inference..."
python model_runners/inference.py \
    --model_path "$medgemma27b" \
    --data_path "$llm_input_json" \
    --file_name $output_path \
    --run_id "$base_run_id"

echo "start evaluating..."
python scripts/eval/eval_main.py \
    --input_pred medgemma \
    --output_folder "$folder_path" \
    --metrics "$metrics"