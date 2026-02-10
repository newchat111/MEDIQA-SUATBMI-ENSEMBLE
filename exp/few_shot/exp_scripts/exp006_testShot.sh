#!/bin/bash

# Base configuration
base_run_id="exp-testshot"
infer_path="exp/few_shot/datasets/val.csv"
medgemma27b="/data1/xinzhe/microsoft_nlp/workspace/mediqa-competition/models/medgemma1.5-27b"

# Loop through shots 1 to 14
for i in {0..14}
do
    # 1. Setup specific paths for this iteration
    folder_path="exp/few_shot/runs/$base_run_id/shot${i}"
    shot_path="exp/few_shot/datasets/shots/shot${i}.csv"
    llm_input_json="$folder_path/input.json"

    # 2. Create the run directory
    mkdir -p "$folder_path"

    # 3. Prepare input (combines shots and inference data into prompt format)
    python exp/few_shot/scripts/make_shot_main.py \
        --shot_path "$shot_path" \
        --infer_path "$infer_path" \
        --output_path "$llm_input_json"

    # 4. Run medgemma27b inference
    echo "Running MedGemma-27b inference..."
    python model_runners/inference.py \
        --model_path "$medgemma27b" \
        --data_path "$llm_input_json" \
        --file_name "$folder_path/medgemma.json" \
        --run_id "$base_run_id"

    # 5. Evaluate MedGemma results
    echo "Starting evaluation..."
    python scripts/eval/eval_main.py \
        --input_pred medgemma \
        --output_folder "$folder_path"

done