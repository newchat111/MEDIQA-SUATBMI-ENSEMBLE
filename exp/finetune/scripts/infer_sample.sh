base_run_id=run003
folder_path=exp/finetune/runs/run-per-metric/
output_path="$folder_path/medgemma.json"

metrics="['relevance']"

medgemma27b=exp/finetune/runs/run-per-metric/$metrics/checkpoint-125
llm_input_json=exp/finetune/runs/run-per-metric/$metrics/infer.json

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
    --metrics "$metrics" \
    --eval_sft True

