base_run_id="exp020-rag"
# infer_path="datasets/mediqa-eval-2026-valid-aligned.csv"

folder_path=exp/few_shot/runs/$base_run_id
mkdir $folder_path

infer_path="datasets/mediqa-eval-2026-valid-aligned.csv"
medgemma27b="/workspace/models/medgemma-27b-it"
llm_input_json="datasets/output_rag.json"
output_path="$folder_path/medgemma.json"

python model_runners/inference.py \
    --model_path "$medgemma27b" \
    --data_path "$llm_input_json" \
    --file_name $output_path \
    --run_id "$base_run_id"