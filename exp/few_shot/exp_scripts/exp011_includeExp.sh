base_run_id="exp-includeExp"
infer_path="exp/few_shot/datasets/val.csv"
medgemma27b="/data1/xinzhe/microsoft_nlp/workspace/mediqa-competition/models/medgemma1.5-27b"
folder_path=exp/few_shot/runs/$base_run_id

mkdir $folder_path

llm_input_json=$folder_path/shot_disagreeExp.json
output_path="$folder_path/medgemma.json"
metrics="['disagree_flag']"

# echo "Running MedGemma-27b inference..."
# python model_runners/inference.py \
#     --model_path "$medgemma27b" \
#     --data_path "$llm_input_json" \
#     --file_name $output_path \
#     --run_id "$base_run_id"

echo "start evaluating..."
python scripts/eval/eval_main.py \
    --input_pred medgemma \
    --output_folder "$folder_path" \
    --metrics "$metrics"

