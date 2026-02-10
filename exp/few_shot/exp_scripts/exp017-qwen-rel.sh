base_run_id="exp017-rel-qwen-all"
# infer_path="datasets/mediqa-eval-2026-valid-aligned.csv"
infer_path="datasets/mediqa-eval-2026-valid-aligned.csv"
qwen="/data1/xinzhe/microsoft_nlp/mediqa-competition/models/QWEN3-30B-A3B"
folder_path=exp/few_shot/runs/$base_run_id

mkdir $folder_path

llm_input_json=$folder_path/input.json
output_path="$folder_path/qwen30b.json"

metrics="['relevance']"

for m in $metrics; do
    echo "preparing inputs"
    python exp/few_shot/scripts/make_shot_main.py \
            --split_from exp/few_shot/datasets/train.csv \
            --infer_path "$infer_path" \
            --output_path "$llm_input_json" \
            --shot_num 3 \
            --metrics "$m" \
            --exclude_image True

    echo "Running QWEN30B inference..."
    python model_runners/infer_qwen3.py \
        --model_path "$qwen" \
        --data_path "$llm_input_json" \
        --file_name $output_path \
        --run_id "$base_run_id"

    echo "start evaluating..."
    python scripts/eval/eval_main.py \
        --input_pred qwen30b \
        --output_folder "$folder_path" \
        --metrics "$m" \
        # --impute_knn True \
        # --knn_train_path datasets/mediqa-eval-2026-valid-aligned.csv \
        # --knn_train_metrics "$m"
done