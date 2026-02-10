run_folder=""
train_knn_metrics="['completeness','factual-accuracy','relevance','writing-style','overall']"
pred_metrics="['writing-style']"
metrics="['disagree_flag','completeness','factual-accuracy','relevance','writing-style','overall']"

python scripts/eval/eval_main.py \
        --input_pred pred \
        --output_folder "$run_folder" \
        --metrics "$pred_metrics" \
        # --impute_knn True \
        # --knn_train_path datasets/mediqa-eval-2026-valid-aligned.csv \
        # --knn_train_metrics "$metrics" \
        # --knn_pre_metrics $pred_metrics \
        # --model knn
