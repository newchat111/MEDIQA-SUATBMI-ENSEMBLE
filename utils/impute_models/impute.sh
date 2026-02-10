python utils/impute_models/impute.py \
    --input_path exp/few_shot/runs/exp014-all-data/medgemma_submit.csv \
    --output_path exp/few_shot/runs/exp014-all-data/medgemma_imputed.csv \
    --train_data_path exp/few_shot/datasets/train.csv \
    --train_metrics "['completeness','factual-accuracy','relevance','writing-style','overall']" \
    --pred_metrics "['disagree_flag']" \
    --model "knn"