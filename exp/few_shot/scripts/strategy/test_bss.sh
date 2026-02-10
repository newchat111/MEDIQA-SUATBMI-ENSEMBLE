shot_df_path=exp/few_shot/datasets/train.csv
infer_df_path=exp/few_shot/datasets/val.csv
llm_input_json=my_test/test_bootstrap_shots/input.json
metrics="['disagree_flag','completeness','factual-accuracy','relevance','writing-style','overall']"

python exp/few_shot/scripts/strategy/bootstrap_shots.py \
    --bs_sample_from exp/few_shot/datasets/train.csv \
    --infer_path $infer_df_path \
    --output_path "$llm_input_json" \
    --shot_num 7 \
    --metrics "$metrics" 