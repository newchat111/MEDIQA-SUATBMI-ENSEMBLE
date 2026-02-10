
folder_path=my_test/shots
llm_input_json=$folder_path/input_gold_7shot.json
# output_path="$folder_path/qwen-7shot.json"
metrics="['disagree_flag','completeness','factual-accuracy','relevance','writing-style','overall']"
# metrics="['disagree_flag','factual-accuracy','overall']"
# metrics="['relevance']"
infer_path="datasets/test/mediqa-eval-2026-test-aligned.csv"

python exp/few_shot/scripts/make_gold_shot_main.py \
        --split_from exp/few_shot/datasets/train.csv \
        --infer_path "$infer_path" \
        --output_path "$llm_input_json" \
        --shot_num 7 \
        --metrics "$metrics" \
        --en_only True