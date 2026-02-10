
folder_path=my_test/shots
llm_input_json=$folder_path/input_rel.json
output_path="$folder_path/medgemma.json"
# metrics="['disagree_flag','completeness','factual-accuracy','relevance','writing-style','overall']"
# metrics="['disagree_flag','factual-accuracy','overall']"
metrics="['relevance']"
infer_path="datasets/mediqa-eval-2026-valid-aligned.csv"

python exp/few_shot/scripts/make_shot_main.py \
        --split_from exp/few_shot/datasets/train.csv \
        --infer_path "$infer_path" \
        --output_path "$llm_input_json" \
        --shot_num 2 \
        --metrics "$metrics" \
        --woundcare_only False \
        --exclude_image True \
        --flatten True