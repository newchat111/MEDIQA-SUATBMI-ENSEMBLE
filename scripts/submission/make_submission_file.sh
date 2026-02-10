python scripts/submission/make_submission_file.py \
    --pred_path runs/QWEN-ragv2-wgold/qwen.json \
    --save_path runs/QWEN-ragv2-wgold/prediction.csv \
    --metrics "['disagree_flag','completeness','factual-accuracy','relevance','writing-style','overall']"