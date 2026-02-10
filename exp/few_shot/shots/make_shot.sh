infer_sample=50
llm_input_json=exp/few_shot/shots/shot_sample.json

python exp/few_shot/scripts/make_shot_main.py \
    --shot_path datasets/shot_df.csv \
    --infer_path datasets/infer_df.csv \
    --sample_n $infer_sample \
    --output_path $llm_input_json