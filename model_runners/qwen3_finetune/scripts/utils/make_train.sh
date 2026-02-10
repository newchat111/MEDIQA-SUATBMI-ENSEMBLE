PATH_TO_FOLDER="/workspace/xinzhe"
#path to the dataframes
commercial_df=$PATH_TO_FOLDER/data/gemma3_summary_commercial.csv
noncommercial_df=$PATH_TO_FOLDER/data/gemma3_summary_noncommercial.csv
#path to where the images exist
base_commercial="/workspace/retrieval/datasets/commercial_extracted"
base_noncommercial="/workspace/retrieval/datasets/noncommercial_extracted"
#specify the path to the folder
output_com=data_json/all_new_prompt/com
output_noncom=data_json/all_new_prompt/noncom

prompt_path=data_json/prompts/p001.txt

# python tools/utils/make_train_json_main.py \
#     --input_path $commercial_df \
#     --output_path $output_com \
#     --base_path $base_commercial \
#     --prompt_path $prompt_path \
#     --train_ratio 0.995

python tools/utils/make_train_json_main.py \
    --input_path $noncommercial_df \
    --output_path $output_noncom \
    --base_path $base_noncommercial \
    --prompt_path $prompt_path \
    --train_ratio 0.995