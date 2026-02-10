orig_train=data_json/all/train_dataset_conversations.json
orig_val=data_json/all/val_dataset_conversations.json

output_train=data_json/all_new_prompt/train_new.json
output_val=data_json/all_new_prompt/val_new.json

prompt_path=data_json/prompts/p001.txt

python tools/utils/change_prompt.py \
    --input_json $orig_train \
    --output_json $output_train \
    --prompt_path $prompt_path

python tools/utils/change_prompt.py \
    --input_json $orig_val \
    --output_json $output_val \
    --prompt_path $prompt_path


