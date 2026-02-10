base_run_id="exp-explain"
medgemma27b="/data1/xinzhe/microsoft_nlp/workspace/mediqa-competition/models/medgemma1.5-27b"
folder_path=exp/reinforced_few_shot/runs/$base_run_id

mkdir $folder_path

output_path=$folder_path/medgemma.json
llm_input_json=exp/reinforced_few_shot/input_explain.json

python model_runners/inference.py \
    --model_path "$medgemma27b" \
    --data_path "$llm_input_json" \
    --file_name $output_path \
    --run_id "$base_run_id"
