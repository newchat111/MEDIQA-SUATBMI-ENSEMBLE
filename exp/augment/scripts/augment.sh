metrics="writing-style relevance completeness"
output_folder=exp/augment/runs/augment_woundcare
base_run_id=augment_woundcare
medgemma27b="/data1/xinzhe/microsoft_nlp/workspace/mediqa-competition/models/medgemma1.5-27b"

mkdir -p $output_folder

for m in $metrics;do
    llm_input_json=exp/augment/inputs/fold4/${m}_woundcare_augment_fold4.json
    output_path=$output_folder/${m}_augment.json

    python model_runners/inference.py \
        --model_path "$medgemma27b" \
        --data_path "$llm_input_json" \
        --file_name $output_path \
        --run_id "$base_run_id"
done