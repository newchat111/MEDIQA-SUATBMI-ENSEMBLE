run_id=001
folder_path=exp/reinforced_few_shot/runs/$run_id

mkdir exp/reinforced_few_shot/runs/$run_id

input_path=exp/reinforced_few_shot/shots/shots.json

python model_runners/inference.py \
    --model_path /data1/xinzhe/microsoft_nlp/workspace/mediqa-competition/models/medgemma1.5-27b \
    --data_path $input_path \
    --file_name $folder_path/medgemma.json \
    --run_id $run_id \
