run_id=exp005_all

mkdir exp/few_shot/runs/$run_id

folder_path=exp/few_shot/runs/$run_id
shot_path=datasets/shot_df.csv
infer_path=datasets/mediqa-eval-2026-valid-aligned.csv

# infer_sample=50
llm_input_json=exp/few_shot/runs/$run_id/input.json
qwen_model=models/Qwen2.5-3B-VL-instruct
medgemma27b=/data1/xinzhe/microsoft_nlp/workspace/mediqa-competition/models/medgemma1.5-27b

# device=cuda:1

#prepare input
python exp/few_shot/scripts/make_shot_main.py \
    --shot_path $shot_path \
    --infer_path $infer_path \
    --output_path $llm_input_json

#run medgemma27b
python model_runners/inference.py \
    --model_path /data1/xinzhe/microsoft_nlp/workspace/mediqa-competition/models/medgemma1.5-27b \
    --data_path $llm_input_json \
    --file_name $folder_path/medgemma.json \
    --run_id $run_id \
    # --device $device

# run qwen
python model_runners/inference.py \
    --model_path models/Qwen2.5-3B-VL-instruct \
    --data_path $llm_input_json \
    --file_name $folder_path/qwen.json \
    --run_id $run_id \


#evaluate