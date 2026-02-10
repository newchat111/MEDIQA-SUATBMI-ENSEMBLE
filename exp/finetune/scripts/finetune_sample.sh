base_folder=exp/finetune/runs
run_id=run004-overall-27b
run_folder=$base_folder/$run_id

mkdir $run_folder

# model_path=/data1/xinzhe/microsoft_nlp/mediqa-competition/models/medgemma1.5-4b/
model_path=/data1/xinzhe/microsoft_nlp/workspace/mediqa-competition/models/medgemma1.5-27b
train_data="exp/finetune/data/new_temp/train_ft_input.json"
val_data="exp/finetune/data/new_temp/val_ft_input.json"

python exp/finetune/scripts/finetune.py \
    --pretrained_model_id $model_path \
    --train_data_path $train_data \
    --val_data_path $val_data \
    --output_dir $run_folder \
    --push_to_hub False \
    --num_train_epochs 5 \
    --logging_steps 2 \
    --eval_steps 10 \
    --learning_rate 2e-6 \
    --warmup_ratio 0.4 \
    --per_device_train_batch_size 4 \
    --lora_dropout 0.1 \
    --run_name $run_id 

