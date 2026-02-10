base_folder=exp/finetune/runs
run_id=run-27b
run_folder=$base_folder/$run_id

mkdir $run_folder

model_path=/data1/xinzhe/microsoft_nlp/workspace/mediqa-competition/models/medgemma1.5-27b
train_data="exp/finetune/data/fold_1/train_ft_input.json"
val_data="exp/finetune/data/fold_1/val_ft_input.json"

python exp/finetune/scripts/finetune.py \
    --pretrained_model_id $model_path \
    --train_data_path $train_data \
    --val_data_path $val_data \
    --output_dir $run_folder \
    --push_to_hub False \
    --num_train_epochs 10 \
    --logging_steps 10 \
    --eval_steps 30

