base_folder=exp/finetune/runs/finetune-per-metric
mkdir $base_folder
metrics="['factual-accuracy']"
# metrics="['relevance']"
for m in $metrics; do
    run_id=$m
    run_folder=$base_folder/$run_id
    mkdir -p $run_folder
    #prepare data
    python exp/finetune/scripts/setup/prepare_input.py \
        --train_input_path exp/few_shot/datasets/train.csv \
        --val_input_path exp/few_shot/datasets/val.csv \
        --output_folder $run_folder \
        --metrics $m

    model_path=/data1/xinzhe/microsoft_nlp/workspace/mediqa-competition/models/medgemma1.5-27b
    train_data=$run_folder/train.json
    val_data=$run_folder/val.json
    infer_data=$run_folder/infer.json
    #start finetuning
    python exp/finetune/scripts/finetune.py \
        --pretrained_model_id $model_path \
        --train_data_path $train_data \
        --val_data_path $val_data \
        --output_dir $run_folder \
        --push_to_hub False \
        --num_train_epochs 4 \
        --logging_steps 1 \
        --eval_steps 4 \
        --learning_rate 2e-7 \
        --warmup_ratio 0.1 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 1 \
        --eval_accumulation_steps 10 \
        --lora_dropout 0.1 \
        --lr_scheduler_type cosine \
        --run_name $run_id \
        --wandb_project "medgemma" \
        --wandb_group $m \
        --lora_r 8


    # #start_inference
    # echo "Running MedGemma-27b inference..."
    # python model_runners/inference.py \
    #     --model_path "$medgemma27b" \
    #     --data_path "$llm_input_json" \
    #     --file_name $infer_data \
    #     --run_id "$base_run_id"

    # #start evaluation
    # echo "start evaluating..."
    # python scripts/eval/eval_main.py \
    #     --input_pred medgemma \
    #     --output_folder "$folder_path" \
    #     --metrics "$m" \
    #     --eval_sft True

done


