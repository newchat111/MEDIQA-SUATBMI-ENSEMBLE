base_folder=exp/finetune/runs/run-per-metric-lr2e-4
mkdir $base_folder
metrics="['relevance'] ['writing-style'] ['disagree_flag'] ['completeness'] ['factual-accuracy'] ['overall']"
# metrics="['relevance']"
for m in $metrics; do
    run_id=$m
    run_folder=$base_folder/$run_id
    mkdir $run_folder

    model_path=exp/finetune/runs/run-per-metric-lr2e-4/$m/checkpoint-50
    train_data=$run_folder/train.json
    val_data=$run_folder/val.json
    llm_input_json=$run_folder/infer.json
    #start finetuning

    #start_inference
    echo "Running MedGemma-27b inference..."
    python model_runners/inference.py \
        --model_path "$model_path" \
        --data_path "$llm_input_json" \
        --file_name $run_folder/medgemma.json \
        --run_id "$base_run_id"

    #start evaluation
    echo "start evaluating..."
    python scripts/eval/eval_main.py \
        --input_pred medgemma \
        --output_folder "$run_folder" \
        --metrics "$m" \
        --eval_sft True

done


