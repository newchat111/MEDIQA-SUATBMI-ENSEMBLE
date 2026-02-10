batch_size=32
run_id=run001

python tools/eval_func/inference.py \
    --model_path Qwen2.5-VL-3B-Instruct \
    --data_path data_json/all/inference/infer_dataset_conversations.json \
    --file_name infer_vanilla_test.json \
    --batch_size $batch_size \
    --run_id $run_id

python tools/eval_func/inference.py \
    --model_path runs/run001 \
    --data_path data_json/all/inference/infer_dataset_conversations.json \
    --file_name infer_finetuned_test.json \
    --batch_size $batch_size \
    --run_id $run_id