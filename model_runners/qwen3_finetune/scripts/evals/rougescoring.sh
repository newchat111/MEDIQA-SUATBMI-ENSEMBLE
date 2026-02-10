python tools/eval_func/rougescoring.py \
    --input_path eval_result/run001/infer_result/infer_finetune.json \
    --file_name score_infer_finetune.json \
    --run_id run001 \


python tools/eval_func/rougescoring.py \
    --input_path eval_result/run001/infer_result/infer_vanilla.json \
    --file_name score_infer_vanilla.json \
    --run_id run001 \