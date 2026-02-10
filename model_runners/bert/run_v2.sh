prepare_data_out=exp/bert_run/runs/mediqa-train-2026-en_final_dataset.csv

python prepare_data.py \
    --input_file datasets/mediqa-eval-2026-valid.csv \
    --derma_json_root datasets/original_datasets/iiyi/valid_ht.json \
    --derma_image_root /data/liyuan/datasets/competition/derma/data/iiyi/images_final/images_valid \
    --wound_json_root datasets/original_datasets/woundcare/valid.json \
    --wound_image_root /data/liyuan/datasets/competition/woundcare/dataset-challenge-mediqa-2025-wv/images_final/images_valid \
    --model_path /home/liyuan/models/blip2-opt-2.7b/ \
    --output_file $prepare_data_out

python train_bert.py \
    --strategy_type dataset_specific \
    --input_path $prepare_data_out \
    --template_path /home/liyuan/projects/competition/mine/datasets/mediqa-train-2026-en.csv \
    --output_path exp/bert_run/runs

