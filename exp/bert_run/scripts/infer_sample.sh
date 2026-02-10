model_folder=weights/pubmedbert
infer_path=datasets/mediqa-eval-2026-valid-aligned.csv
metric='writing-style'

python exp/bert_run/scripts/infer_main.py \
    --model_folder $model_folder \
    --infer_path $infer_path \
    --metric $metric \
    --batch_size 32 \
    --output_dir exp/bert_run/runs/INFER