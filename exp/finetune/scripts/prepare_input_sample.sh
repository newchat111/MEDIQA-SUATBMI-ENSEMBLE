<<<<<<< HEAD
run_folder=my_test

python exp/finetune/scripts/setup/prepare_input.py \
    --train_input_path exp/few_shot/datasets/train.csv \
    --val_input_path exp/few_shot/datasets/val.csv \
=======
run_folder=exp/finetune/runs/data

python exp/finetune/scripts/setup/prepare_input.py \
    --train_input_path exp/finetune/data/fold0/train.csv \
    --val_input_path exp/finetune/data/fold0/val.csv \
>>>>>>> master
    --output_folder $run_folder \
    --metrics "['overall']"