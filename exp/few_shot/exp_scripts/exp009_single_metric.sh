single_metrics="['disagree_flag'] ['overall'] ['writing-style'] ['relevance'] ['factual-accuracy'] ['completeness']"

base_run_id=exp-single-metric
folder_path=exp/few_shot/runs/exp-single-metric
mkdir $folder_path

infer_path=exp/few_shot/datasets/val.csv

infer_path="exp/few_shot/datasets/val.csv"
medgemma27b="/data1/xinzhe/microsoft_nlp/workspace/mediqa-competition/models/medgemma1.5-27b"

for m in $single_metrics; do
    run_folder=$folder_path/$m
    mkdir $run_folder
    #path to the input of LLM
    llm_input_json=$run_folder/input.json
    #path to the output of LLm
    output_path=$run_folder/medgemma.json

    #make input
    python exp/few_shot/scripts/make_shot_main.py \
        --split_from exp/few_shot/datasets/train.csv \
        --infer_path "$infer_path" \
        --output_path "$llm_input_json" \
        --shot_num 7 \
        --metrics "$m" \

    #run inference
    echo "Running MedGemma-27b inference..."
    python model_runners/inference.py \
        --model_path "$medgemma27b" \
        --data_path "$llm_input_json" \
        --file_name "$output_path" \
        --run_id "$base_run_id"
    #run evaluation
    echo "start evaluating..."
    python scripts/eval/eval_main.py \
        --input_pred medgemma \
        --output_folder "$run_folder" \
        --metrics "$m"
done
    