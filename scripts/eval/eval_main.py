from dataclasses import dataclass
import draccus
from draccus import field
from utils.eval.eval_json import get_prediction, en_small_sample_merge, sft_get_prediction_csv, get_correlations,en_make_submission
from utils.eval import mediqa_eval_script as official
from utils.impute_models.knn import apply_knn_main
import ast
import pandas as pd
import json

@dataclass
class paths:
    input_pred:str = ""
    output_folder:str = ""
    metrics:str = "[]"
    eval_sft:bool = False
    impute_knn:bool = False
    knn_train_path:str = ""
    knn_train_metrics:str = "['completeness','overall']"
    knn_pred_metrics:str = "['disagree_flag']"

@draccus.wrap()
def main(p:paths):
    #turn the str into an actual list
    p.metrics = ast.literal_eval(p.metrics)
    p.knn_train_metrics = ast.literal_eval(p.knn_train_metrics)

    prediction_path = f"{p.output_folder}/{p.input_pred}.json"
    scoresave_path = f"{p.output_folder}/{p.input_pred}_score.json"
    true_path = f"datasets/mediqa-eval-2026-valid.csv"
    df_human = pd.read_csv(true_path)

    if p.eval_sft:
        true_df,pred_df = sft_get_prediction_csv(prediction_path=prediction_path, metrics = p.metrics) #metrics must contain only one metric
    else:  
        true_df,pred_df = get_prediction(true_path=true_path, prediction_path=prediction_path, in_mark_down=True, metrics = p.metrics)

    if p.impute_knn:
        knn_train_df = pd.read_csv(p.knn_train_path)
        pred_df = apply_knn_main(train_df=knn_train_df, pred_df = pred_df, metrics = p.knn_train_metrics, pred_metric_list=p.knn_pred_metrics)

    merged_df = en_small_sample_merge(true_df = true_df, pred_df = pred_df)
    scores = {}
    total_score = 0
    for metric in p.metrics:

        per_metric_df = merged_df[merged_df['metric'] == metric]
        # print(per_metric_df)
        kendalltau, pearson, spearman, _, _, _ = get_correlations(x = per_metric_df['value_x'], y = per_metric_df['value_y'])

        mean_corr = (kendalltau + pearson + spearman) / 3
        scores[metric] = mean_corr
        total_score += mean_corr

    scores["ALL_en_ALL_mean"] = total_score / len(p.metrics)
    #save the score
    with open(scoresave_path, 'w') as f:
        json.dump(scores, f, indent = 2)

    submission = en_make_submission(df = pred_df)
    submission.to_csv(f"{p.output_folder}/{p.input_pred}_submit.csv", index = False)
    #dump the official score
    official.main(df_human = df_human, df_auto=submission, save_path = f"{p.output_folder}/{p.input_pred}_officialscore.json")
    #save the dataframe
    # pred_df.to_csv(f"{p.output_folder}/{p.input_pred}.csv", index = False)
    
if __name__ == "__main__":
    main()
    