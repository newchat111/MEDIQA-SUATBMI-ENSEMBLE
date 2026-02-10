import json
from utils.dataset_helper.get_data import retrieve_data, METRICS, PER_SYSTEM_KEY
from utils.eval.mediqa_eval_script import LANG2METRICS, EVAL_COLS_UNIQUE, get_correlations
import pandas as pd
from utils.impute_models.knn import apply_knn_main

# def unformatted_output_corrector():
#     target_map = {
#             "factual-accuracy": ["factual_accuracy", "factual-accuracy"],
#             "writing-style": ["writing_style", "writing-style"]
#         }


def sft_get_prediction_csv(prediction_path:list, metrics:list[str]):
    with open(prediction_path, 'r') as f:
        output = json.load(f)

    df = pd.read_csv("datasets/mediqa-eval-2026-valid.csv").sort_values(by=["encounter_id", "candidate_author_id"])

    orig_mask = (df['lang'] != 'zh') & (df['rater_id'] != 'SG')
    zh_mask = (df['lang'] != 'zh')

    df = df[orig_mask]
    #this the index for score_template

    score_template = df
    prediction_df = pd.DataFrame(columns=df.columns)
    orig_df = pd.DataFrame(columns=df.columns)

    for o in output:
        
        response = o['response'][0]


        key = o["input"][0]["key"][0]
        #for storing output
        rates = {'disagree_flag': None, 'completeness': None,'factual-accuracy':None, 'relevance':None, 'writing-style':None, 'overall':None}
        #only supports one metric for now
        m = metrics[0]
        rates[m] = response

        for metric, score in rates.items():

            if score is None:
                continue

            mask = (
                (df['dataset'] == key['dataset']) &
                (df['encounter_id'] == key['encounter_id']) &
                (df['lang'] == key['lang']) &
                (df['candidate_author_id'] == key['candidate_author_id']) &
                (df['metric'] == metric)
                )
            
            slice_df = score_template.loc[mask].copy()
            orig_slice_df = score_template.loc[mask].copy()
            slice_df['value'] = float(score)

            orig_df = pd.concat([orig_df, orig_slice_df], ignore_index=True)
            prediction_df = pd.concat([prediction_df, slice_df], ignore_index=True)

            orig_df = pd.concat([orig_df, orig_slice_df], ignore_index=True)
            prediction_df = pd.concat([prediction_df, slice_df], ignore_index=True)

    return orig_df, prediction_df

def get_prediction(true_path:str,
                   prediction_path, 
                   in_mark_down:bool=False, 
                   metrics:list[str]=["overall"], 
                   apply_knn:bool = False, 
                   knn_train_path:str = None,
                   ensemb_strat:str = "vote"):
    


    with open(prediction_path, 'r') as f:
        output = json.load(f)

    df = pd.read_csv(true_path).sort_values(by=["encounter_id", "candidate_author_id"])
    orig_mask = (df['lang'] != 'zh') & (df['rater_id'] != 'SG')
    zh_mask = (df['lang'] != 'zh')

    df = df[orig_mask]
    #this the index for score_template

    score_template = df
    prediction_df = pd.DataFrame(columns=df.columns)
    orig_df = pd.DataFrame(columns=df.columns)

    for o in output:
        if in_mark_down:
            try:
                response = read_gemma_output(o['response'][0])
                # print(response)
            except:
                print(o['response'][0])
                response = read_gemma_output(o['response'][0])
                print(response)
                raise ValueError("not a strict json")
        else:
            response = json.loads(o['response'][0])

        key = o["input"][0]["key"][0]
        #for storing output
        rates = {'disagree_flag': None, 'completeness': None,'factual-accuracy':None, 'relevance':None, 'writing-style':None, 'overall':None}
        #make parser more robust to llm outputs
        target_map = {
            "factual-accuracy": ["factual_accuracy", "factual-accuracy"],
            "writing-style": ["writing_style", "writing-style"]
        }
        raw_dict = response[0]
        normalized_response = {}
        for k, v in raw_dict.items():
            # Check if the current key is one of our "problem" keys
            if k in target_map["factual-accuracy"]:
                normalized_response["factual-accuracy"] = v
            elif k in target_map["writing-style"]:
                normalized_response["writing-style"] = v
            else:
                # Keep everything else (like disagree_flag) exactly as it is
                normalized_response[k] = v
        normalized_response = [normalized_response]
        
        for m in metrics:
            rates[m] = normalized_response[0][m]

        for metric, score in rates.items():

            if score is None:
                continue

            mask = (
                (df['dataset'] == key['dataset']) &
                (df['encounter_id'] == key['encounter_id']) &
                (df['lang'] == key['lang']) &
                (df['candidate_author_id'] == key['candidate_author_id']) &
                (df['metric'] == metric)
                )

            slice_df = score_template.loc[mask].copy()
            orig_slice_df = score_template.loc[mask].copy()
            slice_df['value'] = float(score)

            #add knn
            if apply_knn:
                knn_train_df = pd.read_csv(knn_train_path)
                disagree_flag_score = apply_knn_main(train_df=knn_train_df, pred_df=slice_df)

            orig_df = pd.concat([orig_df, orig_slice_df], ignore_index=True)
            prediction_df = pd.concat([prediction_df, slice_df], ignore_index=True)
            # print(slice_df)
        
    # orig_df = pd.concat([orig_df,complement], ignore_index=True)
    # prediction_df = pd.concat([prediction_df,complement], ignore_index = True)
    
    return orig_df, prediction_df

def en_small_sample_merge(true_df:pd.DataFrame, pred_df:pd.DataFrame) -> pd.DataFrame:

    merged_df = pd.merge(left = true_df, right=pred_df, on = EVAL_COLS_UNIQUE)    
    # for metric in LANG2METRICS['en']:
    return merged_df

def read_gemma_output(js:str):
    clean_str = js.strip("```json").strip("```").strip()
    # Step 2: Parse it
    data = json.loads(clean_str)
    return data

def en_make_submission(df:pd.DataFrame,true_df:pd.DataFrame)->pd.DataFrame:
    zh_df = true_df
    zh_df = zh_df[zh_df['lang'] == 'zh']
    #work only on en
    # zh_df.loc[:, 'rater_id'] = 'NA'
    zh_df.loc[:, 'value'] = -1

    submission_df = pd.concat([df,zh_df])
    return submission_df

def make_submission(df:pd.DataFrame,true_df:pd.DataFrame)->pd.DataFrame:
    zh_df = true_df
    zh_df = zh_df[zh_df['lang'] == 'zh']
    #work only on en
    # zh_df.loc[:, 'rater_id'] = 'NA'
    zh_df.loc[:, 'value'] = -1

    submission_df = pd.concat([df,zh_df])
    return submission_df



if __name__ == "__main__":
    metrics = METRICS

    prediction_path = "exp/few_shot/runs/exp-testshot/shot3/medgemma.json"
    save_path = "my_test/medgemma_score.json"
    true_df, pred_df = get_prediction(prediction_path=prediction_path, in_mark_down=True, metrics = metrics)
    pred_df.to_csv("my_test/pred_df.csv", index = False)
    merged_df = en_small_sample_merge(true_df = true_df, pred_df = pred_df)

    merged_df.to_csv("my_test/merged_df.csv", index = False)
    scores = {}
    total_score = 0
    for metric in metrics:

        per_metric_df = merged_df[merged_df['metric'] == metric]
        # print(per_metric_df)
        kendalltau, pearson, spearman, _, _, _ = get_correlations(x = per_metric_df['value_x'], y = per_metric_df['value_y'])

        mean_corr = (kendalltau + pearson + spearman) / 3
        scores[metric] = mean_corr
        total_score += mean_corr

    scores["ALL_en_ALL_mean"] = total_score / len(metrics)

    with open(save_path, 'w') as f:
        json.dump(scores, f, indent = 2)
        # print(kendalltau)
    
    
    
        

        







