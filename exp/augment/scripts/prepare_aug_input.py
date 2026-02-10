import pandas as pd
from exp.augment.scripts.prompts import prompt
import json
from utils.dataset_helper.get_data import PER_SYSTEM_KEY

def prepare_input(df:pd.DataFrame, augment_multiplier:int, metrics:list[str] = None, scores:list[str] = None, datasets:list[str] = ["woundcare", "iiyi"]):
    if metrics and scores:
        df = df[(df['metric'].isin(metrics)) & (df['label'].isin(scores)) & (df['dataset'].isin(datasets))]

    llm_input = []


    grouped_by_key = df.groupby(by=PER_SYSTEM_KEY)
    print(f"length of df: {len(grouped_by_key)}")
    for key_val, group in grouped_by_key:
        entry_temp = {
        "role": "user",
        "content": None,
        "key": None
        }

        content_temp = [{"type": "text",
                        "text": None
                        }]

        response = group['candidate'].iloc[0]
        query = group['query_text'].iloc[0]
        
        content_temp[0]['text'] = prompt(n = augment_multiplier, query = query, response=response)
        entry_temp["content"] = content_temp
        entry_temp["key"] = key_val
        # print(key_val)

        llm_input.append(entry_temp)

    return llm_input

if __name__ == "__main__":
    augment_map = {"writing-style": [0.0,0.5],
                    "completeness": [0.0],
                    "relevance": [0.0,0.5]
                    }


    df = pd.read_csv("exp/augment/inputs/fold4/train.csv")
    en_df = df[df['lang'] == 'en']

    for metric, scores in augment_map.items():
        save_path = f"exp/augment/inputs/fold4/{metric}_woundcare_augment_fold4.json"


        input_json = prepare_input(df = en_df, 
                                augment_multiplier=5, 
                                metrics = [metric], 
                                scores = scores, 
                                datasets=['woundcare'])
        
        with open(save_path, 'w') as f:
            json.dump(input_json, f, indent=2)
    # print(len(en_df))
    # print(en_df[PER_SYSTEM_KEY].isna().sum())