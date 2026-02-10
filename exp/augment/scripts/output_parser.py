import json
import pandas as pd
from utils.eval.eval_json import read_gemma_output

def parse(path:str, 
          df_to_augment:pd.DataFrame,
          metric:str):
    with open(path, 'r') as f:
        output = json.load(f)
    #to store the augmentations
    augmented_rows = []

    for o in output:
        response = read_gemma_output(o['response'][0])
        #collect the keys
        key = o["input"][0]["key"]
        dataset = key[0]
        lang = key[1]
        encounter_id = key[2]
        candidate_author_id = key[3]

        base_slice = df_to_augment[
            (df_to_augment["dataset"] == dataset) &
            (df_to_augment["lang"] == lang) &
            (df_to_augment["encounter_id"] == encounter_id) &
            (df_to_augment["candidate_author_id"] == candidate_author_id) &
            (df_to_augment["metric"] == metric)
        ]

        for entry in response:
            question = next(v for k, v in entry.items() if "question" in k)
            response = next(v for k, v in entry.items() if "response" in k)

            augment_slice = base_slice.copy()
            augment_slice["query_text"] = question
            augment_slice["candidate"] = response

            augmented_rows.append(augment_slice)
    #report
    print(f"total augmentation rows:{len(augmented_rows)}")
    # print(f"original length:{len(df_to_augment)}")

    return pd.concat(augmented_rows, ignore_index=True)

if __name__ == "__main__":

    rel_path = "exp/augment/runs/augment_woundcare/relevance_augment.json"
    comp_path = "exp/augment/runs/augment_woundcare/completeness_augment.json"
    wrt_path = "exp/augment/runs/augment_woundcare/writing-style_augment.json"

    paths = [rel_path, comp_path, wrt_path]
    metrics = ['relevance', 'completeness', 'writing-style']

    df_to_augment = pd.read_csv("exp/augment/inputs/fold4/train.csv")
    print(f"original len: {len(df_to_augment)}")
    all_augments = []
    for i in range(len(paths)):
        augmented = parse(path = paths[i], df_to_augment=df_to_augment, metric = metrics[i])
        all_augments.append(augmented)
        # print(f"augment len: {len(augmented)}")

    df_to_augment = pd.concat([df_to_augment] + all_augments, ignore_index=True)
    df_to_augment.to_csv("exp/augment/inputs/fold4/train_augmented.csv", index = False)
    print(f"after augmentation length:{len(df_to_augment)}")
    
