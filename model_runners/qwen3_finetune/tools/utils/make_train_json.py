import pandas as pd
import json
import os
import torch
import random
from tqdm import tqdm

# Define Constants
COMMERCIAL_PATH = "/workspace/retrieval/datasets/commercial_extracted"
NONCOMMERCIAL_PATH = "/workspace/retrieval/datasets/noncommercial_extracted"

PATH_TO_COM_DF = "/workspace/xinzhe/data/gemma3_summary_commercial.csv"
PATH_TO_NONCOM_DF = "/workspace/xinzhe/data/gemma3_summary_noncommercial.csv"


def build_json_data(data, base_path, prompt_path,train_ratio = 0.99):
    datasets = []
    #split the data
    data = data.sample(frac=1).reset_index(drop=True)

    split_index = int(len(data) * train_ratio)
    train_data = data.iloc[:split_index]
    val_data = data.iloc[split_index:]
    
    for df in [train_data, val_data]:
        dataset_list = []
        # Read the content of the .txt file
        with open(prompt_path, 'r', encoding = 'utf-8') as p:
            prompt_text = p.read().strip()

        for _, row in tqdm(df.iterrows(), total=len(df)):
            prefix = row['file_prefix']
            summary_text = row['caption_summary']
            # Construct file paths
            image_path = os.path.join(base_path, f"{prefix}.jpg")
            text_path = os.path.join(base_path, f"{prefix}.txt")
            
            # Structure for the JSON
            entry = {
                "image": image_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\n" + prompt_text
                    },
                    {
                        "from": "gpt",
                        "value": summary_text
                    }
                ]
            }
            dataset_list.append(entry)

        datasets.append(dataset_list)
    
    train_json = datasets[0]
    val_json = datasets[1]

    return train_json, val_json