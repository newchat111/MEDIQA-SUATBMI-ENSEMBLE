import pandas as pd
import json
import os
import torch
import flash_attn_2_cuda as flash_attn_cuda

# Define Constants
COMMERCIAL_PATH = "/workspace/retrieval/datasets/commercial_extracted"
NONCOMMERCIAL_PATH = "/workspace/retrieval/datasets/noncommercial_extracted"
PATH_TO_FOLDER = "/workspace/xinzhe"

# Load Dataframes
commercial_df = pd.read_csv(f"{PATH_TO_FOLDER}/data/gemma3_summary_commercial.csv")[0:5]
noncommercial_df = pd.read_csv(f"{PATH_TO_FOLDER}/data/gemma3_summary_noncommercial.csv")[0:5]

def build_json_data(df, base_path, prompt_path):
    dataset_list = []
    
    for _, row in df.iterrows():
        prefix = row['file_prefix']
        
        # Construct file paths
        image_path = os.path.join(base_path, f"{prefix}.jpg")
        text_path = os.path.join(base_path, f"{prefix}.txt")
        
        # Read the content of the .txt file
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                summary_text = f.read().strip()
            with open(prompt_path, 'r', encoding = 'utf-8') as p:
                prompt_text = p.read()
        except FileNotFoundError:
            print(f"Warning: Text file not found for {prefix}")
            continue
        
        # Structure for the JSON
        entry = {
            "image": image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nWhat's the main object in this picture?"
                },
                {
                    "from": "gpt",
                    "value": summary_text
                }
            ]
        }
        dataset_list.append(entry)
    
    return dataset_list

# Process and merge
full_dataset = []
full_dataset.extend(build_json_data(commercial_df, COMMERCIAL_PATH))
full_dataset.extend(build_json_data(noncommercial_df, NONCOMMERCIAL_PATH))

# Export
output_file = "dataset_conversations.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(full_dataset, f, indent=4, ensure_ascii=False)

print(f"Done! Created {output_file} with {len(full_dataset)} entries.")