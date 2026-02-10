import pandas as pd
import json

aligned_df = pd.read_csv("datasets/mediqa-eval-2026-valid-aligned.csv")

#read the captions
with open('exp/image_to_cap/outputs/medgem4b_caption.json', 'r') as f:
    captions = json.load(f)

if __name__ == "__main__":
    #start a new column named caption

    save_path = "exp/image_to_cap/outputs/mediqa-eval-2026-valid-aligned-captioned.csv"
    aligned_df['caption'] = ""

    for entry in captions:
        cap = entry["response"][0]
        key = entry["input"][0]["key"]
        
        #insert the caption
        aligned_df.loc[aligned_df['encounter_id'] == key, 'caption'] = cap

    aligned_df.to_csv(save_path, index = False)