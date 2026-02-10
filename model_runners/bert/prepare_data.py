import pandas as pd
import argparse
import os
from sklearn.model_selection import KFold
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import sys
import random

def remove_zh(input_file):
    df = pd.read_csv(input_file)
    if 'lang' in df.columns:
        df_en = df[df['lang'] == 'en'].copy()
    else:
        print("Warning: 'lang' column not found, assuming all data is valid.")
        df_en = df.copy()
    return df_en

def keep_one_rater(df):
    dedup_columns = [
        'dataset', 
        'encounter_id', 
        'lang', 
        'candidate', 
        'candidate_author_id', 
        'metric'
    ]
    cols_to_use = [c for c in dedup_columns if c in df.columns]
    df_one_rater = df.drop_duplicates(subset=cols_to_use, keep='first')
    return df_one_rater

def align(df, derma_json_root, derma_image_root, wound_json_root, wound_image_root):
    def clean_text(text):
        if text is None or pd.isna(text):
            return ""
        text = str(text)
        return text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()

    def load_json_db(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {str(item['encounter_id']): item for item in data}

    def get_gold_data(json_item, lang):
        responses = json_item.get('responses', [])
        if not responses:
            return "", "[]"
        
        all_gold_list = []
        for r in responses:
            content = r.get(f"content_{lang}")
            if content:
                cleaned_content = clean_text(content)
                if cleaned_content:
                    all_gold_list.append(cleaned_content)

        all_golds_json = json.dumps(all_gold_list, ensure_ascii=False)
        return all_golds_json

    def find_image_path(json_item, root_dir):
        image_ids = json_item.get('image_ids', [])
        if not image_ids:
            return None
        img_filename = image_ids[0]
        full_path = os.path.join(root_dir, img_filename)
        if os.path.exists(full_path):
            return full_path
        else:
            return None

    print("Loading JSON databases...")
    db_derma = load_json_db(derma_json_root)
    db_wound = load_json_db(wound_json_root)

    aligned_data = []
    missing_count = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Aligning Data"):
        dataset = row['dataset']
        enc_id = str(row['encounter_id']) 
        lang = row['lang']
        candidate_author_id = row['candidate_author_id']
        
        if dataset == 'iiyi':
            db = db_derma
            img_root = derma_image_root
        elif dataset == 'woundcare':
            db = db_wound
            img_root = wound_image_root
        else:
            continue
            
        item_data = db.get(enc_id)
        
        if not item_data:
            missing_count += 1
            continue

        q_title = item_data.get(f"query_title_{lang}", "")
        q_content = item_data.get(f"query_content_{lang}", "")
        query_text = clean_text(f"{q_title or ''} {q_content or ''}")
        all_golds_json = get_gold_data(item_data, lang)
        candidate_text = clean_text(row['candidate'])
        img_path = find_image_path(item_data, img_root)
        
        new_row = {
            "dataset": dataset,
            "encounter_id": enc_id,
            "lang": lang,
            "candidate": candidate_text,
            "candidate_author_id": candidate_author_id,
            "metric": row['metric'],
            "label": row['value'],
            "query_text": query_text,
            "image_path": img_path,
            "gold_texts": all_golds_json
        }
        aligned_data.append(new_row)
        
    if missing_count > 0:
        print(f"Warning: {missing_count} rows failed to align (ID not found in JSON).")
        
    return pd.DataFrame(aligned_data)

def create_folds(df, fold_num):
    df["fold"] = -1
    unique_ids = df["encounter_id"].unique()    
    kf = KFold(n_splits=fold_num, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(unique_ids)):
        val_ids = unique_ids[val_idx]
        df.loc[df["encounter_id"].isin(val_ids), "fold"] = fold
    
    return df


def generate_captions(df, model_path):
    final_model_path = model_path
    if os.path.exists(os.path.join(model_path, "snapshots")):
        snapshot_dir = os.path.join(model_path, "snapshots")
        subfolders = [f.path for f in os.scandir(snapshot_dir) if f.is_dir()]
        if subfolders:
            final_model_path = subfolders[0]
            print(f"Detected snapshot directory, using: {final_model_path}")

    print(f"Loading VLM model from {final_model_path}...")
    try:
        processor = AutoProcessor.from_pretrained(final_model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            final_model_path,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        ).eval()    
    except Exception as e:
        print(f"Error loading model: {e}")
        return df

    unique_df = df[['dataset', 'encounter_id', 'image_path']].drop_duplicates(subset=['dataset', 'encounter_id'])
    key_to_caption = {} 

    PROMPT_DERM = (
        "You are a dermatologist. Describe the lesion morphology using strictly telegraphic medical terms. "
        "NO full sentences. NO Markdown. Limit to 60 words. "
        "Focus on: "
        "1. Primary Lesion: (e.g., Macule, Papule, Plaque, Wheal, Nodule). "
        "2. Surface: (e.g., Scale, Crust/Scab, Ulcer, Excoriation). "
        "3. Color: (e.g., Erythematous, Purpuric, Hyperpigmented). "
        "4. Arrangement: (e.g., Grouped, Linear, Disseminated)."
    )
    PROMPT_WOUND = (
        "You are a wound specialist. Analyze this wound for infection and healing status. "
        "Output a concise paragraph (under 80 words). NO Markdown. "
        "CRITICAL: You MUST explicitly state 'Signs of infection present' or 'No signs of infection'. "
        "Describe: "
        "1. Tissue Viability: (Granulation, Slough, Eschar/Necrosis). "
        "2. Exudate: (None, Serous, Purulent/Pus, Bloody). "
        "3. Wound Edges: (Erythema/Redness, Swelling, Maceration). "
        "4. Depth & Status: (Superficial vs Deep; Healing vs Dehisced)."
    )
    
    for _, row in tqdm(unique_df.iterrows(), total=len(unique_df), desc="Generating Captions"):
        enc_id = str(row['encounter_id'])
        dataset = row['dataset']
        key = (dataset, enc_id)
        
        img_path = str(row.get('image_path', '')).strip()
        dataset_name = str(row.get('dataset', '')).lower()
        
        if 'wound' in dataset_name:
            user_prompt = PROMPT_WOUND
        else:
            user_prompt = PROMPT_DERM
            
        if img_path and img_path != 'None' and os.path.exists(img_path):
            try:
                image = Image.open(img_path).convert("RGB")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": user_prompt}
                        ]
                    }
                ]
                text = processor.apply_chat_template(messages, add_generation_prompt=True)
                
                inputs = processor(
                    text=text, 
                    images=image, 
                    return_tensors="pt"
                ).to(model.device)

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs, 
                        max_new_tokens=90,
                        temperature=0.1,
                        do_sample=False,
                        repetition_penalty=1.1,
                    )
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
                clean_caption = output_text.replace('\n', ' ').strip()
                key_to_caption[key] = clean_caption
                
            except Exception as e:
                print(f"\n[Error] {img_path}: {e}")
                key_to_caption[key] = "Error generating caption."
        else:
            key_to_caption[key] = "No image available."
            
    def map_caption(row):
        k = (row['dataset'], str(row['encounter_id']))
        return key_to_caption.get(k, "")
        
    df['image_caption'] = df.apply(map_caption, axis=1)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="输入 CSV 文件路径")
    parser.add_argument("--derma_image_root", type=str, required=True, help="iiyi 图像根目录")
    parser.add_argument("--wound_image_root", type=str, required=True, help="woundcare 图像根目录")
    parser.add_argument("--derma_json_root", type=str, required=True, help="iiyi JSON 文件路径")
    parser.add_argument("--wound_json_root", type=str, required=True, help="woundcare JSON 文件路径")
    parser.add_argument("--model_path", type=str, required=True, help="VLM 模型路径")
    parser.add_argument("--fold_num", type=int, default=5, help="分折数，默认5折")
    parser.add_argument("--output_file", type=str, required=True, help="输出csv文件路径")
    args = parser.parse_args()
    os.makedirs(args.output_file, exist_ok=True)
    
    # 1. Load & Filter
    df = remove_zh(args.input_file)
    df.to_csv(os.path.join(args.output_file, "template_en.csv"), index=False)
    print(f"Filtered non-English entries, saved to {os.path.join(args.output_file, 'template_en.csv')}")
    
    # 2. Dedup
    df = keep_one_rater(df)
    
    # 3. Align
    df = align(df, args.derma_json_root, args.derma_image_root, args.wound_json_root, args.wound_image_root)

    # 4. Create Folds
    df = create_folds(df, args.fold_num)
    
    # 5. Caption
    df = generate_captions(df, args.model_path)
    
    # 6. Save (Optional but recommended)
    df.to_csv(os.path.join(args.output_file, "final_dataset.csv"), index=False)
    print(f"Saved to {os.path.join(args.output_file, 'final_dataset.csv')}")