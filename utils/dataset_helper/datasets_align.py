import pandas as pd
import json
import os
from tqdm import tqdm

# ================= 1. 路径配置 =================
# PATH_CSV = "/data/liyuan/datasets/competition/mediqa-eval-2026-valid.csv"
# PATH_JSON_DERMA = "datasets/original_datasets/iiyi/valid_ht.json"
# PATH_IMG_DERMA_ROOT = "/data/liyuan/datasets/competition/derma/data/iiyi/images_final/images_valid"
# PATH_JSON_WOUND = "datasets/original_datasets/woundcare/valid.json"
# PATH_IMG_WOUND_ROOT = "/data/liyuan/datasets/competition/woundcare/dataset-challenge-mediqa-2025-wv/images_final/images_valid"
# OUTPUT_FILE = "/data1/xinzhe/microsoft_nlp/workspace/mediqa-competition/datasets/my_aligned.csv"
# =================================================

def clean_text(text):

    if text is None:
        return ""
    text = str(text)
    return text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()

def load_json_db(path):
    print(f"[INFO] Loading JSON from: {path}")
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {item['encounter_id']: item for item in data}

def get_gold_data(json_item, lang):
    """
    提取标准答案信息。
    返回:
    1. primary_gold: 列表中的第一个回答 (字符串)
    2. all_golds_json: 所有回答的列表 (JSON 字符串)
    """
    responses = json_item.get('responses', [])
    

    if not responses:
        return "", "[]"
    

    all_gold_list = []
    for r in responses:
        content = r.get(f"content_{lang}")
        if content:

            cleaned_content = clean_text(content)
            if cleaned_content: # 确保不存空字符串
                all_gold_list.append(cleaned_content)
    all_golds_json = json.dumps(all_gold_list, ensure_ascii=False)
    
    return all_golds_json

def find_image_path(json_item, root_dir):
    image_ids = json_item.get('image_ids', [])
    if not image_ids:
        return None
    # img_filename = image_ids[0]
    # print(image_ids)
    full_path = [os.path.join(root_dir, img_id) for img_id in image_ids]
    # print(root_dir)
    # print(image_ids)

    return full_path

def main(df, output_file, path_json_derma, path_json_wound, path_img_derma, path_img_wound) -> pd.DataFrame:

    db_derma = load_json_db(path_json_derma)
    db_wound = load_json_db(path_json_wound)
    df_raw = df
    print(f"[INFO] Processing {len(df_raw)} rows...")

    aligned_data = []
    stats = {"success": 0, "missing_img": 0, "missing_json": 0}

    for _, row in tqdm(df_raw.iterrows(), total=len(df_raw)):
        dataset = row['dataset']
        enc_id = row['encounter_id']
        lang = row['lang']
        candidate_author_id = row['candidate_author_id']
        rater = row['rater_id']
    
        if dataset == 'iiyi':
            db = db_derma
            img_root = path_img_derma
        elif dataset == 'woundcare':
            db = db_wound
            img_root = path_img_wound
        else:
            continue
            
        if enc_id not in db:
            stats["missing_json"] += 1
            continue
            
        item_data = db[enc_id]
        # print(item_data)
        # Query
        q_title = item_data.get(f"query_title_{lang}", "")
        q_content = item_data.get(f"query_content_{lang}", "")
        query_text = clean_text(f"{q_title or ''} {q_content or ''}")
        
        all_golds_json = get_gold_data(item_data, lang)
        
        # Candidate
        candidate_text = clean_text(row['candidate'])
        
        # Image
        img_path = find_image_path(item_data, img_root)
        print(img_path)
        print(enc_id)
        if img_path is None:
            stats["missing_img"] += 1
        
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
            "gold_texts": all_golds_json,
            "rater_id": rater
        }
        aligned_data.append(new_row)
        stats["success"] += 1

    df_aligned = pd.DataFrame(aligned_data)
    df_aligned.to_csv(output_file, index=False, encoding='utf-8-sig')

    return df_aligned

if __name__ == "__main__":
    main()