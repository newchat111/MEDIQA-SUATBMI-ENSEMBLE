import pandas as pd
import numpy as np
import torch

from sklearn.metrics import accuracy_score, f1_score
# from dataset_bert import Dataset
from datasets import DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)
import os
import gc
import json
import datetime
import argparse
import shutil 

# ================= 1. 全局配置 =================
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
METRIC_LIST = [
    "overall", "completeness", "factual-accuracy", 
    "relevance", "disagree_flag", "writing-style"
]

DEBUG_MODE = False
MAX_LEN = 512
BATCH_SIZE = 32
LR = 2e-5
EPOCHS = 5 if not DEBUG_MODE else 1

# ================= 2. 策略定义  =================
def format_cand_only(row):
    """[CANDIDATE] """
    candidate = str(row.get('candidate', ''))
    return f"[CANDIDATE] {candidate}"

def format_context_first(row):
    """[IMG] [QUERY] [CANDIDATE] """
    candidate = str(row.get('candidate', ''))
    query = str(row.get('query_text', ''))
    caption = str(row.get('image_caption', ''))
    if len(caption) > 300: caption = caption[:300]
    return f"[IMG] {caption} [QUERY] {query} [CANDIDATE] {candidate}"

def format_query_cand(row):
    """[QUERY] [CANDIDATE] """
    candidate = str(row.get('candidate', ''))
    query = str(row.get('query_text', ''))
    return f"[QUERY] {query} [CANDIDATE] {candidate}"

# ---  策略路由 ---
def get_router_fn(iiyi_strategy, wound_strategy):
    def dynamic_router(row):
        ds = str(row.get('dataset', '')).lower()
        if 'iiyi' in ds:
            return iiyi_strategy(row)
        else:
            return wound_strategy(row)
    
    dynamic_router.__name__ = f"Router[iiyi={iiyi_strategy.__name__}, wound={wound_strategy.__name__}]"
    return dynamic_router

# ---  策略配置仓库 ---
def get_strategy_config(strategy_type):
    if strategy_type == "dataset_specific":
        return {
            "completeness": format_cand_only,
            "disagree_flag": get_router_fn(
                iiyi_strategy=format_query_cand, 
                wound_strategy=format_context_first
            ),
            "factual-accuracy": get_router_fn(
                iiyi_strategy=format_context_first, 
                wound_strategy=format_query_cand
            ),
            "relevance": format_context_first,
            "writing-style": get_router_fn(
                iiyi_strategy=format_context_first, 
                wound_strategy=format_cand_only
            ),
            "overall": format_cand_only
        }
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

# ================= 3. 数据处理函数  =================
def load_and_process_df(path, target_metric, fold, formatting_fn, is_train=True):
    df = pd.read_csv(path)
    df = df[df['metric'] == target_metric]
    
    # 折数拆分
    if is_train:
        df = df[df['fold'] != fold]
    else:
        df = df[df['fold'] == fold]
    
    # 标签映射
    def map_label(val):
        try: val = float(val)
        except: val = 0.0
        if val < 0.25: return 0
        elif val < 0.75: return 1
        else: return 2
    
    df['labels'] = df['label'].apply(map_label)
    
    # 自适应数据平衡
    
    if is_train and len(df) > 0:
        balanced_dfs = []
        
        # 按数据集分组 (iiyi 和 woundcare 分开看)
        for ds_name, ds_df in df.groupby('dataset'):
            class_counts = ds_df['labels'].value_counts()
            
            # 只有当存在多类且不平衡严重时才干预
            if len(class_counts) > 1:
                max_count = class_counts.max()
                min_count = class_counts.min()
                
                # 计算不平衡比率
                ratio = max_count / (min_count + 1e-9)
                IMBALANCE_THRESHOLD = 7.0 # 超过 7:1 平衡
                
                if ratio > IMBALANCE_THRESHOLD:
                    print(f"[{ds_name}] Imbalanced (Ratio {ratio:.1f}:1). From {min_count} balancing to {max_count}...")
                    
                    # 遍历该数据集内的所有类别
                    ds_balanced = []
                    for label_val, count in class_counts.items():
                        sub_df = ds_df[ds_df['labels'] == label_val]
                        ds_balanced.append(sub_df)
                        
                        if count < max_count:
                            n_gap = max_count - count
                            aug_part = sub_df.sample(n=n_gap, replace=True, random_state=42)
                            ds_balanced.append(aug_part)
                            
                    # 合并平衡结果
                    ds_df = pd.concat(ds_balanced)
                else:
                    # 比较平衡则保持原样
                    pass 
            
            balanced_dfs.append(ds_df)
            
        # 重新组合数据集
        df = pd.concat(balanced_dfs, axis=0)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Training Set Processed. Total Size: {len(df)}")

    # 文本格式化
    df['text'] = df.apply(formatting_fn, axis=1)
    
    df = df.reset_index(drop=True)
    return df[['text', 'labels', 'label', 'dataset', 'encounter_id', 'candidate_author_id']]

def compute_metrics(eval_pred):
    logits, label_ids = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(label_ids, predictions),
        "f1_macro": f1_score(label_ids, predictions, average="macro")
    }

# ================= 4. 日志 =================
def save_strategy_log(strategy_config, exp_name, output_root):
    readable_strategy = {metric: func.__name__ for metric, func in strategy_config.items()}
    log_data = {
        "experiment_name": exp_name,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hyperparameters": {
            "lr": LR, "batch_size": BATCH_SIZE, "epochs": EPOCHS, "model": MODEL_NAME
        },
        "strategy_map": readable_strategy
    }
    with open(os.path.join(output_root, "strategy_config.json"), "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4, ensure_ascii=False)
    print(f"策略配置文件已保存至: {os.path.join(output_root, 'strategy_config.json')}")

# ================= 5. 主流程 =================
def main():
    set_seed(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy_type", type=str, required=True, help="策略类型名称")
    parser.add_argument("--exp_suffix", type=str, default="", help="实验名称后缀")
    parser.add_argument("--input_path", type=str, required=True, help="训练数据路径")
    parser.add_argument("--template_path", type=str, required=True, help="模板路径")
    parser.add_argument("--output_path", type=str, required=True, help="输出路径")
    args = parser.parse_args()
    
    global EXP_NAME, OUTPUT_ROOT
    EXP_NAME = f"pubmedbert_{args.strategy_type}{args.exp_suffix}"
    OUTPUT_ROOT = os.path.join(args.output_path, EXP_NAME)
    
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    STRATEGY_CONFIG = get_strategy_config(args.strategy_type)
    
    print(f"Starting Experiment: {EXP_NAME}")
    print(f"Strategy: {args.strategy_type} | Mode: Joint Training (Adaptive)")
    
    save_strategy_log(STRATEGY_CONFIG, EXP_NAME, OUTPUT_ROOT)
        
    global_predictions = {}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    for metric in METRIC_LIST:
        print(f"\n{'='*40}")
        print(f"Processing Metric: {metric}")
        
        current_strategy_fn = STRATEGY_CONFIG.get(metric, format_context_first)
        print(f"Strategy used: {current_strategy_fn.__name__}")
        
        folds_to_run = range(5) if not DEBUG_MODE else [0]
        
        for fold in folds_to_run:
            print(f"\nRunning Fold {fold}/5 for {metric}...")
            
            # 加载训练集
            train_df = load_and_process_df(args.input_path, target_metric=metric, fold=fold, 
                                           formatting_fn=current_strategy_fn, is_train=True)
            
            if train_df.empty: continue

            # 加载验证集
            val_df = load_and_process_df(args.input_path, target_metric=metric, fold=fold, 
                                         formatting_fn=current_strategy_fn, is_train=False)
            
            if val_df.empty: continue

            dataset = DatasetDict({
                "train": Dataset.from_pandas(train_df, preserve_index=False),
                "validation": Dataset.from_pandas(val_df, preserve_index=False),
            })
            
            def tokenize_fn(examples):
                return tokenizer(examples["text"], truncation=True, max_length=MAX_LEN)
            
            tokenized_datasets = dataset.map(
                tokenize_fn, batched=True, remove_columns=["text", "label"] 
            )

            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=3, problem_type="single_label_classification"
            )
            
            fold_output_dir = os.path.join(OUTPUT_ROOT, metric, f"fold_{fold}")
            
            args_train = TrainingArguments(
                output_dir=fold_output_dir,
                eval_strategy="epoch",
                save_strategy="epoch",
                learning_rate=LR,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                num_train_epochs=EPOCHS,
                weight_decay=0.01,
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                greater_is_better=True,
                save_total_limit=1,
                fp16=True,
                report_to="none"
            )
            
            trainer = Trainer(
                model=model,
                args=args_train,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                compute_metrics=compute_metrics,
            )
            
            trainer.train()
            
            preds_output = trainer.predict(tokenized_datasets["validation"])
            pred_classes = np.argmax(preds_output.predictions, axis=-1)
            idx2score = {0: 0.0, 1: 0.5, 2: 1.0}
            
            for i, row in val_df.iterrows():
                dset = str(row['dataset'])
                enc_id = str(row['encounter_id'])
                cand_id = str(row['candidate_author_id'])
                score = idx2score[pred_classes[i]]
                global_predictions[(dset, enc_id, cand_id, metric)] = score

            # 清理
            del model, trainer
            torch.cuda.empty_cache()
            gc.collect()
            if os.path.exists(fold_output_dir):
                shutil.rmtree(fold_output_dir)

    # ================= 6. 生成 Prediction CSV =================
    print("\n>>> Generating Prediction CSV...")
    if os.path.exists(args.template_path):
        df_official = pd.read_csv(args.template_path)
        
        new_values = []
        hit_count = 0
        
        for _, row in df_official.iterrows():
            key = (str(row['dataset']), str(row['encounter_id']), str(row['candidate_author_id']), str(row['metric']))
            if key in global_predictions:
                new_values.append(global_predictions[key])
                hit_count += 1
            else:
                new_values.append(0.0)
                
        df_official['value'] = new_values
        df_official['rater_id'] = EXP_NAME
        
        save_path = os.path.join(OUTPUT_ROOT, "prediction.csv")
        df_official.to_csv(save_path, index=False)
        print(f"Submission saved to: {save_path}")
        print(f"Matched {hit_count} rows.")
    
if __name__ == "__main__":
    main()