import torch
from torch.utils.data import Dataset
import json
import random
import pandas as pd
import numpy as np
from PIL import Image
import os

# 定义固定的输出顺序，模型输出的第0位对应completeness，第1位对应factual-accuracy...
# 务必不要改变这个顺序！
METRIC_ORDER = [
    "completeness", 
    "factual-accuracy", 
    "relevance", 
    "disagree_flag", 
    "writing-style", 
    "overall"
]

class BaseMediqaDataset(Dataset):
    def __init__(self, df, tokenizer, transform=None, max_len=512, is_train=False, use_caption=False):
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len
        self.is_train = is_train
        self.use_caption = use_caption
        
        # === 核心逻辑：将长表转换为宽表 ===
        # 我们需要把同一个 (dataset, encounter_id, candidate_author_id) 的多行合并成一个样本
        self.samples = []
        
        # 过滤掉不需要的列以加速分组
        groups = df.groupby(['dataset', 'encounter_id', 'candidate_author_id'])
        
        print(f"[Dataset] Grouping data... Found {len(groups)} unique QA pairs.")
        
        for (dset, enc_id, cand_id), group in groups:
            # 1. 构建标签向量 [6]
            # 初始化为 -1 (标记缺失值，或者用 0.0)
            # 实际竞赛数据通常是完整的，这里默认给 0.0
            label_vec = np.zeros(len(METRIC_ORDER), dtype=np.float32)
            
            # 填充标签
            for _, row in group.iterrows():
                m = str(row['metric'])
                if m in METRIC_ORDER:
                    idx = METRIC_ORDER.index(m)
                    val = float(row['label']) if 'label' in row else 0.0
                    
                    # 在这里统一翻转 disagree_flag
                    if m == 'disagree_flag' and 'label' in row:
                        val = 1.0 - val
                    
                    label_vec[idx] = val
            
            # 2. 提取公共信息 (文本、图片路径等对于同一个组是一样的)
            # 取第一行即可
            first_row = group.iloc[0]
            
            self.samples.append({
                'row_data': first_row,
                'labels': label_vec

            })

    def __len__(self):
        return len(self.samples)
    
    def get_gold_text(self, row):
        gold_json = row.get('gold_texts', '[]')
        try:
            gold_list = json.loads(gold_json)
            if not isinstance(gold_list, list): gold_list = []
        except:
            gold_list = []
        
        if not gold_list:
            return ""

        # === 策略优化 ===
        if self.is_train:
            # 训练时：打乱顺序，然后尽可能拼接，增加多样性
            random.shuffle(gold_list)
            
            # 贪婪拼接：拼到字符串长度大概够了就停
            # 假设 text_a 占 256，我们希望 text_b 也能占 256
            # 粗略估算：1个单词约等于1.3个token，保险起见拼到 1000 字符左右
            selected_str = ""
            for g in gold_list:
                if len(selected_str) + len(g) < 1000: # 这里的阈值可以根据实际情况调
                    selected_str += g + " ; " # 用分号或 [SEP] 隔开
                else:
                    break
            return selected_str.strip(" ; ")
            
        else:
            # 验证/推理时：固定行为
            # 策略 A: 总是选最长的那个 Gold (信息量最大)
            # return max(gold_list, key=len) 
            
            # 策略 B: 把所有 Gold 拼起来 (让模型自己截断) <- 推荐这个，简单有效
            return " ; ".join(gold_list)

# === 多任务文本数据集 ===
class MultiTaskTextDataset(BaseMediqaDataset):
    def __getitem__(self, idx):
        sample = self.samples[idx]
        row = sample['row_data']
        labels_float = sample['labels']
        
        # 提取文本
        candidate = str(row['candidate']) if pd.notna(row.get('candidate')) else ""
        query = str(row['query_text']) if pd.notna(row.get('query_text')) else ""
        gold = self.get_gold_text(row)
        
        caption_text = ""
        if self.use_caption:
            cap = row.get('image_caption', '')
            if pd.notna(cap) and str(cap).strip():
                # === 优化 1: 简单的文本清洗 ===
                clean_cap = str(cap).replace('**', '').replace('\n', ' ').strip()
                # 限制 caption 长度 (比如取前 200 个字符)，防止喧宾夺主
                # clean_cap = clean_cap[:300] 
                caption_text = f" [IMG_INFO]: {clean_cap}" 
        
        # === 优化 2: 调整拼接顺序 ===
        # 策略：Candidate 是主角，必须完整保留。Caption 是辅助。
        # 格式：Query | Candidate [SEP] Image Info
        
        # 这种拼法确保 Query 和 Candidate 在最前面
        text_a = f"Query: {query} | Candidate: {candidate}"
        
        # 把 Caption 拼在后面，或者拼在 text_b (Gold) 的后面
        # 建议拼在 text_a 的末尾，因为它是 Query 的上下文补充
        text_a = text_a + caption_text
        
        text_b = gold
        
        encoding = self.tokenizer(
            text_a, text_b,
            add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        
        labels_class = np.zeros(6, dtype=np.int64)
        
        for i, val in enumerate(labels_float):
            # 简单的区间映射，容忍浮点误差
            if val < 0.25:
                labels_class[i] = 0 # 对应 0.0
            elif val < 0.75:
                labels_class[i] = 1 # 对应 0.5
            else:
                labels_class[i] = 2 # 对应 1.0
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            
            # 训练用：整数类别 (LongTensor)
            'labels': torch.tensor(labels_class, dtype=torch.long),
            
            # 验证用：保留浮点数，方便以后对比
            'labels_float': torch.tensor(labels_float, dtype=torch.float),
            
            'metadata': json.dumps({ 
                'dataset': str(row.get('dataset', '')),
                'encounter_id': str(row.get('encounter_id', '')),
                'candidate_author_id': str(row.get('candidate_author_id', '')),
                'lang': str(row.get('lang', ''))
            })
        }

# === 多任务多模态数据集 (预留) ===
class MultiTaskMultimodalDataset(MultiTaskTextDataset):
    def __getitem__(self, idx):
        # 复用上面的文本逻辑
        data_dict = super().__getitem__(idx)
        
        sample = self.samples[idx]
        row = sample['row_data']
        
        # 图片逻辑
        image_path = str(row.get('image_path', ''))
        pixel_values = torch.zeros((3, 224, 224))
        
        if image_path and os.path.exists(image_path) and self.transform:
            try:
                img = Image.open(image_path).convert('RGB')
                pixel_values = self.transform(img)
            except:
                pass
        
        data_dict['pixel_values'] = pixel_values
        return data_dict