from huggingface_hub import get_token
from transformers import Trainer,AutoModelForSequenceClassification,AutoTokenizer, TrainingArguments
import draccus
from dataclasses import dataclass
from utils.dataset_helper.bert.bert_dataset import prepare_dataset_path_based
from utils.dataset_helper.bert.bert_metric import compute_metrics
from utils.language_models.models import ALL_MODELS
from utils.dataset_helper.bert.bert_dataset import *

import os
import numpy as np
from scipy import stats

def ensemble_majority(arrays):
   
    # stacked = np.stack(arrays, axis=0) 
    classes = np.array([0.5, 0.0, 1.0]) #if tie, use 0.5

    counts = np.stack([
        np.sum(arrays == c, axis=0) for c in classes
    ], axis=0)  # shape (3, ...)

    majority_idx = np.argmax(counts, axis=0)
    return classes[majority_idx]

def ensemble_mean(arrays):
    # stacked = np.stack(arrays, axis=0)  
    mean_preds = np.mean(arrays, axis=0)  # element-wise mean
    return mean_preds

@dataclass
class config:
    model_folder:str = ""
    infer_path:str = ""
    metric:str = ""
    batch_size:int = ""
    output_dir:str = ""
    ensemble_method:str = "vote"

@draccus.wrap()
def main(cfg:config):
    all_preds = []
    #load the data

    for i in range(5):
        model_id = os.path.join(cfg.model_folder, f"{cfg.metric}/model{i}")
        #set up the model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.model_max_length = 512

        dataset = prepare_infer_dataset(infer_data_path=cfg.infer_path,metric=cfg.metric,tokenizer=tokenizer)
        infer_dataset=dataset['infer']
        labels = dataset["infer"].features["labels"].names
        num_labels = len(labels)
        label2id, id2label = dict(), dict()

        for i, label in enumerate(labels):
            label2id[label] = str(i)
            id2label[str(i)] = label

        model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels, label2id=label2id, id2label=id2label)
        args = TrainingArguments(
            output_dir=cfg.output_dir, 
            per_device_eval_batch_size=cfg.batch_size, 
            do_train=False,
            do_eval=False,
            report_to="none")

        trainer = Trainer(model=model, args = args)
        predictions = trainer.predict(infer_dataset)
        
        label_ids = predictions.label_ids
        pred_labels = np.array([float(id2label[str(i)]) for i in label_ids])
        all_preds.append(pred_labels)

    all_preds = np.stack(all_preds, axis=0)
    print(all_preds.shape)
    if cfg.ensemble_method == "vote":
        final_preds = ensemble_majority(all_preds)
    if cfg.ensemble_method == "average":
        final_preds = ensemble_mean(all_preds)

    classes, counts = np.unique(final_preds, return_counts=True)
    print(dict(zip(classes, counts)))
    
    pred_template = pd.read_csv(cfg.infer_path)
    pred_template = pred_template[(pred_template['metric'] == cfg.metric) & (pred_template['lang'] == 'en')]

    pred_template['label'] = final_preds
    pred_template_save_path = os.path.join(cfg.output_dir, f"pred_{cfg.metric}.csv")

    pred_template.rename(columns = {'label': 'value'}, inplace= True)
    pred_template.to_csv(pred_template_save_path, index = False)


if __name__ == "__main__":
    main()

