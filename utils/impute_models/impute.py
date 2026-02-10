from knn import *
import draccus
from dataclasses import dataclass
from utils.impute_models.knn import apply_knn_main
import pandas as pd
import ast

@dataclass
class config:
    input_path: str = ""
    output_path :str = ""
    train_data_path:str = ""
    train_metrics:str = "['completeness','factual-accuracy','relevance','writing-style','overall']"
    pred_metrics:str = "['disagree_flag']"
    model:str = "knn"

@draccus.wrap()
def main(cfg:config):
    #turn the string inputs into lists
    cfg.train_metrics = ast.literal_eval(cfg.train_metrics)
    cfg.pred_metrics = ast.literal_eval(cfg.pred_metrics)

    pred_df = pd.read_csv(cfg.input_path)
    train_df = pd.read_csv(cfg.train_data_path)

    pred_df.rename(mapper={'label':'value'}, axis='columns', inplace=True)
    train_df.rename(mapper={'label':'value'}, axis='columns', inplace=True)

    en_pred_df = pred_df[pred_df['lang'] == 'en']
    zh_pred_df = pred_df[pred_df['lang'] == 'zh']

    if cfg.model == "knn":
        imputed_df = apply_knn_main(train_df=train_df,
                            pred_df = en_pred_df,
                            metrics=cfg.train_metrics,
                            pred_metric_list=cfg.pred_metrics
                            )
    
    imputed_df = pd.concat([en_pred_df, zh_pred_df])
    imputed_df.to_csv(cfg.output_path, index=False)

if __name__ == "__main__":
    main()
    

    


