import pandas as pd
from utils.dataset_helper.get_data import *
from utils.eval.eval_json import en_make_submission
import numpy as np



# def ensemb_model_outputs(bert_out:pd.DataFrame, gemma_out:pd.DataFrame, bert_selection:list[str], gemma_selection:list[str]):
#     bert_out_selected = bert_out[bert_out['metric'].isin(bert_selection)]
#     gemma_out_selected = gemma_out[gemma_out['metric'].isin(gemma_selection)]

#     return pd.concat([bert_out_selected, gemma_out_selected])

# def ensemb_by_vote(df:pd.DataFrame):
#     cols = KEYS
#     counts = df.groupby(cols).transform("size") 
#     result = ( df.assign(_count=counts) .sort_values("_count", ascending=False) .drop_duplicates(subset=["key"]) .drop(columns="_count") )
#     return result


def ensemb(model_outputs:list, selection:list, all_pred_in_en:bool, template_path:str):
    idx = 0
    final_outputs = []
    template = pd.read_csv(template_path)

    for idx in range(len(model_outputs)):
        df = model_outputs[idx]
        if all_pred_in_en:
            df = df[df['lang'] == 'en']

        df_out_selected = df[df['metric'].isin(selection[idx])]
        print(selection[idx])
        final_outputs.append(df_out_selected)

    if all_pred_in_en:
        final_df = pd.concat(final_outputs)
        final_submission = en_make_submission(df = final_df, true_df = template)
        return final_submission
    else:
        final_df = pd.concat(final_outputs)
        return final_df
    


if __name__ == "__main__":
    df1 = pd.read_csv()