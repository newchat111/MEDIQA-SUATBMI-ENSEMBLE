from exp.finetune.scripts.setup.prepare_sft_input import pd,grouping, make_identity, get_sft_prompt
import json
def infer_per_entry_temp(image_path:str, output:str, metrics:list[str],key:dict, llm_response, query):
    entry_temp = {}
    entry_temp={
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path
            },
            {
                "type": "text",
                "text": get_sft_prompt(metrics=metrics, llm_response=llm_response, query=query)

            },
        ],
    }
    

    entry_temp["key"] = key
    
    return entry_temp

<<<<<<< HEAD
def prepare_infer_input(infer_df:pd.DataFrame, metrics:list[str]) -> list[str]:
    #right now it only supports 1 metric
    input = []
=======
def prepare_infer_input(infer_df:pd.DataFrame, metrics:list[str], en_only:bool = True) -> list[str]:
    #right now it only supports 1 metric
    input = []

    if en_only:
        infer_df = infer_df[infer_df['lang'] == 'en']
>>>>>>> master
    infer_df = grouping(df = infer_df, group_type=['metric'])
    metrics = metrics

    for key_val, group in infer_df:
        key = make_identity(group)
<<<<<<< HEAD

        label = str(group[group['metric'] == metrics[0]]['label'].iloc[0])
=======
        # label = group[group['metric'] == metrics[0]]['label']
        # print(label)
        label = group[group['metric'] == metrics[0]]['label'].iloc[0]
>>>>>>> master
        entry = infer_per_entry_temp(image_path=group['image_path'].iloc[0],
                               output=label, 
                               key = key, 
                               metrics=metrics,
                               llm_response=group['candidate'].iloc[0],
                               query = group['query_text'].iloc[0])
        
        input.append(entry)

    return input

if __name__ == "__main__":
    metrics = ['overall']

    train_df = pd.read_csv("exp/few_shot/datasets/train.csv")
    val_df = pd.read_csv("exp/few_shot/datasets/val.csv")

    infer_val_output_path = "exp/finetune/data/new_temp/val_infer_input.json"
    
    val_infer = prepare_infer_input(infer_df=val_df, metrics=metrics)

    with open(infer_val_output_path, 'w') as f:
        json.dump(val_infer, f, indent=2)
    



