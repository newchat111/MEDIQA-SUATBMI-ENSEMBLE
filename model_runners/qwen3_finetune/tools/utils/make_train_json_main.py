from make_train_json import build_json_data, COMMERCIAL_PATH, NONCOMMERCIAL_PATH, PATH_TO_COM_DF, PATH_TO_NONCOM_DF
import pandas as pd
import json
import draccus
from dataclasses import dataclass

@dataclass
class args:
    input_path:str = "" #file name
    output_path:str = "" #output file name
    prompt_path:str = "" #path to the prompt
    base_path:str = ""
    train_ratio:float = 0.9 #ratio for spliting the data

@draccus.wrap()
def start_make_train_data(arg:args):

    OUTPUT_PATH = arg.output_path

    df = pd.read_csv(arg.input_path)

    train, val = build_json_data(data = df,base_path=arg.base_path,prompt_path=arg.prompt_path,train_ratio= arg.train_ratio)

    train_dataset = []
    train_dataset.extend(train)

    val_dataset = []
    val_dataset.extend(val)

    # Export
    train_output_file = f'{OUTPUT_PATH}_train.json'
    with open(train_output_file, 'w', encoding='utf-8') as f:
        json.dump(train_dataset, f, indent=4, ensure_ascii=False)

    val_output_file = f'{OUTPUT_PATH}_val.json'
    with open(val_output_file, 'w', encoding='utf-8') as f:
        json.dump(val_dataset, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    start_make_train_data()