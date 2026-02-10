from exp.finetune.scripts.setup.prepare_infer_input import prepare_infer_input
from exp.finetune.scripts.setup.prepare_sft_input import prepare_ft_input
import draccus
from dataclasses import dataclass
import ast
import pandas as pd
import json

@dataclass
class input_cfg:
    train_input_path:str = ""
    val_input_path:str = ""
    output_folder:str = ""
    metrics:str = "['overall']"

@draccus.wrap()
def main(icfg:input_cfg):
    train_df = pd.read_csv(icfg.train_input_path)
    val_df = pd.read_csv(icfg.val_input_path)

    metrics = ast.literal_eval(icfg.metrics)

    train_sft = prepare_ft_input(train_df=train_df,metrics=metrics)
    val_sft = prepare_ft_input(train_df=val_df, metrics = metrics)

    val_infer = prepare_infer_input(infer_df=val_df, metrics=metrics)

    train_path = f"{icfg.output_folder}/train.json"
    val_path = f"{icfg.output_folder}/val.json"
    infer_path = f"{icfg.output_folder}/infer.json"

<<<<<<< HEAD
    with open(train_path, 'x') as f:
        json.dump(train_sft, f, indent=2)

    with open(val_path, 'x') as f:
        json.dump(val_sft, f,indent=2)

    with open(infer_path, 'x') as f:
=======
    with open(train_path, 'w') as f:
        json.dump(train_sft, f, indent=2)

    with open(val_path, 'w') as f:
        json.dump(val_sft, f,indent=2)

    with open(infer_path, 'w') as f:
>>>>>>> master
        json.dump(val_infer, f,indent=2)

if __name__ == "__main__":
    main()


    

