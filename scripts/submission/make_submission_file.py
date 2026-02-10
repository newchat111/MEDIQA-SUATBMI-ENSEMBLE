from utils.eval.eval_json import get_prediction, en_make_submission
from utils.dataset_helper.get_data import METRICS
import pandas as pd
import draccus
from dataclasses import dataclass
import ast

@dataclass
class args:
    pred_path:str
    save_path:str
    # template_path:str
    # final_save_path:str
    metrics:str
    en_only:bool = True

@draccus.wrap()
def main(args:args):
    # final = []
    # final_save_path = args.final_save_path
    # true_df = pd.read_csv(args.template_path)
    metrics = ast.literal_eval(args.metrics)
    # for m in METRICS:
    template_path = "datasets/test/mediqa-eval-2026-test-inputonly.csv"
    pred_path = args.pred_path
    save_path = args.save_path



    _, pred_df = get_prediction(true_path=template_path,
                                prediction_path=pred_path, 
                                metrics = metrics,
                                in_mark_down=True)
    
    true_df = pd.read_csv(template_path)
    
    if args.en_only:
        submission = pred_df
        submission.to_csv(save_path, index=False)
    else:
        submission = en_make_submission(df = pred_df, true_df=true_df)
        submission.to_csv(save_path, index=False)

        # final.append(pred_df)
    
    # final_df = pd.concat(final)
    # final_submission = en_make_submission(df = final_df, true_df=true_df)
    # final_submission.to_csv(args.final_save_path, index=False)

if __name__ == "__main__":
    main()
    
