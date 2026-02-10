from utils.eval.eval_json import get_prediction, en_make_submission
from utils.dataset_helper.get_data import METRICS
import pandas as pd

# for m in METRICS:
#     template_path = "datasets/test/mediqa-eval-2026-test-inputonly.csv"
#     pred_path = f"runs/QWEN-all/runs/['{m}']/qwen30b.json"
#     save_path = f"runs/QWEN-all/runs/['{m}']/{m}.csv"

if __name__ == "__main__":
    # for m in METRICS:
    template_path = "datasets/test/mediqa-eval-2026-test-inputonly.csv"
    pred_path = f"runs/QWEN-rag/qwen30b.json"
    save_path = f"runs/QWEN-rag/prediction.csv"
    # metrics = [m]
    # print(metrics)
    _, pred_df = get_prediction(true_path=template_path,
                                prediction_path=pred_path, 
                                metrics = METRICS,
                                in_mark_down=True)
    
    true_df = pd.read_csv(template_path)
    
    submission = en_make_submission(df = pred_df, true_df=true_df)
    submission.to_csv(save_path, index=False)
    
