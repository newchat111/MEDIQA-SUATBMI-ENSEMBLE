import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from utils.dataset_helper.get_data import *
from sklearn.model_selection import train_test_split
from utils.eval.eval_json import *

COLUMNS = ['dataset', 'encounter_id', 'lang', 'candidate', 'candidate_author_id','query_text', 'image_path', 'gold_texts', 'rater_id']

def apply_knn_main(train_df:pd.DataFrame, 
                          pred_df:pd.DataFrame, 
                          metrics:list[str]=['completeness','factual-accuracy','relevance','writing-style','overall'], 
                          pred_metric_list:list[str]=['disagree_flag']):
    
    for pred_metric in pred_metric_list:
        pred_metric = [pred_metric]

        DESIGN = train_df.pivot(index=PER_SYSTEM_KEY, columns="metric", values="value")
        #split the data
        X = DESIGN[metrics]
        y = DESIGN[pred_metric]

        if pred_metric[0] != "disagree_flag":
            score_to_class = {0:int(0), 0.5:int(1), 1:int(2)}

            train_mask = train_df['metric'] == pred_metric[0]
            pred_mask = pred_df['metric'] == pred_metric[0]
            # Apply the mapping only to the 'label' column for those rows
            train_df.loc[train_mask, 'value'] = train_df.loc[train_mask, 'value'].map(score_to_class).astype(int)
            pred_df.loc[pred_mask, 'value'] = pred_df.loc[pred_mask, 'value'].map(score_to_class).astype(int)

        # Initialize the KNN classifier
        knn = KNeighborsClassifier(n_neighbors=5)  

        # Train the model
        knn.fit(X, y.astype(int))
        #inference
        DESIGN_PRED = pred_df.pivot(index=PER_SYSTEM_KEY, columns="metric", values="value")

        X_INFER = DESIGN_PRED[metrics]
        y_infer = knn.predict(X_INFER)

        DESIGN_PRED[pred_metric[0]] = y_infer
        DESIGN_PRED = DESIGN_PRED.reset_index()

        DESIGN_PRED_MELT = DESIGN_PRED.melt(
        id_vars=PER_SYSTEM_KEY,
        value_vars=METRICS,
        var_name='metric',
        value_name='value'
        )

        pred_df = DESIGN_PRED_MELT

    return DESIGN_PRED_MELT.sort_values(by=["encounter_id", "candidate_author_id"])


if __name__ == "__main__":

    save_path = "my_test/knn.json"

    train_df = pd.read_csv("exp/few_shot/datasets/train.csv")
    pred_df = pd.read_csv("exp/few_shot/datasets/val.csv")


    train_df.rename(columns={"label":"value"}, inplace=True)

    pred_df.rename(columns={"label":"value"}, inplace=True)

    imputation = apply_knn(train_df=train_df, 
                                       pred_df = pred_df, 
                                       metrics = ['completeness','writing-style','overall'],
                                       pred_metric_list=['factual-accuracy', 'disagree_flag']
                                       )
    
    imputation.to_csv("my_test/design.csv", index = False)

    merged_df = en_small_sample_merge(true_df = pred_df, pred_df = imputation)
    # merged_df.to_csv("my_test/knn.csv", index = False)

    rename_dict = {"label_x":"value_x", "label_y":"value_y"}

    merged_df.rename(columns=rename_dict, inplace=True)

    scores = {}
    total_score = 0
    for metric in METRICS:

        per_metric_df = merged_df[merged_df['metric'] == metric]
        # print(per_metric_df)
        kendalltau, pearson, spearman, _, _, _ = get_correlations(x = per_metric_df['value_x'], y = per_metric_df['value_y'])

        mean_corr = (kendalltau + pearson + spearman) / 3
        scores[metric] = mean_corr
        total_score += mean_corr

    scores["ALL_en_ALL_mean"] = total_score / len(METRICS)

    with open(save_path, 'w') as f:
        json.dump(scores, f, indent = 2)






