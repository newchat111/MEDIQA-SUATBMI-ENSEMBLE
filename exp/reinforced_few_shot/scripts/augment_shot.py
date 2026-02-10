from exp.few_shot.scripts.make_shot import *
import pandas as pd

add_shot_num = 50
infer_df_path = "exp/reinforced_few_shot/datasets/infer_complement.csv"
shot_df_path = "datasets/shot_df.csv"
save_path = "exp/reinforced_few_shot/datasets/augmented_shot.csv"

if __name__ == "__main__":

    infer_df = pd.read_csv(infer_df_path)
    shot_df = pd.read_csv(shot_df_path)

    # same logic as grouping()
    exclude_cols = ['metric', 'label', 'query_text', 'gold_texts', 'candidate']
    group_cols = [c for c in infer_df.columns if c not in exclude_cols]

    # get unique groups
    group_keys = infer_df[group_cols].drop_duplicates()

    # randomly sample groups
    sampled_keys = group_keys.sample(n=add_shot_num // 2, random_state=42)

    # select all rows belonging to those groups
    infer_sampled = infer_df.merge(sampled_keys, on=group_cols, how='inner')

    # augment
    augmented_shot = pd.concat([shot_df, infer_sampled], ignore_index=True).drop_duplicates()
    augmented_shot.to_csv(save_path, index=False)



