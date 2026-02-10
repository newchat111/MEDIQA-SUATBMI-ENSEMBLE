import pandas as pd

def augment(df:pd.DataFrame, metrics_to_augment:str, dataset_to_augment:str, score_to_augment:float, augment_multiplier:int):
    selected_df = df[(df['metrics'] == metrics_to_augment) & (df['dataset'] == dataset_to_augment) & (df['label'] == score_to_augment)]

    augmented_df = selected_df
    for i in range(augment_multiplier):
        augmented_df = pd.concat([augmented_df, selected_df])
    
    return augmented_df




    

