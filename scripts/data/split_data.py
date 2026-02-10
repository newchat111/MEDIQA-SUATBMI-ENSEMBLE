import pandas as pd
from utils.dataset_helper.get_data import remove_rater_duplicates, PER_OBS_KEY, PER_SYSTEM_KEY
from utils.dataset_helper import datasets_align
from utils.dataset_helper import datasets_cut
import draccus
from dataclasses import dataclass

#datasets\mediqa-eval-2026-valid.csv
#datasets\corrected\mediqa-eval-2026-valid-aligned.csv

@dataclass
class split_arg:
    input_file: str = "datasets/mediqa-eval-2026-valid.csv"
    output_aligned_file: str = "datasets/mediqa-eval-2026-valid-aligned.csv"
    output_folded_file: str = "datasets/mediqa-eval-2026-valid-folded.csv"
    
    PATH_JSON_DERMA: str = "datasets/original_datasets/iiyi/valid_ht.json"
    PATH_IMG_DERMA_ROOT: str = "/data/liyuan/datasets/competition/derma/data/iiyi/images_final/images_valid"
    PATH_JSON_WOUND: str = "datasets/original_datasets/woundcare/valid.json"
    PATH_IMG_WOUND_ROOT: str = "/data/liyuan/datasets/competition/woundcare/dataset-challenge-mediqa-2025-wv/images_final/images_valid"
    
    all_systems_in_same_fold: bool = True
    output_folded:bool = True


@draccus.wrap()
def start_splitting(arg:split_arg):
    orig_data = pd.read_csv(arg.input_file)
    df = remove_rater_duplicates(orig_data)
    #test the df
    #check the length of each row
    print(f"after removing duplicate data, the dataframe has length: {len(df)}. Correct length is 3864")
    #both main function automatically saves the file
    aligned_df = datasets_align.main(df, 
                                     output_file=arg.output_aligned_file,
                                     path_json_derma=arg.PATH_JSON_DERMA, 
                                     path_json_wound=arg.PATH_JSON_WOUND,
                                     path_img_derma=arg.PATH_IMG_DERMA_ROOT,
                                     path_img_wound=arg.PATH_IMG_WOUND_ROOT
                                     )

    if arg.output_folded:
        if arg.all_systems_in_same_fold:
            key = PER_OBS_KEY

        else:
            key = PER_SYSTEM_KEY
        folded_df = datasets_cut.main(aligned_df, keys = key, output_file=arg.output_folded_file)

    # Group by the keys and count distinct folds per group
    fold_per_group = aligned_df.groupby(key)['fold'].nunique()

    # Find any groups that appear in more than one fold
    inconsistent_groups = fold_per_group[fold_per_group > 1]

    if inconsistent_groups.empty:
        print("All observations with the same key are in the same fold.")
    else:
        print("Found groups split across multiple folds!")
        print(f"Number of problematic groups: {len(inconsistent_groups)}")
        # Optional: show examples
        print(inconsistent_groups.head())

    print(f"size of the folds{folded_df.groupby('fold').size()}")



if __name__ == "__main__":
    start_splitting()

    