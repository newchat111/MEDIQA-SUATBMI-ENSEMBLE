import pandas as pd

PATH_TO_FOLDER = "/workspace/xinzhe"

commercial_df = pd.read_csv(f"{PATH_TO_FOLDER}/data/gemma3_summary_commercial.csv")
noncommercial_df = pd.read_csv(f"{PATH_TO_FOLDER}/data/gemma3_summary_noncommercial.csv")

df = pd.concat([commercial_df, noncommercial_df], ignore_index=True)
df.to_csv(f"{PATH_TO_FOLDER}/data/gemma3_all.csv", index = False)