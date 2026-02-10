# MediQA-SUATSMI

Right now, the code contains utility scripts only. 

To use the code, simply do

```
git clone git@8.129.227.88:LiyuanE/mediqa-competition.git
```

## A brief introduction to using the utility scripts

The utilities help you retrieve the data easily and efficiently. As of now, it has two major parts:1.dataset_helper 2.evaluation 3.visualize

### dataset_helper

1. grouping(df:pd.DataFrame, group_type:str = ['metric']) -> pd.DataFrame:

    usage:groups the dataframe by columns except group_type + ['label']

```python
#if you want to get a grouped dataset where each group contains all the metrics
import pandas as pd
from utils.dataset_helper.get_data import grouping

group_list = []

path_to_csv = "my_test/get_data_label_pair.csv"
df = pd.read_csv(path_to_csv)
grouped_df = grouping(df, group_type = ['metric'])

for key_vals, group in grouped_df:
    group_list.append(group)

print(group_list)
```

2. get_train_valid(valid_id) -> pd.DataFrame

    usage:it splits the data to a training set and a validation set based on the validation id, which ranges from 0 to 4 because our data has 5 folds

```python
#returns two pandas dataframe: train and val. train contains 80% of the total data, and val has the rest of the 20%
#it automatically reads  datasets/datasets_5folds.csv
import pandas as pd
from utils.dataset_helper.get_data import get_train_valid

#the data is splitted into 5 folds, (0,1,2,3,4)
#train is (0,1,2,4)
#val is (3)
val_id = 3

train, val = get_train_valid(valid_id=val_id)
```

3. get_data(raters:list[str] = None,lang:str = None, system:int = 0, sample_n:int = None, metrics:list[str] = ['writing-style', 'overall'], seed = 114514)

    usage: it has various arguments that returns the desired dataframe

```python
#to get the data rated by rater 'NM' and sample 12 from them
#returns two datasets: train and val
import pandas as pd
from utils.dataset_helper.get_data import retrieve_data

df, _ = retrieve_data(raters=['NM'],sample_n=12)
```










