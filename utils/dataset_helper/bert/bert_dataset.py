from datasets import Dataset, DatasetDict
from utils.dataset_helper.get_data import get_train_valid, FOLDED_DATA
from transformers import AutoTokenizer
import pandas as pd

def renamer(df:pd.DataFrame, map:dict):
    return df.rename(columns = map)

def shrink_to_one_metric(df:pd.DataFrame, metric:str, lang:str):
    selected_df = df[(df['metric'] == metric) & (df['lang'] == lang)]
    print(len(selected_df))
    return selected_df

def prepare_infer_dataset(infer_data_path:str, metric:str, tokenizer):
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, return_tensors="pt")
    
    infer = pd.read_csv(infer_data_path)
    infer['candidate'] = infer['candidate'].fillna('')
    # na_rows_raw = infer[infer.isna().any(axis=1)]
    
    name_map = {'label': 'labels', 'candidate':'text'}

    infer = shrink_to_one_metric(infer, metric = metric, lang = 'en')
    infer = renamer(infer, map = name_map)
    infer = infer.dropna(subset=['text'])

    # train_dataset = Dataset.from_pandas(train)
    # val_dataset = Dataset.from_pandas(val)
    infer_dataset = Dataset.from_pandas(infer)
    #construct a dict
    dataset = DatasetDict({
        "infer":infer_dataset
    })

    for i in range(3):
        print(f"Example {i}: {dataset['infer'][i]['text']}")

    tokenized_dataset = dataset.map(tokenize, batched=True,remove_columns=["text"]).class_encode_column("labels")
    return tokenized_dataset

def prepare_dataset_path_based(train_data_path:str, val_data_path:str,metric:str, include_query:bool, tokenizer):
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, return_tensors="pt")
    
    train = pd.read_csv(train_data_path)
    val = pd.read_csv(val_data_path)
    if include_query:
        train['candidate_query'] = ('response:' + train['candidate'] + 'query:' + train['query_text'])
        val['candidate_query'] = ('response:' + val['candidate'] + 'query:' + val['query_text'])
        name_map = {'label': 'labels', 'candidate_query':'text'}
    else:
        name_map = {'label': 'labels', 'candidate':'text'}

    train = shrink_to_one_metric(train, metric = metric, lang = 'en')
    val = shrink_to_one_metric(val, metric = metric, lang = 'en')
    
    train = renamer(train, map = name_map)
    val = renamer(val, map = name_map)

    train = train.dropna(subset=['text'])
    val = val.dropna(subset=['text'])

    train_dataset = Dataset.from_pandas(train)
    val_dataset = Dataset.from_pandas(val)
    #construct a dict
    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset, 
    })

    for i in range(3):
        print(f"Example {i}: {dataset['validation'][i]['text']}")

    tokenized_dataset = dataset.map(tokenize, batched=True,remove_columns=["text"]).class_encode_column("labels")
    return tokenized_dataset

def prepare_dataset(data_path:str, fold:int, metric:str, include_query:bool, tokenizer):

    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, return_tensors="pt")
    
    train,val = get_train_valid(data_path=data_path,valid_id=[fold])

    if include_query:
        train['candidate_query'] = ('response:' + train['candidate'] + 'query:' + train['query_text'])
        val['candidate_query'] = ('response:' + val['candidate'] + 'query:' + val['query_text'])
        name_map = {'label': 'labels', 'candidate_query':'text'}
    else:
        name_map = {'label': 'labels', 'candidate':'text'}

    train = shrink_to_one_metric(train, metric = metric, lang = 'en')
    val = shrink_to_one_metric(val, metric = metric, lang = 'en')
    
    train = renamer(train, map = name_map)
    val = renamer(val, map = name_map)

    train = train.dropna(subset=['text'])
    val = val.dropna(subset=['text'])

    train_dataset = Dataset.from_pandas(train)
    val_dataset = Dataset.from_pandas(val)
    #construct a dict
    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset, 
    })

    for i in range(3):
        print(f"Example {i}: {dataset['validation'][i]['text']}")

    tokenized_dataset = dataset.map(tokenize, batched=True,remove_columns=["text"]).class_encode_column("labels")
    return tokenized_dataset

if __name__ == "__main__":
    model_id = "/data1/xinzhe/microsoft_nlp/mediqa-competition/models/bert/albert"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.model_max_length = 512
    
    data_path = FOLDED_DATA
    fold = 0
    metric = 'writing-style'

    dataset = prepare_dataset(data_path=data_path, fold=fold, metric=metric, tokenizer = tokenizer, include_query=True)
    labels = dataset["train"].features["labels"].names

    num_labels = len(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    print(label2id)
    print(id2label)



    
