
from dataclasses import dataclass
import draccus
import json
from tqdm import tqdm
import numpy as np
from rouge import Rouge
from remove_stop import normalize_text
import os

@dataclass
class args:
    input_path:str = ""
    file_name:str = ""
    run_id:str = ""
    eval_method:str = "rougescore"
    output_folder:str = "eval_result"
    sample_n:int = None

@draccus.wrap()
def start_rating(args:args):
    #make a folder for saving the scorings
    folder_path = f"{args.output_folder}/{args.run_id}/scores"

    #read the json file
    with open(args.input_path, 'r') as f:
        input_json = json.load(f)
        if args.sample_n:
            input_json = input_json[0:args.sample_n]

    hyps = []
    refs = []

    for message in tqdm(input_json, desc = "start preparing data for rouge eval"):
        #read the answer and the response
        answer = message['eval'][0]['gpt_answer']
        qwen_response = message['eval'][1]['response']
        #preprocess for bert
        
        hyps.append(normalize_text(qwen_response))
        refs.append(normalize_text(answer))
    
    print("start evaluating...")
    #generate the score
    if args.eval_method == "rougescore":
        rouge = Rouge()
        scores = rouge.get_scores(hyps, refs, avg=True)


    #save the file
    file_path = f"{folder_path}/rouge_{args.file_name}"

    with open(file_path, 'x') as s:
        json.dump(scores, s, indent = 2)

if __name__ == "__main__":
    start_rating()
    
