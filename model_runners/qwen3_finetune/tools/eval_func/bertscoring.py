from bert_score import score
from dataclasses import dataclass
import draccus
import json
from tqdm import tqdm
import numpy as np
from rouge import Rouge 

@dataclass
class args:
    input_path:str = ""
    file_name:str = ""
    run_id:str = ""
    eval_method:str = "bertscore"
    output_folder:str = "eval_result"
    sample_n:int = None
    batch_size:int = 32

@draccus.wrap()
def start_rating(args:args):
    #make the folder for saving the scores
    folder_path = f"{args.output_folder}/{args.run_id}/scores"

    #read the json file
    with open(args.input_path, 'r') as f:
        input_json = json.load(f)
        if args.sample_n:
            input_json = input_json[0:args.sample_n]

    refs = [message['eval'][0]['gpt_answer'] for message in input_json]
    hyps = [message['eval'][1]['response'] for message in input_json]

    if args.eval_method == "bertscore":
        P, R, F1 = score(
            hyps,
            refs,
            lang='en',
            device="cuda:0",
            batch_size=args.batch_size,
            verbose=True
        )

    total_f1_score = 0.0

    for idx, msg in enumerate(input_json):
        f1_value = F1[idx].item()
        total_f1_score += f1_value
        if len(msg['eval']) > 2:
            msg['eval'][2]['score'] = f1_value

    avg_f1_score = total_f1_score / len(input_json)
    file_path = f"{folder_path}/bert_{args.file_name}"

    with open(file_path, "w") as f:
        json.dump(
            {"avg_bert_f1": avg_f1_score},
            f,
            indent=2
        )
    # avg_f1_score = total_f1_score / len(input_json)

    # # save average score
    # with open(file_path, "w") as f:
    #     json.dump({"avg_bert_f1": avg_f1_score}, f, indent=2)


    # for message in tqdm(input_json):
    #     #read the answer and the response
    #     answer = message['eval'][0]['gpt_answer']
    #     qwen_response = message['eval'][1]['response']
    #     #preprocess for bert
    #     answer = [answer]
    #     qwen_response = [qwen_response]
    #     #generate the score
    #     if args.eval_method == "bertscore":
    #         P, R, F1 = score(qwen_response, answer, lang='en', verbose=True, device="cuda:0")
            
    #     message['eval'][2]['score'] = F1[0].item()
    #     total_f1_scores += F1[0].item()
    
    # # #save the file
    # # with open(args.output_path, 'x') as s:
    # #     json.dump(input_json, s, indent = 2)

if __name__ == "__main__":
    start_rating()
    
