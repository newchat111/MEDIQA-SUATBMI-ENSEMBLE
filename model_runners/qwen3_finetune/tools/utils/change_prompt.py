import json
from dataclasses import dataclass
import draccus
from tqdm import tqdm

@dataclass
class arguments:
    input_json:str = ''
    prompt_path:str = ''
    output_json:str = ''

@draccus.wrap()
def start_changing(arg:arguments):
    #read the data
    with open(arg.input_json, 'r') as f:
        data = json.load(f)

    #read the prompt
    with open(arg.prompt_path, 'r') as p:
        prompt = p.read()
    
    #change the prompt
    for item in tqdm(data):
        item['conversations'][0]['value'] = "<image>\n" + prompt

    #save the file
    with open(arg.output_json, 'x') as save:
        json.dump(data, save)

if __name__ == "__main__":
    start_changing()

    