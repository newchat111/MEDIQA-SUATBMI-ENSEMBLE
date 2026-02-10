import shutil
import os
import draccus
from dataclasses import dataclass
import json

@dataclass
class argument:
    json_path: str = ''
    sample_n: int = ''
    save_path:str = ''

@draccus.wrap()
def start_finding_pairs(arg:argument):
    with open(arg.json_path, 'r') as j:
        data = json.load(j)

    identity = 0
    for item in data[0:arg.sample_n]:

        img_source_path = item['image']
        txt = item['conversations'][1]['value']
        
        destination_folder = arg.save_path
        identity_str = str(identity)
        #make the folder 
        folder_path = os.path.join(destination_folder, identity_str)
        os.makedirs(folder_path, exist_ok=False)

        # Option A: Copy to a folder (keeps original filename)
        shutil.copy(img_source_path, os.path.join(folder_path, f'{identity_str}.jpg'))
        #save txt to os.path.join(folder_path, f'{identity}.txt')
        with open(os.path.join(folder_path, f'{identity_str}.txt'),'w', encoding='utf-8') as t:
            t.write(txt)
        identity += 1

if __name__ == "__main__":
    start_finding_pairs()