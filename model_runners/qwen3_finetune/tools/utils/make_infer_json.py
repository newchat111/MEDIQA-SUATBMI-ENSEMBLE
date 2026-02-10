import json
import random
from dataclasses import dataclass
import draccus
import json


@dataclass
class args:
    input_path:str = ""
    output_path:str = ""
    sample_n:int = 200
    prompt:str = "synthesize all available information into a concise, single-sentence academic summary."

@draccus.wrap()
def extract_infer_data(arguments:args):
    INPUT_PATH = arguments.input_path
    OUTPUT_PATH = arguments.output_path

    FIXED_TEXT = arguments.prompt

    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    data = random.sample(data, arguments.sample_n)
    output = []

    for item in data:
        image_path = item["image"]
        ANSWER = item["conversations"][1]["value"]

        output.append({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path
                },
                {
                    "type": "text",
                    "text": FIXED_TEXT
                }
            ],
            "eval": [
                {
                    "gpt_answer": ANSWER
                },
                {
                    "response": ""
                },
                {
                    "score": None
                }
            ]
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"file saved to {OUTPUT_PATH}")

if  __name__ == "__main__":
    extract_infer_data()



