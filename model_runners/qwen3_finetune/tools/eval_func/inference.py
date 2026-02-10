from transformers import AutoModelForImageTextToText, AutoProcessor,AutoModelForCausalLM
import draccus
from dataclasses import dataclass
import json
import torch
from tqdm import tqdm
import os

@dataclass
class infer_args:
    model_path: str
    data_path: str
    file_name: str
    batch_size: int
    run_id: str

@draccus.wrap()
def start_inference(inf_args:infer_args):
    #load the data
    DATA_PATH = inf_args.data_path
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)

    model = AutoModelForCausalLM.from_pretrained(
        inf_args.model_path,
        dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map="auto"
)
    processor = AutoProcessor.from_pretrained(inf_args.model_path)
    processor.tokenizer.padding_side = 'left'

    batch_size = inf_args.batch_size
    #prepare the batch
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i : i + batch_size]
        messages = []

        for m in batch:
            messages.append([m])

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True, 
            add_vision_id =True
        )


        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        results = []
        for sample, output in zip(batch, output_text):
            result = {
                "input": sample,          # original item
                "response": output        # model output
            }
            results.append(result)
        # for j, output in enumerate(output_text):
        #     batch[j]["eval"][1]["response"] = output



    file_path = f"{inf_args.file_name}"
    with open(file_path, 'x') as f:
        json.dump(results, f, indent = 2)

if __name__ == "__main__":
    start_inference()

    

