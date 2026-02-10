from typing import Any
from datasets import Dataset, DatasetDict
import json
import json
from datasets import Dataset, Features, Value, Image as HFImage

def transform_to_model_format(json_data):
    # 1. Define the schema upfront (identical to before)
    features = Features({
        "image": HFImage(),
        "metadata_key": Value("string"),
        "messages": [
            {
                "role": Value("string"),
                "content": [
                    {"text": Value("string"), "type": Value("string")}
                ]
            }
        ]
    })

    # 2. Use a Generator to yield rows one by one
    # This prevents the creation of massive intermediate lists
    def gen():
        for entry in json_data:
            # Pre-extract variables to avoid deep lookups in the dict
            msg_list = entry["messages"]
            
            yield {
                "image": msg_list[0]["content"][0]["image"],
                "metadata_key": str(entry["key"][0]),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": None, "type": "image"},
                            {"text": msg_list[0]["content"][1]["text"], "type": "text"}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"text": msg_list[1]["content"][0]["text"], "type": "text"}
                        ]
                    }
                ]
            }

    # 3. Load using the generator
    # This is significantly more memory-efficient and faster for large lists
    return Dataset.from_generator(gen, features=features)

# def transform_to_model_format(json_data):
#     processed_rows = {
#         "image": [],
#         "messages": [],
#         "metadata_key": [] 
#     }

#     for entry in json_data:
#         img_path = entry["messages"][0]["content"][0]["image"]
#         key_identifier = str(entry["key"][0]) 
        
#         # Get text from your specific JSON structure
#         user_text = entry["messages"][0]["content"][1]["text"]
#         assistant_text = entry["messages"][1]["content"][0]["text"]
        
#         # This matches the specific interleaved format you requested
#         formatted_messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"text": None, "type": "image"},
#                     {"text": user_text, "type": "text"}
#                 ]
#             },
#             {
#                 "role": "assistant",
#                 "content": [
#                     {"text": assistant_text, "type": "text"}
#                 ]
#             }
#         ]

#         processed_rows["image"].append(img_path)
#         processed_rows["messages"].append(formatted_messages)
#         processed_rows["metadata_key"].append(key_identifier)

#     features = Features({
#         "image": HFImage(),
#         "metadata_key": Value("string"),
#         "messages": [
#             {
#                 "role": Value("string"),
#                 "content": [
#                     {"text": Value("string"), "type": Value("string")}
#                 ]
#             }
#         ]
#     })

#     return Dataset.from_dict(processed_rows, features=features)

def collate_fn(processor, examples: list[dict[str, Any]]):
    texts = []
    images = []
    for example in examples:
        images.append([example["image"].convert("RGB")])
        texts.append(processor.apply_chat_template(
            example["messages"], add_generation_prompt=False, tokenize=False
        ).strip())

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # The labels are the input_ids, with the padding and image tokens masked in
    # the loss computation
    labels = batch["input_ids"].clone()

    # Mask image tokens
    image_token_id = [
        processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.special_tokens_map["boi_token"]
        )
    ]
    # Mask tokens that are not used in the loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100

    batch["labels"] = labels
    return batch