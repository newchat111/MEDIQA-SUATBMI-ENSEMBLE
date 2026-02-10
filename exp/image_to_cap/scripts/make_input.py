import pandas as pd
import json

#this script generates a json input file ready for VLM inference.
#the input file will make the VLM generate a detailed caption of what is in the given image

aligned_df = pd.read_csv("datasets/mediqa-eval-2026-valid-aligned.csv")

def get_prompt():
    template = """
You are a senior expert in dermatology and wound care. Think silently if needed
I will provide an image that shows either a skin disease or a wound. 
Your task is to give a clear, detailed, and clinically grounded description of what is visible in the image, sufficient for a text-only medical LLM to identify the condition. Keep it short and concise, limit your response length to under 128 tokens
    """

    return template

def generate_input(image_path, key):
    prompt = get_prompt()

    content = []
    content.append({"type": "text", "text": prompt})
    content.append({"type": "image", "image": image_path})

    return {
        "role": "user",
        "content": content,
        "key": key
    }

if __name__ == "__main__":
    save_path = "exp/image_to_cap/inputs/27b_input_short.json"

    aligned_df = aligned_df.drop_duplicates(subset = ['encounter_id'])
    input_json = []

    for i in range(len(aligned_df)):
        image_path = aligned_df['image_path'].iloc[i]
        key = aligned_df['encounter_id'].iloc[i]

        single_input = generate_input(image_path=image_path, key=key)

        input_json.append(single_input)
        
    with open(save_path, 'w') as f:
        json.dump(input_json, f, indent = 2)
