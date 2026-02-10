import json

path = "exp/image_to_cap/outputs/medgem27b_caption_512.json"

with open(path, 'r') as f:
    output = json.load(f)

for o in output[0:5]:
    response = o["response"][0]
    print(response)