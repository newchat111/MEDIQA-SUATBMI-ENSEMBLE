import json

path = "exp/image_to_cap/inputs/input.json"
with open(path, 'r') as f:
    input = json.load(f)

print(len(input))