import json

path = "exp/few_shot/shots/shot4.json"
save_path = "exp/few_shot/shots/shot4_sample.json"

with open(path, 'r') as f:
    shots = json.load(f)

sampled_shots = shots[0:4]

with open(save_path, 'w') as f:
    json.dump(sampled_shots, f, indent=2)
