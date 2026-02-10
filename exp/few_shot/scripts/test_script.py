from make_shot import *
metrics=["overall","relevance", "disagree_flag"]

# print(prompt_guidance(metrics=metrics))
# print(prompt_outputTemp(metrics=metrics))
print(get_prompt_template(metrics=metrics,shots = 7))