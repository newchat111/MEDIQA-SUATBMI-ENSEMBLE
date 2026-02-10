from peft import LoraConfig

def lora_config(alpha, dropout, r):
    
    peft_config = LoraConfig(
        lora_alpha=alpha,
        lora_dropout=dropout,
        r=r,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=[
            "lm_head",
            "embed_tokens",
        ],
    )
    return peft_config