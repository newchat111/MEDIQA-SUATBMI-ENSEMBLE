# from exp.finetune.scripts.setup.model_config import setup_model
from exp.finetune.scripts.setup.collator import collate_fn,transform_to_model_format
from exp.finetune.scripts.setup.lora import lora_config
from datasets import Dataset, DatasetDict
import draccus
from dataclasses import dataclass
from trl import SFTConfig, SFTTrainer
from typing import Optional, Dict, Any
import json
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import torch
from functools import partial
from utils.dataset_helper.gemma.sft_metric import compute_metrics
import wandb
@dataclass
class config:
    #define run_name
    run_name:str = ""
    #pretrained model path
    pretrained_model_id: str = ""
    #data path
    train_data_path:str = ""
    val_data_path:str = ""
    # save paths
    output_dir: str = "medgemma-4b-it-sft-lora-crc100k"
    push_to_hub: bool = True

    # Training hyperparameters
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    eval_accumulation_steps:int = 4
    learning_rate: float = 2e-4
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_torch_fused"

    # Mixed precision / memory
    bf16: bool = True
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: Dict[str, Any] = None  # e.g., {"use_reentrant": False}

    # Logging & saving
    logging_steps: int = 50
    save_strategy: str = "epoch"
    eval_strategy: str = "steps"
    eval_steps: int = 50
    report_to: str = "wandb"
    wandb_entity:str="charles2605722943-uc-santa-barbara"
    wandb_project:str="gemma"
    wandb_group:str="group"
    save_total_limit:int=1
    metric_for_best_model:str="eval_loss"
<<<<<<< HEAD
=======
    load_best_model_at_end:bool = True
>>>>>>> master

    # Dataset
    dataset_kwargs: Dict[str, Any] = None 
    remove_unused_columns: bool = False
    label_names: list[str] = None 

    #LORA
    apply_lora:bool = True
    lora_alpha:float=16
    lora_dropout:float=0.05 
    lora_r:int=8


@draccus.wrap()
def finetune(cfg:config):
    model_id = cfg.pretrained_model_id

    wandb.init(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        group=cfg.wandb_group,
        name=f"{cfg.run_name}"
    )
    # Check if GPU supports bfloat16
    if torch.cuda.get_device_capability()[0] < 8:
        raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")

    model_kwargs = dict(
        pretrained_model_name_or_path=model_id,
        # attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
        device_map="auto",
    )

    model = AutoModelForImageTextToText.from_pretrained(**model_kwargs)
    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=model_id)

    # Use right padding to avoid issues during training
    processor.tokenizer.padding_side = "right"

    #read and prepare the data
    with open(cfg.train_data_path, 'r') as f:
        train_list = json.load(f)

    with open(cfg.val_data_path, 'r') as f:
        val_list = json.load(f)

    train_processed = transform_to_model_format(train_list)
    val_processed = transform_to_model_format(val_list)

    train_dataset = Dataset.from_list(train_processed)
    val_dataset = Dataset.from_list(val_processed)

    # Combine into a DatasetDict
    data = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })

    #set up config
    args = SFTConfig(
        output_dir=cfg.output_dir,
        push_to_hub=cfg.push_to_hub,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        max_grad_norm=cfg.max_grad_norm,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        optim=cfg.optim,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs=cfg.gradient_checkpointing_kwargs,
        eval_accumulation_steps=cfg.eval_accumulation_steps,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps,
        report_to=cfg.report_to,
        dataset_kwargs=cfg.dataset_kwargs,
        remove_unused_columns=cfg.remove_unused_columns,
        label_names=cfg.label_names,
        run_name=cfg.run_name,
        save_total_limit=cfg.save_total_limit,
<<<<<<< HEAD
        metric_for_best_model=cfg.metric_for_best_model
=======
        metric_for_best_model=cfg.metric_for_best_model,
        load_best_model_at_end = cfg.load_best_model_at_end
>>>>>>> master
    )
    #set up the model and processor
    #set up lora config if decided to apply lora during training
    if cfg.apply_lora:
        peft_config = lora_config(alpha = cfg.lora_alpha, dropout=cfg.lora_dropout, r=cfg.lora_r)
    
    else:
        peft_config = None
        
    collator = partial(collate_fn, processor)

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],  # Use subset of validation set for faster run
        peft_config=peft_config,
        processing_class=processor,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    wandb.finish()

if __name__ == "__main__":
    finetune()





