from huggingface_hub import get_token
from transformers import Trainer, TrainingArguments,AutoModelForSequenceClassification,AutoTokenizer
import draccus
from dataclasses import dataclass
from utils.dataset_helper.bert.bert_dataset import prepare_dataset_path_based
from utils.dataset_helper.bert.bert_metric import compute_metrics
from utils.language_models.models import ALL_MODELS
from utils.dataset_helper.bert.bert_dataset import *
import wandb

@dataclass
class sft_config:
    model_name: str = ""
    train_data_path:str = ""
    val_data_path:str = ""
    metric:str = "writing-style"
    include_query:str = False
    output_dir:str= ""
    per_device_train_batch_size:int=32
    per_device_eval_batch_size:int=16
    learning_rate:float=5e-5
    num_train_epochs:int=5
    bf16:bool=True
    optim:str="adamw_torch_fused" # improved optimizer 
    lr_scheduler_type: str = "linear"
    warmup_ratio:float = 0.2
    # logging & evaluation strategies
    logging_strategy:str="steps"
    logging_steps:int=1
    eval_strategy:str="epoch"
    save_strategy:str="epoch"
    run_id :str= ""
    save_total_limit:int=1
    load_best_model_at_end:bool=True
    metric_for_best_model:str="f1"
    report_to:str="wandb"
    wandb_entity:str="charles2605722943-uc-santa-barbara"
    wandb_project:str="bert"
    wandb_group:str="group"
    push_to_hub:bool=False

@draccus.wrap()
def finetune(cfg:sft_config):
    #set up wandb
    wandb.init(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        group=cfg.wandb_group,
        name=f"{cfg.run_id}"
    )

    #retrieve the model
    if "BERT" in cfg.model_name:
        model_id = ALL_MODELS["BERT"][cfg.model_name]

    #load the model and tokenize 
    # Model id to load the tokenizer

    # Load Tokenizer

    #prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.model_max_length = 512

    dataset = prepare_dataset_path_based(train_data_path=cfg.train_data_path, 
                                         val_data_path = cfg.val_data_path,
                                         metric=cfg.metric, 
                                         tokenizer=tokenizer,
                                         include_query=cfg.include_query)
    
    labels = dataset["train"].features["labels"].names
    num_labels = len(labels)
    label2id, id2label = dict(), dict()

    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    print(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_labels, label2id=label2id, id2label=id2label
    )


    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        bf16=cfg.bf16,
        optim=cfg.optim,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        logging_strategy=cfg.logging_strategy,
        logging_steps=cfg.logging_steps,
        eval_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        report_to=cfg.report_to,
        push_to_hub=cfg.push_to_hub,
        run_name=cfg.run_id
    )

    # Create a Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
    )

    # Start training
    trainer.train()
    wandb.finish()

if __name__ == "__main__":
    finetune()



