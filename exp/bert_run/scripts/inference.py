from transformers import Trainer
from transformers import Trainer, TrainingArguments,AutoModelForSequenceClassification,AutoTokenizer
from utils.dataset_helper.bert.bert_dataset import prepare_infer_dataset

# def bert_infer(model_id, dataset, batch_size, output_dir):
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     tokenizer.model_max_length = 512
#     infer_dataset=dataset

#     model = AutoModelForSequenceClassification.from_pretrained(model_id)
#     args = TrainingArguments(
#         output_dir=output_dir, 
#         per_device_eval_batch_size=batch_size, 
#         do_train=False,
#         do_eval=False,
#         report_to="none")

#     trainer = Trainer(model=model, args = args)
    
#     predictions = trainer.predict(infer_dataset)

#     logits = predictions.predictions
#     preds = logits.argmax(axis=-1)

#     return preds



