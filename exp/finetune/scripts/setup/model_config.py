# import torch
# from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# def setup_model(model_path):
#     model_id = model_path

#     # Check if GPU supports bfloat16
#     if torch.cuda.get_device_capability()[0] < 8:
#         raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")

#     model_kwargs = dict(
#         attn_implementation="flash_attention_2",
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#     )

#     model_kwargs["quantization_config"] = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
#         bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
#     )

#     model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
#     processor = AutoProcessor.from_pretrained(model_id)

#     # Use right padding to avoid issues during training
#     processor.tokenizer.padding_side = "right"

#     return model, processor
