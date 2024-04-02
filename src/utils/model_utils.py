import torch
from transformers import AutoTokenizer, AutoModel

def load_ret_model(model_args, dtype=torch.float32):
    ret_model_name = model_args.simcse_model_name_or_path
    ret_model = AutoModel.from_pretrained(ret_model_name)
    ret_model = ret_model.to(device=model_args.device, dtype=dtype)
    ret_tokenizer = AutoTokenizer.from_pretrained(ret_model_name)
    return ret_model, ret_tokenizer
