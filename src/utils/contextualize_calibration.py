from openprompt.pipeline_base import PromptDataLoader, PromptForClassification
from typing import *
import torch
from tqdm import tqdm

def calibrate(dataset_length, prompt_model: PromptForClassification, dataloader: PromptDataLoader) -> torch.Tensor:

    label_logits = []
    all_logits_sorted100 = []
    all_indices_sorted100 = []
    aver = torch.zeros(1,50265)
    aver = aver.to(prompt_model.device)

    prompt_model.eval()
    for batch in tqdm(dataloader,desc='ContextCali'):
        batch = batch.to(prompt_model.device)
        outputs, logits = prompt_model(batch) 

        label_logits.append(outputs.detach().to("cpu"))
        sorted_logits, sorted_indices = torch.sort(logits.detach(), descending=True)
        top100_sorted_logits = sorted_logits[:, :100]
        top100_sorted_indices = sorted_indices[:, :100]
        all_logits_sorted100.append(top100_sorted_logits.to("cpu"))
        all_indices_sorted100.append(top100_sorted_indices.to("cpu"))
      
        total_logits = torch.sum(logits.detach(), dim=0)
        total_logits.div_(dataset_length) 
        aver_logits = total_logits
        aver.add_(aver_logits)

    label_logits = torch.cat(label_logits, dim=0) 
    all_logits = torch.cat(all_logits_sorted100, dim=0)
    all_indices = torch.cat(all_indices_sorted100, dim=0)
    
    return all_logits, all_indices, aver, label_logits
