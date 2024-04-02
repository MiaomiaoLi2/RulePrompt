import numpy as np
import torch
import math
import torch.nn as nn
import math
import os
import torch.nn.functional as F
from torch_scatter import scatter
from transformers import AutoModel,AutoTokenizer
from .data_utils import to_cuda

class Similarity(nn.Module):
    def __init__(self, norm=True):
        super().__init__()
        self.norm = norm

    def forward(self, x, y, temp=1.0):
        x, y = x.float(), y.float() # for bfloat16 optimization, need to convert to float
        if self.norm:
            x = x / torch.norm(x, dim=-1, keepdim=True)
            y = y / torch.norm(y, dim=-1, keepdim=True)
        return torch.matmul(x, y.t()) / temp

class Model(nn.Module):
    @classmethod
    def load(cls, model_args):
        cache_dir=os.path.join(model_args.cache_dir, model_args.simcse_model_name_short)
        transformer = AutoModel.from_pretrained(model_args.simcse_model_name_or_path, cache_dir = cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_args.simcse_model_name_or_path, cache_dir = cache_dir)
        return cls(transformer, tokenizer, model_args)

    def __init__(self, model, tokenizer, model_args=None):
        super().__init__()
        self.transformer = model
        self.tokenizer = tokenizer
        self.model_args = model_args
        if model_args is not None:
            self.model_name = model_args.simcse_model_name_or_path
            simcse_model_name_short = os.path.basename(self.model_name)
            self.cache_dir = os.path.join(model_args.cache_dir, simcse_model_name_short)
        self.sim = Similarity(norm=True)
        self.pooling = model_args.pooling # default simcse is pooler
        self.temperature = 0.1
        self.max_text_length = model_args.max_text_length
    
    def get_device(self):
        return self.transformer.device

    def reload(self):
        device = self.get_device()
        self.transformer = AutoModel.from_pretrained(self.model_name, cache_dir = self.cache_dir).to(device)

    def generate_pseudo_label_for_augmented_text(self, input_texts, p2l, labels=None, batch_size=500):
        self.eval()
        device = self.get_device()
        id2l = torch.tensor([p2l[desc] for desc in p2l]).to(device)
        l_num = max([e+1 for e in id2l])
        l_pred_count = np.zeros(l_num)
        desc_embeds = self.encode_label_prompt(p2l).float()

        result = []
        P, Q = [], []
        for c_id in range(math.ceil(len(input_texts)/batch_size)):
            texts = input_texts[c_id*batch_size:(c_id+1)*batch_size]
            text_embeds = self.encode(texts)
            text_embeds = text_embeds.float() 
            text_embeds = text_embeds / torch.norm(text_embeds, dim=-1, keepdim=True)
            desc_embeds = desc_embeds / torch.norm(desc_embeds, dim=-1, keepdim=True)

            scores = torch.matmul(text_embeds, desc_embeds.transpose(0,1)) 
            if c_id == 0:
                confidence_scores = scores
            else:
                confidence_scores=torch.cat((confidence_scores,scores),dim=0)

            q_scores = F.softmax(scores / self.temperature, dim=-1)
            Q.append(q_scores.detach().cpu().numpy())
            p_scores = F.softmax(scores, dim=-1)
            P.append(p_scores.detach().cpu().numpy())
            
            preds = torch.argmax(p_scores, dim=-1).cpu().numpy()
            idx, cnts = np.unique(preds, return_counts = True)
            for i in range(len(idx)):
                l_pred_count[idx[i]] += cnts[i]
            result.extend(preds)
        P = np.concatenate(P, 0)
        Q = np.concatenate(Q, 0)

        return result, Q, P, confidence_scores 
    
    def generate_pseudo_label_for_augmented_text2(self, input_texts, p2l, p2l2, weight1, weight2, labels=None, batch_size=500):
        self.eval()
        device = self.get_device()
        id2l = torch.tensor([p2l[desc] for desc in p2l]).to(device)
        id2l2 = torch.tensor([p2l2[desc] for desc in p2l2]).to(device)
        l_num = max([e+1 for e in id2l])
        l_pred_count = np.zeros(l_num)
        desc_embeds = self.encode_label_prompt(p2l).float()
        desc_embeds2 = self.encode_label_prompt(p2l2).float()
        weight1 = np.array(weight1)[:,np.newaxis]
        weight2 = np.array(weight2)[:,np.newaxis]
        desc_embeds = desc_embeds.cpu()*weight1 + desc_embeds2.cpu()*weight2
        desc_embeds = desc_embeds.to(device).float()

        acc,result = [],[]
        P, Q = [], []
        for c_id in range(math.ceil(len(input_texts)/batch_size)):
            texts = input_texts[c_id*batch_size:(c_id+1)*batch_size]
            text_embeds = self.encode(texts)
            text_embeds = text_embeds.float() 
            text_embeds = text_embeds / torch.norm(text_embeds, dim=-1, keepdim=True)
            desc_embeds = desc_embeds / torch.norm(desc_embeds, dim=-1, keepdim=True)
            scores = torch.matmul(text_embeds, desc_embeds.transpose(0,1)) 
            if c_id == 0:
                confidence_scores = scores
            else:
                confidence_scores=torch.cat((confidence_scores,scores),dim=0)
            q_scores = F.softmax(scores / self.temperature, dim=-1)
            Q.append(q_scores.detach().cpu().numpy())
            p_scores = F.softmax(scores, dim=-1)
            P.append(p_scores.detach().cpu().numpy())
            
            preds = torch.argmax(p_scores, dim=-1).cpu().numpy()
            idx, cnts = np.unique(preds, return_counts = True)
            for i in range(len(idx)):
                l_pred_count[idx[i]] += cnts[i]
            result.extend(preds)
        P = np.concatenate(P, 0)
        Q = np.concatenate(Q, 0)
        return result, Q, P, confidence_scores

    def generate_pseudo_label(self, input_texts, p2l, labels=None, batch_size=500):
        self.eval()
        device = self.get_device()
        id2l = torch.tensor([p2l[desc] for desc in p2l]).to(device)
        l_num = max([e+1 for e in id2l])
        lp_count = torch.zeros(l_num)
        l_pred_count = np.zeros(l_num)
        for desc in p2l:
            lp_count[p2l[desc]] += 1
        id2desc = [desc for desc in p2l]
        desc_embeds = self.encode(id2desc)

        acc,result = [],[]
        all_scores = []
        probs = np.zeros(len(input_texts))
        for c_id in range(math.ceil(len(input_texts)/batch_size)):
            texts = input_texts[c_id*batch_size:(c_id+1)*batch_size]
            text_embeds = self.encode(texts)
            scores = torch.matmul(text_embeds, desc_embeds.transpose(0,1))
            scores = scores.view(-1, len(id2desc))
            l_scores = scatter(scores, id2l, dim=-1, reduce='mean')
            l_scores = F.softmax(l_scores, dim=-1)
            all_scores.append(l_scores.detach().cpu().float().numpy())
            preds = torch.argmax(l_scores, dim=-1).cpu().numpy()
            idx, cnts = np.unique(preds, return_counts = True)
            for i in range(len(idx)):
                l_pred_count[idx[i]] += cnts[i]
            result.extend(preds)
            probs = l_scores[range(len(preds)), preds].cpu().float().numpy()
        all_scores = np.concatenate(all_scores, 0)
        top_ids = np.argsort(probs)[::-1]
        return result, top_ids, np.array(all_scores)
    
    @torch.no_grad()
    def encode_label_prompt(self, p2l):
        self.eval()
        device = self.get_device()
        id2l = torch.tensor([p2l[desc] for desc in p2l]).to(device)
        id2desc = [desc for desc in p2l]
        desc_embeds = self.encode(id2desc)
        label_emb = scatter(desc_embeds, id2l, dim=0, reduce='mean')
        self.train()
        return label_emb

    @torch.no_grad()
    def encode(self, texts, batch_size=None, cpu=False):
        self.eval()
        embeddings = []
        def enc(texts):
            inputs = self.tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt", max_length=self.max_text_length)
            inputs = to_cuda(inputs, device=self.transformer.device)
            outputs = self.transformer(**inputs, output_hidden_states=True, return_dict=True).pooler_output
            return outputs
        if batch_size is None or len(texts) <= batch_size:
            embeddings = enc(texts)
        else:
            for i in range(len(texts)//batch_size + 1):
                s = i*batch_size
                e = (i+1)*batch_size
                if s >= len(texts):
                    break
                outputs = enc(texts[s:e])
                embeddings.append(outputs)
            embeddings = torch.cat(embeddings, 0)
        if cpu:
            embeddings = embeddings.cpu()
        self.train()
        return embeddings
