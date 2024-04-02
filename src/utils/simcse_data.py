import numpy as np
import os, os.path as osp
from collections import defaultdict
import torch
from .data_utils import load_data

def load_data_label_desc(data_args):
    text_dir = data_args.data_dir
    train_text, train_label, train_l2d = load_and_proc_text(text_dir, "train")
    label_desc, num_label = load_label_desc(text_dir)
    return {
        "train": (train_text, train_label, train_l2d),
        "test": (train_text, train_label, train_l2d),
        "label_desc": label_desc,
        "label_desc_2": label_desc,
    }

def load_text_data(data_args):
    text_dir = data_args.data_dir
    train_text, train_label, train_l2d = load_and_proc_text(text_dir, "train")
    test_text, test_label, test_l2d = load_and_proc_text(text_dir, "test")
    label_desc, num_label = load_label_desc(text_dir)
    return {
        "train": (train_text, train_label, train_l2d),
        "test": (test_text, test_label, test_l2d),
        "label_desc": label_desc,
        "label_desc_2": label_desc,
    }

def augment_text(text, generated, num_samples=5):
    generated = np.array(generated).reshape(len(text), num_samples)
    aug_text = []
    for i, x in enumerate(text):
        for a in generated[i]:
            aug_text.append(x + " " + a.strip())
    return aug_text
  
def load_and_proc_text(path, mode='train'):
    text_path = os.path.join(path, f'{mode}.txt')
    label_path = os.path.join(path, f'{mode}_labels.txt')
    texts = load_data(text_path)
    labels = load_data(label_path)
    labels = [int(l) for l in labels]
    
    l_count = defaultdict(int)
    for l in labels:
        l_count[l] += 1
    l2d = defaultdict(list)
    for i,label in enumerate(labels):
        l2d[label].append(i)
    
    return texts, labels, l2d

def load_label_desc(data_dir):
    descs = [line.strip() for line in open(os.path.join(data_dir, 'label_names.txt'))]
    label_num = len(descs)
    return descs, label_num

def dest2prompt(desc, data_name):
    label_vocab = defaultdict(list)
    for l,line in enumerate(desc):
        label_vocab[l].append(line.strip())

    p2l,w2l = {},{}

    if 'news' in data_name.lower():
        for l in label_vocab:
            for w in label_vocab[l]:
                p2l['A {} news.'.format(w)] = l
    elif 'topics' in data_name.lower():
        for l in label_vocab:
            for w in label_vocab[l]:
                p2l['Topic: {}.'.format(w)] = l              
    elif 'imdb' in data_name.lower():
        for l in label_vocab:
            for w in label_vocab[l]:
                p2l['In summary , the film was {}.'.format(w)] = l
    elif 'dbpedia' in data_name:
        for l in label_vocab:
            for w in label_vocab[l]:
                p2l['Category: {}.'.format(w)] = l
    elif 'yelp' in data_name:
        for l in label_vocab:
            for w in label_vocab[l]:
                p2l['In summary, the restaurant is {}.'.format(w)] = l
    elif 'amazon' in data_name:
        for l in label_vocab:
            for w in label_vocab[l]:
                p2l['In summary, the product was {}.'.format(w)] = l
    l2p = defaultdict(list)
    for p in p2l:
        l2p[p2l[p]].append(p)
    return p2l, l2p

def get_list(d, ids):
    return [d[i] for i in ids]

def unpack_field(ds, train_data, test_data, label_desc):
    ds.train_texts, ds.train_labels, ds.train_l2d = train_data
    ds.test_texts, ds.test_labels, ds.test_l2d = test_data
    ds.label_desc = label_desc
    ds.label_num = len(label_desc)
    ds.p2l, ds.l2p = dest2prompt(ds.label_desc, ds.data_name)

def unpack_field_2(ds, train_data, test_data, label_desc, label_desc2):
    ds.train_texts, ds.train_labels, ds.train_l2d = train_data
    ds.test_texts, ds.test_labels, ds.test_l2d = test_data
    ds.label_desc = label_desc
    ds.label_desc2 = label_desc2
    ds.label_num = len(label_desc)
    ds.label_num2 = len(label_desc2)
    ds.p2l, ds.l2p = dest2prompt(ds.label_desc, ds.data_name)
    ds.p2l2, ds.l2p2 = dest2prompt(ds.label_desc2, ds.data_name)

class DataUtils:
    def __init__(self, data_dict, args):
        super().__init__()
        self.data_dict = data_dict
        self.data_name = osp.basename(args.data_dir)
        self.args = args
        self.train_data = data_dict['train'] 
        self.test_data = data_dict['test']
        self.label_desc = data_dict['label_desc']
        unpack_field(self, self.train_data, self.test_data, self.label_desc)
        if "train_sents" in data_dict:
            self.train_sents = data_dict['train_sents']
        self.train_num = len(self.train_texts)
        
    def reset_label_desc(self, label_desc):
        self.label_desc = label_desc
        self.label_num = len(label_desc)
        self.p2l, self.l2p = dest2prompt(self.label_desc, self.data_name)

    def get_label_prompt(self):
        return self.p2l, self.l2p

    def generate_pseudo_label(self, model, eval_batch_size=500, refresh_index=True):
        model.eval()
        pseudo_labels, Q, label_scores, similarity_score = model.generate_pseudo_label_for_augmented_text(
            self.train_texts, 
            self.p2l, labels=self.train_labels,
            batch_size=eval_batch_size)
        if refresh_index:
            self.pseudo_labels = pseudo_labels
            self.Q = torch.from_numpy(Q)
            self.label_scores = label_scores
        model.train()
        return pseudo_labels, Q, label_scores, similarity_score

class DataUtils2:   
    def __init__(self, data_dict, args, weight1, weight2):
        super().__init__()
        self.data_dict = data_dict
        self.weight1 = weight1
        self.weight2 = weight2
        self.data_name = osp.basename(args.data_dir)
        self.args = args
        self.train_data = data_dict['train'] 
        self.test_data = data_dict['test'] 
        self.label_desc = data_dict['label_desc']
        self.label_desc2 = data_dict['label_desc2']
        unpack_field_2(self, self.train_data, self.test_data, self.label_desc, self.label_desc2)
        if "train_sents" in data_dict:
            self.train_sents = data_dict['train_sents']
        self.train_num = len(self.train_texts)
        
    def reset_label_desc(self, label_desc):
        self.label_desc = label_desc
        self.label_num = len(label_desc)
        self.p2l, self.l2p = dest2prompt(self.label_desc, self.data_name)
        self.p2l2, self.l2p2 = dest2prompt(self.label_desc2, self.data_name)
    
    def get_label_prompt(self):
        return self.p2l, self.l2p
    
    def generate_pseudo_label2(self, model, eval_batch_size=500, refresh_index=True):
        model.eval()
        pseudo_labels, Q, label_scores, similarity_score = model.generate_pseudo_label_for_augmented_text2(
            self.train_texts, 
            self.p2l,
            self.p2l2,
            self.weight1,
            self.weight2,
            labels=self.train_labels,
            batch_size=eval_batch_size)
        if refresh_index:
            self.pseudo_labels = pseudo_labels
            self.Q = torch.from_numpy(Q)
            self.label_scores = label_scores
        model.train()
        return pseudo_labels, Q, label_scores, similarity_score