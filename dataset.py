import torch
import pytorch_lightning as pl
import transformers
import numpy as np
import matplotlib.pyplot as plt
import pdb
from bertviz.bertviz import head_view
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer, BertConfig, BertForSequenceClassification
import re, os, tqdm, requests
from datasets import load_dataset
import torch.nn as nn
import logging
import torchmetrics
import wandb 
from collections import OrderedDict
from torch import optim
from torch.utils.data import DataLoader
import argparse

import collections
from typing import *

#https://github.com/HelloJocelynLu/t5chem/blob/main/t5chem/archived/MultiTask.py for more info
# collections.Counter(dataset["test"]["label"])

class SequenceDataset(torch.utils.data.Dataset):
    """Protein sequence dataset"""
    def __init__(self, inputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.utils.data.Dataset:
        super().__init__()
        self.inputs = inputs
        self.targets = targets
    
    def __len__(self):
        return len(self.targets) #train length...

    def __getitem__(self, idx):
        input_ids = self.inputs["input_ids"][idx] #B, L
        token_type_ids = self.inputs["token_type_ids"][idx] #B, L
        attention_mask = self.inputs["attention_mask"][idx] #B, L
        input_reformats = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
        target_reformats = {"labels": self.targets[idx]}
        return input_reformats, target_reformats

class CustomSequenceDataset(torch.utils.data.Dataset):
    """Protein sequence dataset"""
    def __init__(self, inputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.utils.data.Dataset:
        super().__init__()
        self.inputs = inputs
        self.targets = targets
    
    def __len__(self):
        return len(self.targets) #train length...

    def __getitem__(self, idx):
        input_ids = self.inputs["input_ids"][idx] #B, L
        token_type_ids = self.inputs["token_type_ids"][idx] #B, L
        attention_mask = self.inputs["attention_mask"][idx] #B, L
        input_reformats = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
        target_reformats = {"labels": self.targets[idx]}
        return input_reformats, target_reformats
    
    
class NERSequenceDataset(torch.utils.data.Dataset):
    """Protein sequence dataset
    WIP!"""
    def __init__(self, inputs: Dict) -> torch.utils.data.Dataset:
        super().__init__()
        self.inputs, self.targets = self.process_inputs(inputs) #proper inputs and made
        
    @staticmethod
    def process_inputs(inputs: Dict):
        nframes = len(inputs)
        
        return inputs_formatted, targets
        
    def __len__(self):
        return len(self.inputs) #train length...

    def __getitem__(self, idx):
        input_ids = self.inputs["input_ids"][idx] #B, L
        token_type_ids = self.inputs["token_type_ids"][idx] #B, L
        attention_mask = self.inputs["attention_mask"][idx] #B, L
        input_reformats = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
        target_reformats = {"labels": self.targets[idx]}
        return input_reformats, target_reformats

    @classmethod
    def from_json(cls, filename: str):
        assert os.path.split(filename)[-1].split(".")[-1] == "json", "not a json file!" #get extension
        with open(filename, "r") as f:
            data = json.load(filename)
        assert isinstance(data, dict), "wrong data format!"
        return cls(data)    
    
if __name__ == "__main__":
    from train import get_args
    hparam = get_args()
    dataset = load_dataset("yarongef/human_proteome_triplets", cache_dir=hparam.load_data_directory)
    tokenizer=BertTokenizer.from_pretrained("Rostlab/prot_bert",do_lower_case=False, return_tensors="pt",cache_dir=hparam.load_model_directory)
    
#     stage="train"
#     x = []
#     for i in range(len(dataset[stage]["Seq"])):
#         x.append(' '.join(dataset[stage]["Seq"][i]))
#     proper_inputs = x #Spaced btw letters

    import explore_data as ed
    parser = ed.DataParser(filename="pfcrt.csv")
    data = parser.data
    data_trunc = parser.select_columns() 
    
    x = []
    for i in range(len(data_trunc)):
        x.append(' '.join(data_trunc.iloc[i,1])) #AA Sequence
    proper_inputs = x
    
    inputs = tokenizer.batch_encode_plus(proper_inputs,
                                      add_special_tokens=True,
                                      padding=True, truncation=True, return_tensors="pt",
                                      max_length=hparam.max_length) #Tokenize inputs as a dict type of Tensors
    
    targets = data_trunc.iloc[:,2:].values #list type including nans; (B,3)
    targets = torch.from_numpy(targets).view(len(targets), -1) #target is originally list -> change to Tensor (B,1)
        
