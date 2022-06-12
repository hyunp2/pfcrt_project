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
import dataset as dl
import argparse
from typing import *
from torchcrf import CRF

class BertNERTokenizer(torch.nn.Module):
    def __init__(self, ner_config):
        super().__init__()
        self.ner_config
        
    def batch_encode_plus(self, inputs: torch.Tensor, targets: torch.LongTensor):
        #inputs: B,residue_length; targets: B,residue_length
        pass
    
