import torch
# import pytorch_lightning as pl
import lightning as L
import transformers
import numpy as np
import matplotlib.pyplot as plt
import pdb
from bertviz.bertviz import head_view
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer, BertConfig, BertForSequenceClassification
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from packaging import version
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

class BertEmbeddingsNER(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, ner_config):
        super().__init__()
        self.word_embeddings = nn.Embedding(ner_config.get("vocab_size"), ner_config.get("hidden_size"), padding_idx=ner_config.get("pad_token_id"))
        self.position_embeddings = nn.Embedding(ner_config.get("max_position_embeddings"), ner_config.get("hidden_size"))
        self.token_type_embeddings = nn.Embedding(ner_config.get("type_vocab_size"), ner_config.get("hidden_size"))

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(ner_config.get("hidden_size"), eps=ner_config.get("layer_norm_eps"))
        self.dropout = nn.Dropout(ner_config.get("hidden_dropout_prob"))
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(ner_config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(ner_config.get("max_position_embeddings")).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long),
                persistent=False,
            )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertNER(torch.nn.Module):
    def __init__(self, bert_model: BertModel, ner_config: dict=None):
       assert isinstance(bert_model, BertModel), "must be a class of BertModel..."
       self.bert_model = bert_model
       embeddings = BertEmbeddingsNER(**ner_config) #Overwrite BertModel's embedding
       self.bert_model.embeddings = embeddings #Overwrite BertModel's embedding
       
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:      
        return self.bert_model(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds,
                               encoder_hidden_states, encoder_attention_mask, 
                              past_key_values, use_cache, output_attentions, output_hidden_states, return_dict)
                         
                             

    

    
