import torch
# import pytorch_lightning as pl
import lightning as L
import transformers
import numpy as np
import matplotlib.pyplot as plt
import pdb
from bertviz import head_view
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer, BertConfig, BertForSequenceClassification, get_linear_schedule_with_warmup, Adafactor, AdamW
import re, os, tqdm, requests
from datasets import load_dataset
import torch.nn as nn
import logging
import torchmetrics
import wandb 
from collections import OrderedDict
import collections
from torch import optim
from torch.utils.data import DataLoader
import dataset as dl
import argparse
from typing import *
from torchcrf import CRF
import BertNER
import BertNERTokenizer

#https://github.com/HelloJocelynLu/t5chem/blob/main/t5chem/archived/MultiTask.py for more info

# classifier.bert.pooler.dense.weight.requires_grad
# classifier.bert.pooler.dense.training
# classifier.classifier.weight.requires_grad
logging.basicConfig()
logger = logging.getLogger("BERT Fine-tuning")
logger.setLevel(logging.DEBUG)

class ProtBertClassifier(L.LightningModule):
    """
    # https://github.com/minimalist-nlp/lightning-text-classification.git
    
    Sample model to show how to use BERT to classify sentences.
    
    :param hparams: ArgumentParser containing the hyperparameters.
    """

    def __init__(self, hparam: argparse.ArgumentParser) -> None:
        super(ProtBertClassifier, self).__init__()
#         self.save_hyperparameters() #self.hparams to access all; no need to specify arguments when loading checkpoint???
        self.hparam = hparam
        self.ner_config = self.hparam.ner_config
        self.batch_size = self.hparam.batch_size
        self.model_name = self.hparam.model_name #"Rostlab/prot_bert_bfd"  
        self.ner = self.hparam.ner #bool
        
        self.num_labels = np.unique(self.dataset["train"]["label"]).__len__() #2 for Filippo; many for Matt 

        self.z_dim = self.hparam.z_dim #Add this!
        if self.hparam.loss == "contrastive": self.register_parameter("W", torch.nn.Parameter(torch.rand(self.z_dim, self.z_dim))) #CURL purpose

        ###FINETUNE OPTIONS 1 (loss weight, dataset)
        self.ce_loss_weight_initialized = False
        parser = DataParser.get_data("pfcrt.xlsx")
        data_trunc = parser.select_columns(fill_na=self.hparam.fillna_val)
        self.dataset = data_trunc

        ###FINETUNE OPTIONS 2 (finetune classfication)
        if not self.hparam.finetune:
            # build model
            _ = self.__build_model() if not self.ner else self.__build_model_ner()

            # Loss criterion initialization.
            _ = self.__build_loss() if not self.ner else self.__build_model_ner()

            self.freeze_encoder()
        else:
            # build model
            _ = self.__build_model_finetune() if not self.ner else self.__build_model_ner()

            # init loss weight
            self.__build_weight()
            
            # Loss criterion initialization.
            _ = self.__build_loss_finetune() if not self.ner else self.__build_model_ner()
            
            self.freeze_encoder()

    def __build_model(self) -> None:
        """ Init BERT model + tokenizer + classification head."""
        #model = locals()["model"] if locals()["model"] and isinstance(locals()["model"], BertModel) else BertModel.from_pretrained(self.model_name, cache_dir=self.hparam.load_model_directory)
        model = BertModel.from_pretrained(self.model_name, cache_dir=self.hparam.load_model_directory)
 
        self.model = model
        self.encoder_features = self.z_dim

        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, do_lower_case=False, return_tensors="pt", cache_dir=self.hparam.load_model_directory)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(self.encoder_features*4, self.encoder_features),
            nn.Linear(self.encoder_features, self.num_labels),
            nn.Tanh(),
        )
        if self.hparam.loss == "contrastive": 
            self.make_hook()

        wandb.init(project="DL_Sequence_Collab", entity="hyunp2", group="DDP_runs")
        wandb.watch(self.head)

    def __build_model_ner(self) -> None:
        """ Init BERT model + tokenizer + classification head.
        Model and Tokenizer has to be rewritten! 
        WIP!"""
        #model = locals()["model"] if locals()["model"] and isinstance(locals()["model"], BertModel) else BertModel.from_pretrained(self.model_name, cache_dir=self.hparam.load_model_directory)
        model = BertModel.from_pretrained(self.model_name, cache_dir=self.hparam.load_model_directory)
        model = BertNER.BertNER(model, self.ner_config)
        
        self.model = model
        self.encoder_features = self.z_dim

        # Tokenizer
        self.tokenizer = BertNERTokenizer.BertNERTokenizer(self.ner_config)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(self.encoder_features, self.encoder_features),
            nn.Linear(self.encoder_features, self.num_labels),
            nn.Tanh(),
        )
        if self.hparam.loss == "contrastive": 
            self.make_hook()

        wandb.init(project="DL_Sequence_Collab_Matt", entity="hyunp2")
        wandb.watch(self.head)

    def __build_model_finetune(self) -> None:
        """ Init BERT model + tokenizer + classification head."""
        #model = locals()["model"] if locals()["model"] and isinstance(locals()["model"], BertModel) else BertModel.from_pretrained(self.model_name, cache_dir=self.hparam.load_model_directory)
        model = BertModel.from_pretrained(self.model_name, cache_dir=self.hparam.load_model_directory)
 
        self.model = model
        self.encoder_features = self.z_dim

        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, do_lower_case=False, return_tensors="pt", cache_dir=self.hparam.load_model_directory)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(self.encoder_features*4, self.encoder_features),
            nn.Linear(self.encoder_features, self.num_labels),
            nn.Tanh(),
        )
        
        self.classifier0 = nn.Sequential(
            nn.Linear(self.encoder_features, self.encoder_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_features, self.encoder_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_features, self.encoder_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_features, 3),
        )
        
        self.classifier1 = nn.Sequential(
            nn.Linear(self.encoder_features, self.encoder_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_features, self.encoder_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_features, self.encoder_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_features, 3),
        )
        
        self.classifier2 = nn.Sequential(
            nn.Linear(self.encoder_features, self.encoder_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_features, self.encoder_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_features, self.encoder_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_features, 2),
        )
        
        if self.hparam.loss == "contrastive": 
            self.make_hook()

        self.wandb_run = wandb.init(project="DL_Sequence_Collab", entity="hyunp2", group="DDP_runs")
        wandb.watch(self.head)
    
    def make_hook(self, ):
        self.fhook = dict()
        def hook(m, i, o):
            self.fhook["encoded_feats"] = o #(B,1024)
        self.fhook_handle = self.head[0].register_forward_hook(hook) #Call Forward hook with "model.fhook["encoded_feats"]" of (B,C); for NER, it is (B,L,C)

    def __build_weight(self, nonuniform_weight=True):
        targets = self.dataset.iloc[:,2:].values #list type including nans; (B,3)
        targets = torch.from_numpy(targets).view(len(targets), -1).long().to(self.device) #target is originally list -> change to Tensor (B,1)
        if nonuniform_weight:
            valid_targets = (targets < self.hparam.fillna_val) #B,3
            valid_targets0 = targets[valid_targets[:,0]][:,0].to(targets) #only for targ0
            valid_targets1 = targets[valid_targets[:,1]][:,1].to(targets) #only for targ1
            valid_targets2 = targets[valid_targets[:,2]][:,2].to(targets) #only for targ2
#             print(cf.red(f"0: {valid_targets0.size()}, 1: {valid_targets1.size()}, 2: {valid_targets2.size()}" ))
            self.weight0 = (1 / (torch.nn.functional.one_hot(valid_targets0).sum(dim=0) / valid_targets0.size(0) + torch.finfo(torch.float32).eps)).to(self.device)
            self.weight1 = (1 / (torch.nn.functional.one_hot(valid_targets1).sum(dim=0) / valid_targets1.size(0) + torch.finfo(torch.float32).eps)).to(self.device)
            self.weight2 = (1 / (torch.nn.functional.one_hot(valid_targets2).sum(dim=0) / valid_targets2.size(0) + torch.finfo(torch.float32).eps)).to(self.device)
        else:
            self.weight0 = targets.new_ones(3)
            self.weight1 = targets.new_ones(3)
            self.weight2 = targets.new_ones(2)
            
        self.ce_loss_weight_initialized = True
    
    def __build_loss(self):
        """ Initializes the loss function/s. """
        self._loss = nn.CrossEntropyLoss(label_smoothing=self.hparam.label_smoothing) if self.num_labels > 2 else nn.BCEWithLogitsLoss()

    def __build_loss_ner(self):
        """ Initializes the loss function/s. """
        self._loss = CRF(num_tags=self.num_labels, batch_first=True)

    def __build_loss_finetune(self):
        assert self.ce_loss_weight_initialized, "Loss weights for classification have to be initialized..."
        
        """ Initializes the loss function/s. """
        if self.hparam.use_ce:
            loss0 = nn.CrossEntropyLoss(label_smoothing=self.hparam.label_smoothing, ignore_index=self.hparam.fillna_val, weight=self.weight0) #(logits0, target0) #ignore_index=100 is from dataset!
            loss1 = nn.CrossEntropyLoss(label_smoothing=self.hparam.label_smoothing, ignore_index=self.hparam.fillna_val, weight=self.weight1) #(logits1, target1) #ignore_index=100 is from dataset!
            loss2 = nn.CrossEntropyLoss(label_smoothing=self.hparam.label_smoothing, ignore_index=self.hparam.fillna_val, weight=self.weight2) #(logits2, target2) #ignore_index=100 is from dataset!
        else:
            loss0 = FocalLoss(beta=0.9999, weight=self.weight0) #(logits0, target0) #ignore_index=100 is from dataset!
            loss1 = FocalLoss(beta=0.9999, weight=self.weight1) #(logits1, target1) #ignore_index=100 is from dataset!
            loss2 = FocalLoss(beta=0.9999, weight=self.weight2) #(logits2, target2) #ignore_index=100 is from dataset!

        self.loss0 = loss0
        self.loss1 = loss1
        self.loss2 = loss2
    
    def compute_logits_CURL(self, z_a, z_pos):
        """
        WIP!
        https://github.com/MishaLaskin/curl/blob/8416d6e3869e38ca0e46fcbc54a2f784dc09d7fc/curl_sac.py#:~:text=def%20compute_logits(self,return%20logits
        Uses logits trick for CURL:
        - z_a and z_pos are last layer of classifier! z_pos is T(z_a) where T is Transformation func.
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        assert z_a.size(-1) == z_pos.size(-1) and z_a.size(-1) == self.z_dim, "dimension for CURL mismatch!"
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None] #(B,B)
        labels = torch.arange(logits.shape[0]).to(self.device).long() #(B,)
        loss = self._loss(logits, labels) #Use the defined CE
        return loss

    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            logger.info(f"\n-- Encoder model fine-tuning")
            for param in self.model.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """ freezes the encoder layer. """
        for param in self.model.parameters():
            param.requires_grad = False
        self._frozen = True
    
    # https://github.com/UKPLab/sentence-transformers/blob/eb39d0199508149b9d32c1677ee9953a84757ae4/sentence_transformers/models/Pooling.py
    @staticmethod
    def pool_strategy(features,
                      pool_cls=True, pool_max=True, pool_mean=True,
                      pool_mean_sqrt=True):
        token_embeddings = features['token_embeddings']
        cls_token = features['cls_token_embeddings']
        attention_mask = features['attention_mask']

        ## Pooling strategy
        output_vectors = []
        if pool_cls:
            output_vectors.append(cls_token)
        if pool_max:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if pool_mean or pool_mean_sqrt:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            #If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if pool_mean:
                output_vectors.append(sum_embeddings / sum_mask)
            if pool_mean_sqrt:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        output_vector = torch.cat(output_vectors, 1)
        return output_vector #B,1024*4
    
    def forward(self, input_ids, token_type_ids, attention_mask, return_dict=True):    
        result = self.forward_classify(input_ids, token_type_ids, attention_mask, return_dict=True) if not self.ner else self.forward_ner(input_ids, token_type_ids, attention_mask, return_dict=True)
        return result 

    def forward_classify(self, input_ids, token_type_ids, attention_mask, return_dict=True):
        word_embeddings = self.model(input_ids=input_ids, token_type_ids=token_type_ids,
                                           attention_mask=attention_mask)[0] #last_hidden_state

        pooling = self.pool_strategy({"token_embeddings": word_embeddings,
                                      "cls_token_embeddings": word_embeddings[:, 0],
                                      "attention_mask": attention_mask,
                                      }) 
        logits = self.head[0](pooling) #B,dim; only the first module of sequential
        logits0 = self.classifier0(logits) #B,3
        logits1 = self.classifier1(logits) #B,3
        logits2 = self.classifier2(logits) #B,2
        
        if return_dict:
            if self.hparam.loss == "classification":
                return {"logits0": logits0, "logits1": logits1, "logits2": logits2} #B,num_labels
            elif self.hparam.loss == "contrastive":
                logits = self.fhook["encoded_feats"]
                return {"logits": logits} #B, z_dim (differentiable)
        else:
            if self.hparam.loss == "classification":
                return logits0, logits1, logits2  #(B,3); (B,3); (B,2)
            elif self.hparam.loss == "contrastive":
                logits = self.fhook["encoded_feats"]
                return logits #B, z_dim (differentiable)

    def forward_ner(self, input_ids, token_type_ids, attention_mask, return_dict=True):
        """ Usual pytorch forward function.
        :param tokens: text sequences [batch_size x src_seq_len]
        :param lengths: source lengths [batch_size]
        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        word_embeddings = self.model(input_ids=input_ids, token_type_ids=token_type_ids,
                                           attention_mask=attention_mask)[0] #last_hidden_state
        logits = self.head(word_embeddings) #BLC
        if return_dict:
            if self.hparam.loss == "ner":
                return {"logits": logits} #BLC
        else:
            if self.hparam.loss == "ner":
                return logits #BLC

    def loss(self, predictions: dict, targets: torch.Tensor) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param predictions: model specific output. Must contain a key 'logits' with
            a tensor [batch_size x 1] with model predictions
        :param labels: Label values [batch_size]
        Returns:
            torch.tensor with loss value.
        """
        if self.hparam.loss == "classification" and not self.ner:
            if self.num_labels > 2:
                return self._loss(predictions["logits"], targets["labels"].view(-1, )) #Crossentropy ;; input: (B,2) target (B,)
            else:
                return self._loss(predictions["logits"].amax(dim=-1).view(-1, ), targets["labels"].view(-1, ).to(torch.float32)) #BinaryCrossentropy ;; input: (B,) target (B,)
        elif self.hparam.loss == "classification" and self.ner:
            return self._loss(predictions["logits"], targets["labels"].view(-1, self.num_labels)) #CRF ;; input (B,L,C) target (B,L) ;; B->num_frames & L->num_aa_residues & C->num_lipid_types
        elif self.hparam.loss == "contrastive":
            return self.compute_logits_CURL(predictions["logits"], predictions["logits"]) #Crossentropy -> Need second pred to be transformed! each pred is (B,z_dim) shape

    def on_train_epoch_start(self, ) -> None:
        self.metric_acc0 = torchmetrics.Accuracy(task="multiclass") if self.num_labels > 2 else torchmetrics.Accuracy(task="binary")
        self.metric_acc1 = torchmetrics.Accuracy(task="multiclass") if self.num_labels > 2 else torchmetrics.Accuracy(task="binary")
        self.metric_acc2 = torchmetrics.Accuracy(task="multiclass") if self.num_labels > 2 else torchmetrics.Accuracy(task="binary")
        
    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        #import pdb; pdb.set_trace()
        inputs, targets = batch
        model_out = self.forward(**inputs)
        #print(model_out.size(), targets["labels"].size())
        loss_train = self.loss(model_out, targets)
        
        y = targets["labels"].view(-1,3) #B3
        y0 = y[:,0]
        y1 = y[:,1]
        y2 = y[:,2]
        y_hat0 = model_out["logits0"] #(B,3);(B,3),(B,2)
        y_hat1 = model_out["logits1"] #(B,3);(B,3),(B,2)
        y_hat2 = model_out["logits2"] #(B,3);(B,3),(B,2)
        labels_hat0 = torch.argmax(y_hat0, dim=-1).to(y)
        labels_hat1 = torch.argmax(y_hat1, dim=-1).to(y)
        labels_hat2 = torch.argmax(y_hat2, dim=-1).to(y)

#         train_acc0 = self.metric_acc0(labels_hat0.detach().cpu().view(-1), y0.detach().cpu().view(-1)) #Must mount tensors to CPU;;;; ALSO, val_acc should be returned!
#         train_acc1 = self.metric_acc1(labels_hat1.detach().cpu().view(-1), y1.detach().cpu().view(-1)) #Must mount tensors to CPU;;;; ALSO, val_acc should be returned!
#         train_acc2 = self.metric_acc2(labels_hat2.detach().cpu().view(-1), y2.detach().cpu().view(-1)) #Must mount tensors to CPU;;;; ALSO, val_acc should be returned!
        train_acc0 = balanced_accuracy_score(y0.detach().cpu().numpy().reshape(-1), labels_hat0.detach().cpu().numpy().reshape(-1))
        train_acc1 = balanced_accuracy_score(y1.detach().cpu().numpy().reshape(-1), labels_hat1.detach().cpu().numpy().reshape(-1))
        train_acc2 = balanced_accuracy_score(y2.detach().cpu().numpy().reshape(-1), labels_hat2.detach().cpu().numpy().reshape(-1))
        predY = np.stack([labels_hat0.detach().cpu().numpy().reshape(-1), labels_hat1.detach().cpu().numpy().reshape(-1), labels_hat2.detach().cpu().numpy().reshape(-1)]).T #data,3
        dataY = np.stack([y0.detach().cpu().numpy().reshape(-1), y1.detach().cpu().numpy().reshape(-1), y2.detach().cpu().numpy().reshape(-1)]).T #data,3

        output = {"train_loss": loss_train, "train_acc0": train_acc0, "train_acc1": train_acc1, "train_acc2": train_acc2} #NEVER USE ORDEREDDICT!!!!
        self.wandb_run.log(output)
#         self.log("train_loss", loss_train, prog_bar=True)
#         self.log("train_acc0", train_acc0, prog_bar=True)
#         self.log("train_acc1", train_acc1, prog_bar=True)
#         self.log("train_acc2", train_acc2, prog_bar=True)

        return {"loss": loss_train, "train_acc0": train_acc0, "train_acc1": train_acc1, "train_acc2": train_acc2, "predY": predY, "dataY": dataY}

    def on_train_epoch_end(self, outputs: list) -> dict:
        train_loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        train_predY = np.concatenate([x['predY'] for x in outputs], axis=0)
        train_dataY = np.concatenate([x['dataY'] for x in outputs], axis=0)
        predy0, predy1, predy2 = train_predY[:,0], train_predY[:,1], train_predY[:,2]
        datay0, datay1, datay2 = train_dataY[:,0], train_dataY[:,1], train_dataY[:,2]

        train_acc0 = balanced_accuracy_score(datay0, predy0)
        train_acc1 = balanced_accuracy_score(datay1, predy1)
        train_acc2 = balanced_accuracy_score(datay2, predy2)
        
        self.log("epoch", self.current_epoch)
        self.log("train_loss_mean", train_loss_mean, prog_bar=True)

        tqdm_dict = {"epoch_train_loss": train_loss_mean, "epoch_train_acc0": train_acc0, "epoch_train_acc1": train_acc1, "epoch_train_acc2": train_acc2}
        self.wandb_run.log(tqdm_dict)
        
        self.metric_acc0.reset()   
        self.metric_acc1.reset()   
        self.metric_acc2.reset()   
        
    def on_validation_epoch_start(self, ) -> None:
        self.metric_acc0 = torchmetrics.Accuracy(task="multiclass") if self.num_labels > 2 else torchmetrics.Accuracy(task="binary")
        self.metric_acc1 = torchmetrics.Accuracy(task="multiclass") if self.num_labels > 2 else torchmetrics.Accuracy(task="binary")
        self.metric_acc2 = torchmetrics.Accuracy(task="multiclass") if self.num_labels > 2 else torchmetrics.Accuracy(task="binary")

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        inputs, targets = batch
#         print(inputs, targets)
        model_out = self.forward(**inputs)
        #print(model_out.size(), targets["labels"].size())
#         print(model_out)
        loss_val = self.loss(model_out, targets)
        
        y = targets["labels"].view(-1,3) #B3
        y0 = y[:,0]
        y1 = y[:,1]
        y2 = y[:,2]
        y_hat0 = model_out["logits0"] #(B,3);(B,3),(B,2)
        y_hat1 = model_out["logits1"] #(B,3);(B,3),(B,2)
        y_hat2 = model_out["logits2"] #(B,3);(B,3),(B,2)
        labels_hat0 = torch.argmax(y_hat0, dim=-1).to(y)
        labels_hat1 = torch.argmax(y_hat1, dim=-1).to(y)
        labels_hat2 = torch.argmax(y_hat2, dim=-1).to(y)

#         val_acc0 = self.metric_acc0(labels_hat0.detach().cpu().view(-1), y0.detach().cpu().view(-1)) #Must mount tensors to CPU;;;; ALSO, val_acc should be returned!
#         val_acc1 = self.metric_acc1(labels_hat1.detach().cpu().view(-1), y1.detach().cpu().view(-1)) #Must mount tensors to CPU;;;; ALSO, val_acc should be returned!
#         val_acc2 = self.metric_acc2(labels_hat2.detach().cpu().view(-1), y2.detach().cpu().view(-1)) #Must mount tensors to CPU;;;; ALSO, val_acc should be returned!
        val_acc0 = balanced_accuracy_score(y0.detach().cpu().numpy().reshape(-1), labels_hat0.detach().cpu().numpy().reshape(-1))
        val_acc1 = balanced_accuracy_score(y1.detach().cpu().numpy().reshape(-1), labels_hat1.detach().cpu().numpy().reshape(-1))
        val_acc2 = balanced_accuracy_score(y2.detach().cpu().numpy().reshape(-1), labels_hat2.detach().cpu().numpy().reshape(-1))
        predY = np.stack([labels_hat0.detach().cpu().numpy().reshape(-1), labels_hat1.detach().cpu().numpy().reshape(-1), labels_hat2.detach().cpu().numpy().reshape(-1)]).T #data,3
        dataY = np.stack([y0.detach().cpu().numpy().reshape(-1), y1.detach().cpu().numpy().reshape(-1), y2.detach().cpu().numpy().reshape(-1)]).T #data,3
#         print(predY.shape, dataY.shape)
        
        output = {"val_loss": loss_val, "val_acc0": val_acc0, "val_acc1": val_acc1, "val_acc2": val_acc2} #NEVER USE ORDEREDDICT!!!!
        self.log("val_loss", loss_val, prog_bar=True)

        self.wandb_run.log(output)

        return {"val_loss": loss_val, "val_acc0": val_acc0, "val_acc1": val_acc1, "val_acc2": val_acc2, "predY": predY, "dataY": dataY} #NEVER USE ORDEREDDICT!!!!

        
    def on_validation_epoch_end(self, outputs: list) -> dict:
        if not self.trainer.sanity_checking:
            val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
            val_predY = np.concatenate([x['predY'] for x in outputs], axis=0)
            val_dataY = np.concatenate([x['dataY'] for x in outputs], axis=0)
            predy0, predy1, predy2 = val_predY[:,0], val_predY[:,1], val_predY[:,2]
            datay0, datay1, datay2 = val_dataY[:,0], val_dataY[:,1], val_dataY[:,2]
            val_acc0 = balanced_accuracy_score(datay0, predy0)
            val_acc1 = balanced_accuracy_score(datay1, predy1)
            val_acc2 = balanced_accuracy_score(datay2, predy2)
            
            self.log("val_loss_mean", val_loss_mean, prog_bar=True)
            #For ModelCheckpoint Metric, something is wrong with using numbers at the end of string: e.g. epoch_val_acc0
            #https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/callbacks/model_checkpoint.html#:~:text=filename%20%3D%20filename.replace(group%2C%20f%22%7B%7B0%5B%7Bname%7D%5D%22)
            self.log("epoch_val_acc_A", val_acc0, prog_bar=True)
            self.log("epoch_val_acc_B", val_acc1, prog_bar=True)
            self.log("epoch_val_acc_C", val_acc2, prog_bar=True)
            self.log("epoch", self.current_epoch, prog_bar=True)

            tqdm_dict = {"epoch_val_loss": val_loss_mean, "epoch_val_acc0": val_acc0, "epoch_val_acc1": val_acc1, "epoch_val_acc2": val_acc2}

            self.wandb_run.log(tqdm_dict)
            self.metric_acc0.reset()   
            self.metric_acc1.reset()   
            self.metric_acc2.reset()   

    def on_test_epoch_start(self, ) -> None:
        self.metric_acc0 = torchmetrics.Accuracy(task="multiclass") if self.num_labels > 2 else torchmetrics.Accuracy(task="binary")
        self.metric_acc1 = torchmetrics.Accuracy(task="multiclass") if self.num_labels > 2 else torchmetrics.Accuracy(task="binary")
        self.metric_acc2 = torchmetrics.Accuracy(task="multiclass") if self.num_labels > 2 else torchmetrics.Accuracy(task="binary")
        
    def test_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:

        inputs, targets = batch
        model_out = self.forward(**inputs)
        #print(model_out.size(), targets["labels"].size())
        loss_test = self.loss(model_out, targets)
        
        y = targets["labels"].view(-1,3) #B3
        y0 = y[:,0]
        y1 = y[:,1]
        y2 = y[:,2]
        y_hat0 = model_out["logits0"] #(B,3);(B,3),(B,2)
        y_hat1 = model_out["logits1"] #(B,3);(B,3),(B,2)
        y_hat2 = model_out["logits2"] #(B,3);(B,3),(B,2)
        labels_hat0 = torch.argmax(y_hat0, dim=-1).to(y)
        labels_hat1 = torch.argmax(y_hat1, dim=-1).to(y)
        labels_hat2 = torch.argmax(y_hat2, dim=-1).to(y)

#         test_acc0 = self.metric_acc0(labels_hat0.detach().cpu().view(-1), y0.detach().cpu().view(-1)) #Must mount tensors to CPU;;;; ALSO, val_acc should be returned!
#         test_acc1 = self.metric_acc1(labels_hat1.detach().cpu().view(-1), y1.detach().cpu().view(-1)) #Must mount tensors to CPU;;;; ALSO, val_acc should be returned!
#         test_acc2 = self.metric_acc2(labels_hat2.detach().cpu().view(-1), y2.detach().cpu().view(-1)) #Must mount tensors to CPU;;;; ALSO, val_acc should be returned!
        test_acc0 = balanced_accuracy_score(y0.detach().cpu().numpy().reshape(-1), labels_hat0.detach().cpu().numpy().reshape(-1))
        test_acc1 = balanced_accuracy_score(y1.detach().cpu().numpy().reshape(-1), labels_hat1.detach().cpu().numpy().reshape(-1))
        test_acc2 = balanced_accuracy_score(y2.detach().cpu().numpy().reshape(-1), labels_hat2.detach().cpu().numpy().reshape(-1))
        predY = np.stack([labels_hat0.detach().cpu().numpy().reshape(-1), labels_hat1.detach().cpu().numpy().reshape(-1), labels_hat2.detach().cpu().numpy().reshape(-1)]).T #data,3
        dataY = np.stack([y0.detach().cpu().numpy().reshape(-1), y1.detach().cpu().numpy().reshape(-1), y2.detach().cpu().numpy().reshape(-1)]).T #data,3

        output = {"test_loss": loss_test, "test_acc0": test_acc0, "test_acc1": test_acc1, "test_acc2": test_acc2} #NEVER USE ORDEREDDICT!!!!
#         self.log("test_loss", loss_test, prog_bar=True)

        self.wandb_run.log(output)
        
        return {"test_loss": loss_test, "test_acc0": test_acc0, "test_acc1": test_acc1, "test_acc2": test_acc2, "predY": predY, "dataY": dataY}

    def on_test_epoch_end(self, outputs: list) -> dict:

        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_predY = np.concatenate([x['predY'] for x in outputs], axis=0)
        test_dataY = np.concatenate([x['dataY'] for x in outputs], axis=0)
        predy0, predy1, predy2 = test_predY[:,0], test_predY[:,1], test_predY[:,2]
        datay0, datay1, datay2 = test_dataY[:,0], test_dataY[:,1], test_dataY[:,2]
        test_acc0 = balanced_accuracy_score(datay0, predy0)
        test_acc1 = balanced_accuracy_score(datay1, predy1)
        test_acc2 = balanced_accuracy_score(datay2, predy2)
        
        self.log("test_loss_mean", test_loss_mean, prog_bar=True)
        tqdm_dict = {"epoch_test_loss": test_loss_mean, "epoch_test_acc0": test_acc0, "epoch_test_acc1": test_acc1, "epoch_test_acc2": test_acc2}
        
        self.wandb_run.log(tqdm_dict)
        self.metric_acc0.reset()   
        self.metric_acc1.reset()   
        self.metric_acc2.reset()   
        
        artifact = wandb.Artifact(name="finetune", type="torch_model")
        path_and_name = os.path.join(self.hparam.load_model_directory, self.hparam.load_model_checkpoint)
        artifact.add_file(str(path_and_name)) #which directory's file to add; when downloading it downloads directory/file
        self.wandb_run.log_artifact(artifact)

    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        parameters = [
            {"params": self.head.parameters()},
            {
                "params": self.model.parameters(), 
            },
        ]
        if self.hparam.optimizer == "adafactor":
            optimizer = Adafactor(parameters, relative_step=True)
        elif self.hparam.optimizer == "adamw":
            optimizer = torch.optim.AdamW(parameters, lr=self.hparam.learning_rate)
        total_training_steps = len(self.train_dataloader()) * self.hparam.max_epochs
        warmup_steps = total_training_steps // self.hparam.warm_up_split
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def tokenizing(self, stage="train"):
        x = []
        for i in range(len(self.dataset)):
            x.append(' '.join(self.dataset.iloc[i,1])) #AA Sequence
        proper_inputs = x #List[seq] of no. of elements (B,)
    
        inputs = self.tokenizer.batch_encode_plus(proper_inputs,
                                          add_special_tokens=True,
                                          padding=True,
                                          truncation=True, return_tensors="pt",
                                          max_length=self.hparam.max_length) #Tokenize inputs as a dict type of Tensors
        targets = self.dataset.iloc[:,2:].values #list type including nans; (B,3)
        targets = torch.from_numpy(targets).view(len(targets), -1).long() #target is originally list -> change to Tensor (B,1)

        dataset = dl.SequenceDataset(inputs, targets)
        train, val = torch.utils.data.random_split(dataset, self._get_split_sizes(self.hparam.train_frac, dataset),
                                                                generator=torch.Generator().manual_seed(0))
        
        if stage == "train":
            dataset = train
        elif stage == "val":
            dataset = val
        elif stage == "test":
            dataset = val
            
        return dataset #torch Dataset

    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        self._train_dataset = self.tokenizing(stage="train")
        return DataLoader(
            dataset=self._train_dataset,
            sampler=torch.utils.data.RandomSampler(self._train_dataset),
            batch_size=self.hparam.batch_size,
            num_workers=self.hparam.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        self._dev_dataset = self.tokenizing(stage="test")
        return DataLoader(
            dataset=self._dev_dataset,
            batch_size=self.hparam.batch_size,
            num_workers=self.hparam.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        self._test_dataset = self.tokenizing(stage="test")
        return DataLoader(
            dataset=self._test_dataset,
            batch_size=self.hparam.batch_size,
            num_workers=self.hparam.num_workers,
        )
