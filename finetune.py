import torch
import pytorch_lightning as pl
import transformers
import numpy as np
import matplotlib.pyplot as plt
import pdb
from bertviz.bertviz import head_view
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer, BertConfig, BertForSequenceClassification, get_linear_schedule_with_warmup, Adafactor, AdamW
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
# import BertNER
# import BertNERTokenizer
from model import ProtBertClassifier
from explore_data import DataParser
from focal_loss import *
import curtsies.fmtfuncs as cf
from sklearn.metrics import balanced_accuracy_score
import warnings
import argparse
#https://github.com/HelloJocelynLu/t5chem/blob/main/t5chem/archived/MultiTask.py for more info

# classifier.bert.pooler.dense.weight.requires_grad
# classifier.bert.pooler.dense.training
# classifier.classifier.weight.requires_grad
logging.basicConfig()
logger = logging.getLogger("BERT Fine-tuning")
logger.setLevel(logging.DEBUG)
warnings.simplefilter("ignore")

class ProtBertClassifier(ProtBertClassifier):
    def __init__(self, hparam: argparse.ArgumentParser) -> None:
        super(ProtBertClassifier, self).__init__(hparam=hparam)
        parser = DataParser.get_data("pfcrt.csv")
        data = parser.data
        data_trunc = parser.select_columns(fill_na=self.hparam.fillna_val)
        self.dataset = data_trunc
        print(data_trunc.shape)
        self.num_labels = 2 #; Filippo dataset
        # self.metric_acc = torchmetrics.Accuracy()
        self.z_dim = self.hparam.z_dim #Add this!
        if self.hparam.loss == "contrastive": self.register_parameter("W", torch.nn.Parameter(torch.rand(self.z_dim, self.z_dim))) #CURL purpose
        
#         self.__augment_data()
        
        # build model
        _ = self.__build_model() if not self.ner else self.__build_model_ner()

        # build weights for CE loss
        _ = self.__build_weight(self.hparam.nonuniform_weight) 
#         print(self.weight0, self.weight1, self.weight2)
        
        # Loss criterion initialization.
        _ = self.__build_loss() if not self.ner else self.__build_model_ner()

        self.freeze_encoder()
        
        print(cf.on_yellow(f"CE-loss {self.hparam.use_ce}; Non-uniform weights {self.hparam.nonuniform_weight}"))
        print(cf.on_yellow(f"Non-uniform weights are {self.weight0} and {self.weight1} and {self.weight2}"))
        print(cf.on_yellow(f"Data augmentation is applied {self.hparam.aug}..."))

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

#         self.classifier0 = nn.Sequential(
#             nn.Linear(self.encoder_features, self.encoder_features),
#             nn.BatchNorm1d(self.encoder_features),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(self.encoder_features, self.encoder_features),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(self.encoder_features, 3),
#         )
        
#         self.classifier1 = nn.Sequential(
#             nn.Linear(self.encoder_features, self.encoder_features),
#             nn.BatchNorm1d(self.encoder_features),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(self.encoder_features, self.encoder_features),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(self.encoder_features, 3),
#         )
        
#         self.classifier2 = nn.Sequential(
#             nn.Linear(self.encoder_features, self.encoder_features),
#             nn.BatchNorm1d(self.encoder_features),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(self.encoder_features, self.encoder_features),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(self.encoder_features, 2),
#         )
        
        if self.hparam.loss == "contrastive": 
            self.make_hook()

        self.wandb_run = wandb.init(project="DL_Sequence_Collab", entity="hyunp2", group="DDP_runs")
        wandb.watch(self.head)

    def make_hook(self, ):
        self.fhook = dict()
        def hook(m, i, o):
            self.fhook["encoded_feats"] = o #(B,1024)
        self.fhook_handle = self.head[0].register_forward_hook(hook) #Call Forward hook with "model.fhook["encoded_feats"]" of (B,C); for NER, it is (B,L,C)
 
    def __build_loss(self):
        """ Initializes the loss function/s. """
#         global loss_fn #fixes: AttributeError: Can't pickle local object 'ProtBertClassifier.__build_loss.<locals>.loss_fn'
#         def loss_fn(predictions: dict, targets: dict, hparam: argparse.ArgumentParser, *weight_args):
#         logits0 = predictions.get("logits0", 0)
#         logits1 = predictions.get("logits1", 0)
#         logits2 = predictions.get("logits2", 0)
#         target0 = targets.get("labels", None)[:,0].to(logits0).long()
#         target1 = targets.get("labels", None)[:,1].to(logits0).long()
#         target2 = targets.get("labels", None)[:,2].to(logits0).long()
# #         assert len(weight_args) == 3, "must pass three aurgmnents to weights..."
# #         weight0, weight1, weight2 = weight_args

        if self.hparam.use_ce:
            loss0 = nn.CrossEntropyLoss(label_smoothing=self.hparam.label_smoothing, ignore_index=self.hparam.fillna_val, weight=self.weight0) #(logits0, target0) #ignore_index=100 is from dataset!
            loss1 = nn.CrossEntropyLoss(label_smoothing=self.hparam.label_smoothing, ignore_index=self.hparam.fillna_val, weight=self.weight1) #(logits1, target1) #ignore_index=100 is from dataset!
            loss2 = nn.CrossEntropyLoss(label_smoothing=self.hparam.label_smoothing, ignore_index=self.hparam.fillna_val, weight=self.weight2) #(logits2, target2) #ignore_index=100 is from dataset!
        else:
            loss0 = FocalLoss(beta=0.9999, weight=self.weight0) #(logits0, target0) #ignore_index=100 is from dataset!
            loss1 = FocalLoss(beta=0.9999, weight=self.weight1) #(logits1, target1) #ignore_index=100 is from dataset!
            loss2 = FocalLoss(beta=0.9999, weight=self.weight2) #(logits2, target2) #ignore_index=100 is from dataset!
#         return (loss0 + loss1 + loss2).mean()
        self._loss = [loss0, loss1, loss2]

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
            
#         self.weight0 = torch.nn.functional.normalize(self.weight0, dim=-1) if self.hparam.nonuniform_weight else None
#         self.weight1 = torch.nn.functional.normalize(self.weight1, dim=-1) if self.hparam.nonuniform_weight else None
#         self.weight2 = torch.nn.functional.normalize(self.weight2, dim=-1) if self.hparam.nonuniform_weight else None
        
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
#             for param in self.head.parameters():
#                 param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """ freezes the encoder layer. """
        for param in self.model.parameters():
            param.requires_grad = False
#         for param in self.head.parameters():
#             param.requires_grad = False
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
        return result #(B,2) or (BLC)

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
        logits2 = self.classifier2(logits) #B,3
        
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

    def loss(self, predictions: dict, targets: torch.Tensor) -> torch.tensor:
        if self.hparam.loss == "classification" and not self.ner:
            loss0_fn, loss1_fn, loss2_fn = self._loss
            losses = loss0_fn(predictions["logits0"], targets["labels"][:,0].long()) \ 
                            + loss1_fn(predictions["logits1"], targets["labels"][:,1].long()) \
                            + loss2_fn(predictions["logits2"], targets["labels"][:,2].long())
            return losses.mean()
#             return self._loss(predictions, targets, self.hparam, self.weight0, self.weight1, self.weight2) #Crossentropy ;; input: dict[(B,3);(B,3);(B,2)] target dict(B,3)
#         elif self.hparam.loss == "classification" and self.ner:
#             return self._loss(predictions["logits"], targets["labels"].view(-1, self.num_labels)) #CRF ;; input (B,L,C) target (B,L) ;; B->num_frames & L->num_aa_residues & C->num_lipid_types
#         elif self.hparam.loss == "contrastive":
#             return self.compute_logits_CURL(predictions["logits"], predictions["logits"]) #Crossentropy -> Need second pred to be transformed! each pred is (B,z_dim) shape

    def on_train_epoch_start(self, ) -> None:
        self.metric_acc0 = torchmetrics.Accuracy()
        self.metric_acc1 = torchmetrics.Accuracy()
        self.metric_acc2 = torchmetrics.Accuracy()
        
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
        wandb.log(output)
#         self.log("train_loss", loss_train, prog_bar=True)
#         self.log("train_acc0", train_acc0, prog_bar=True)
#         self.log("train_acc1", train_acc1, prog_bar=True)
#         self.log("train_acc2", train_acc2, prog_bar=True)

        return {"loss": loss_train, "train_acc0": train_acc0, "train_acc1": train_acc1, "train_acc2": train_acc2, "predY": predY, "dataY": dataY}

    def training_epoch_end(self, outputs: list) -> dict:
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
        wandb.log(tqdm_dict)
        
        self.metric_acc0.reset()   
        self.metric_acc1.reset()   
        self.metric_acc2.reset()   
        
    def on_validation_epoch_start(self, ) -> None:
        self.metric_acc0 = torchmetrics.Accuracy()
        self.metric_acc1 = torchmetrics.Accuracy()
        self.metric_acc2 = torchmetrics.Accuracy()

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

        wandb.log(output)

        return {"val_loss": loss_val, "val_acc0": val_acc0, "val_acc1": val_acc1, "val_acc2": val_acc2, "predY": predY, "dataY": dataY} #NEVER USE ORDEREDDICT!!!!

        
    def validation_epoch_end(self, outputs: list) -> dict:
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

            wandb.log(tqdm_dict)
            self.metric_acc0.reset()   
            self.metric_acc1.reset()   
            self.metric_acc2.reset()   

    def on_test_epoch_start(self, ) -> None:
        self.metric_acc0 = torchmetrics.Accuracy()
        self.metric_acc1 = torchmetrics.Accuracy()
        self.metric_acc2 = torchmetrics.Accuracy()
        
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

        wandb.log(output)
        
        return {"test_loss": loss_test, "test_acc0": test_acc0, "test_acc1": test_acc1, "test_acc2": test_acc2, "predY": predY, "dataY": dataY}

    def test_epoch_end(self, outputs: list) -> dict:

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
        
        wandb.log(tqdm_dict)
        self.metric_acc0.reset()   
        self.metric_acc1.reset()   
        self.metric_acc2.reset()   
        
        artifact = wandb.Artifact(name="finetune", type="torch_model")
        path_and_name = os.path.join(self.hparam.load_model_directory, self.hparam.load_model_checkpoint)
        artifact.add_file(str(path_and_name)) #which directory's file to add; when downloading it downloads directory/file
        self.wandb_run.log_artifact(artifact)

    def on_predict_epoch_start(self, ):
        if self.hparam.loss == "classification":
            self.make_hook() #Get a hook if classification was originally trained
        if self.hparam.loss == "ner":
            self.make_hook() #Get a hook if ner was originally trained
        elif self.hparam.loss == "contrastive": 
            pass

    def predict_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:

        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss_test = self.loss(model_out, targets)
        ground_truth = targets["labels"].view(-1,) #B or (BL)
        y_hat = model_out["logits"]
        predictions = y_hat.view(-1, self.num_labels) #B,2 or (BL,C)

        logits = self.fhook["encoded_feats"] #(B,z_dim) or (B,L,z_dim)
        #import pdb; pdb.set_trace()
        
        return {"ground_truth": ground_truth, "predictions": predictions, "last_layer": logits}

    def on_predict_epoch_end(self, outputs: list) -> dict:
        ground_truth = torch.cat([x['ground_truth'] for x in outputs[0]], dim=0).contiguous().view(-1,).detach().cpu().numpy() #catted results (B,) or (BL, )
        b = ground_truth.shape[0]
        predictions = torch.cat([x['predictions'] for x in outputs[0]], dim=0).contiguous().view(b,-1).detach().cpu().numpy() #catted results ... weird outputs indexing (B, num_labels) or (BL, num_labels)
        predictions = predictions.argmax(axis = -1) #(B,) or (BL,)
        class_names = np.arange(self.num_labels).tolist() #(num_labels)
        logits = torch.cat([x['last_layer'] for x in outputs[0]], dim=0).contiguous().view(b,-1).detach().cpu().numpy() #catted results (B,reduced_dim)
        logits_ = logits #(B,zdim) or (B,L,zdim)
        if self.ner: logits_ = logits_[:,0,:] #to (B,zdim) from [CLS]
        
        self.plot_confusion(ground_truth, predictions, class_names)
        self.plot_manifold(self.hparam, logits_, ground_truth) #WIP to have residue projection as well!
        self.plot_ngl(self.hparam)
 
    @staticmethod
    def plot_confusion(ground_truth: np.ndarray, predictions: np.ndarray, class_names: np.ndarray=np.array([0,1])):
        from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
        from sklearn.calibration import CalibrationDisplay
        cm = wandb.plot.confusion_matrix(
            y_true=ground_truth,
            preds=predictions,
            class_names=class_names) 
        wandb.log({"Confusion Matrix": cm}) #Needs (B, )
        
        disp = RocCurveDisplay.from_predictions(ground_truth, predictions)
        fig = disp.figure_
        wandb.log({"ROC": fig}) #Needs (B/BL,num_labels)
        disp = PrecisionRecallDisplay.from_predictions(ground_truth, predictions)
        fig = disp.figure_
        wandb.log({"PR": fig}) #Needs (B/BL,num_labels)
#         disp = CalibrationDisplay.from_predictions(ground_truth, predictions)
#         fig = disp.figure_
#         wandb.log({"Calibration": fig}) #Needs (B/BL,num_labels)

    @staticmethod
    def plot_manifold(hparam: argparse.ArgumentParser, logits_: np.ndarray, ground_truth: np.ndarray):
        #WIP for PCA or UMAP or MDS
        #summary is 
        import sklearn.manifold
        import plotly.express as px
        tsne = sklearn.manifold.TSNE(2)
        logits_tsne = tsne.fit_transform(logits_) #(B,2) of tsne
        path_to_plotly_html = os.path.join(hparam.load_model_directory, "plotly_figure.html")
        fig = px.scatter(x=logits_tsne[:,0], y=logits_tsne[:,1], color=ground_truth)
        fig.write_html(path_to_plotly_html, auto_play = False)
        table = wandb.Table(columns = ["plotly_figure"])
        table.add_data(wandb.Html(path_to_plotly_html))
        wandb.log({"TSNE Plot": table})

    @staticmethod
    def plot_ngl(hparam: argparse.ArgumentParser):
        #WIP for filename!
        import nglview as nv
        import MDAnalysis as mda
        universe = mda.Universe("/Scr/hyunpark/ZIKV_ConcatDCD_for_DL/REDO/data/alanine-dipeptide-nowater_charmmgui.psf", "/Scr/hyunpark/ZIKV_ConcatDCD_for_DL/REDO/data/alanine-dipeptide-nowater_charmmgui.pdb")   
        u = universe.select_atoms("all")
        w = nv.show_mdanalysis(u)
        w.clear_representations()
        #w.add_licorice(selection="protein")
        w.add_representation('licorice', selection='all', color='blue')
        w.render_image(factor=1, frame=3, transparent=False)
        path_to_ngl_html = os.path.join(hparam.load_model_directory, "nglview_figure.html")
        nv.write_html(path_to_ngl_html, [w])     
        table = wandb.Table(columns = ["nglview_figure"])
        table.add_data(wandb.Html(path_to_ngl_html))
        wandb.log({"NGL View": table}) 

    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        parameters = [
            {"params": self.head.parameters(), "lr": self.hparam.learning_rate*0.1},
            {"params": self.model.parameters()},
            {"params": self.classifier0.parameters()},
            {"params": self.classifier1.parameters()},
            {"params": self.classifier2.parameters()}
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
#         optimizer = {"optimizer": optimizer, "frequency": 1}
        #https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#:~:text=is%20shown%20below.-,lr_scheduler_config,-%3D%20%7B%0A%20%20%20%20%23%20REQUIRED
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1} #Every step/epoch with Frequency 1etc by monitoring val_loss if needed

        return [optimizer], [scheduler]

    @staticmethod
    def _get_split_sizes(train_frac: float, full_dataset: torch.utils.data.Dataset) -> Tuple[int, int, int]:
        """DONE: Need to change split schemes!"""
        len_full = len(full_dataset)
        len_train = int(len_full * train_frac)
#         len_test = int(0.1 * len_full)
        len_val = len_full - len_train #- len_test
        return len_train, len_val #, len_test  
    
    def __augment_data_index(self, tmp_dataloader: DataLoader):
        inputs, targets = iter(tmp_dataloader).next()
        self.make_hook()
        input_ids = inputs["input_ids"].to(self.device)
        token_type_ids = inputs["token_type_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        out = self.forward(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask) #Called only once after loading ckpt!
#         print(self.fhook)
        ext = self.fhook["encoded_feats"] #B,dim
        
        from imblearn.combine import SMOTEENN, SMOTETomek
        from imblearn.over_sampling import RandomOverSampler
        smote_enn = RandomOverSampler(random_state=0)
        X = ext.detach().cpu().numpy() #B,dim
        y = targets["labels"][:, self.hparam.basis].detach().cpu().numpy() #0,1,2 columns (chosen 0 bc of extreme imbalance)
        X_resampled, y_resampled = smote_enn.fit_resample(X, y)
        return smote_enn.sample_indices_ #(OverN,)
    
    @staticmethod
    def __augment_data(tmp_dataloader: DataLoader, os_indices: np.ndarray):
        inputs, targets = iter(tmp_dataloader).next() #Dict, Dict
        inputs["input_ids"] = inputs["input_ids"][os_indices]
        inputs["token_type_ids"] = inputs["token_type_ids"][os_indices]
        inputs["attention_mask"] = inputs["attention_mask"][os_indices]
        targets = targets["labels"][os_indices] #Should be a Tensor
        aug_dataset = dl.SequenceDataset(inputs, targets)
        return aug_dataset
    
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
        if self.hparam.aug:
            tmp_dataloader = DataLoader(
                dataset=train,
                shuffle=None,
                batch_size=len(train),
                num_workers=self.hparam.num_workers,
            )
            os_indices = self.__augment_data_index(tmp_dataloader)
            print(cf.on_yellow(f"Original Train data size is {len(train)}, and augmented to {os_indices.shape[0]}"))
            train = self.__augment_data(tmp_dataloader, os_indices)
    #         print(aug_train)
        
        if stage == "train":
            dataset = train
        elif stage == "val":
            dataset = val
        elif stage == "test":
            dataset = val
            
        return dataset #torch Dataset

    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        return super().train_dataloader()

    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        return super().val_dataloader()

    def test_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        return super().test_dataloader()


# python -m train --ngpus "auto" --accelerator gpu --strategy none -b 16 --finetune -ckpt ckpt_for_finetune.ckpt --patience 100 --train_frac 0.8 --use_ce --aug
