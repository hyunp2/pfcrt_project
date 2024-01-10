import torch
# import pytorch_lightning as pl
import lightning as L
import transformers
import numpy as np
import matplotlib.pyplot as plt
import pdb
from bertviz import head_view, model_view
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
import dataset as dl
import model as Model
import finetune as FModel
from typing import *
import MDAnalysis as mda
import sklearn.manifold
import deprecation
from captum.attr import visualization

#https://github.com/HelloJocelynLu/t5chem/blob/main/t5chem/archived/MultiTask.py for more info

class ModelAnalyzer(object):
    def __init__(self, hparam: argparse.ArgumentParser):
        """Load trained classifier"""
        self.hparam = hparam
        self.model = Model.ProtBertClassifier.load_from_checkpoint(checkpoint_path=os.path.join(hparam.load_model_directory, \
                                                                                                hparam.load_model_checkpoint), hparam=hparam) \
                                                                                                if not self.hparam.finetune \
                                                                                                else FModel.ProtBertClassifierFinetune.load_from_checkpoint(checkpoint_path=os.path.join(hparam.load_model_directory, \
                                                                                                hparam.load_model_checkpoint), hparam=hparam)
        self.model.freeze()
        self.tokenizer = self.model.tokenizer
        self.dmo_train = self.model.train_dataloader() #Test dataloader instance
        self.dmo_test = self.model.test_dataloader() #Test dataloader instance

        wandb.init(project="DL_Sequence_Collab", entity="hyunp2")
        #self.test_dataset = self.model._test_dataset #Original Test Data"SET" instance (different from dataloader ... called after calling dmo)

    @deprecation.deprecated(deprecated_in="0.0", removed_in="0.0",
                         current_version="0.0",
                        details="Use truncated version...")
    def show_head_view(self, model, tokenizer, sequence):
        """Attention of letters to each per Layer;; use truncated version"""
        inputs = tokenizer.batch_encode_plus(sequence, return_tensors='pt', add_special_tokens=True)
        input_ids = inputs['input_ids']
        attention = model(input_ids)[-1] #Get last
        input_id_list = input_ids[0].tolist() # Batch index 0
        tokens = tokenizer.convert_ids_to_tokens(input_id_list)    
        head_view(attention, tokens)

    def get_predictions(self, ):
        """call test_step function of pl.LightningModule"""
        csv_logger = L.pytorch.loggers.CSVLogger(save_dir=self.hparam.load_model_directory)
        trainer = L.Trainer(devices=self.hparam.ngpus, accelerator=self.hparam.accelerator, strategy=self.hparam.strategy, logger=[csv_logger])

        w0, w1, w2 = self.hparam.loss_weights
        for c, w in enumerate([w0, w1, w2]):
            if w == 1:
                break
        assert int(self.hparam.load_model_directory[-1]) == c, "directory last digit must match loss_weights non-zero index"
        
        trainer.predict(self.model, [self.dmo_train, self.dmo_test]) #Get loss and acc
        #[Jan 9th 2024] git pull && python -m utils --ngpus 1 --accelerator gpu --strategy auto -b 512 --finetune --loss_weights 0 1 0 --load_model_directory output_finetune_l1 --load_model_checkpoint best_pretrained.ckpt

    def get_highlights(self, attribute="saliency", normed=False, inputs: torch.Tensor =None, tgt_idx: int = 0, additional_forward_args = None):
        """Most important residues are highlighted
        WIP!!! do this again"""
        assert inputs.size(0) == tgt_idx.size(0), "Input token ids and target labels must have the same batch dimension..."
        assert inputs.size(0) == additional_forward_args.get("token_type_ids").size(0) and inputs.size(0) == additional_forward_args.get("attention_mask").size(0), "Input token ids and type ids and masks must have the same batch dimension..."

        hparam = self.hparam
        vis_list = list()
        if attribute == "saliency":
            from captum.attr import Saliency
            highlight = Saliency(self.model)
        elif attribute == "integrated_gradients":
            from captum.attr import IntegratedGradients
            highlight = IntegratedGradients(self.model)  
        elif attribute == "layer_integrated_gradients":
            from captum.attr import LayerIntegratedGradients
            highlight = LayerIntegratedGradients(self.model, self.model.model.embeddings)  

        #if len(inputs) > 1: idx = np.random.randint(len(inputs))
        for idx in range(len(inputs)):
            inputs, tgt_idx, tk_id, attn_mask = list(map(lambda inp: inp[idx:idx+1], (inputs, tgt_idx, additional_forward_args[0], additional_forward_args[1] ))) #only one sample chosen;; must be done for BERT visualization!
            additional_forward_args = (tk_id, attn_mask, additional_forward_args[-1]) #lambda_fn mapped results + False arg passing...
        
            if attribute == "layer_integrated_gradients":
                assert len(inputs) == 1 and len(tgt_idx) == 1, "Only one sequence info can be used..."
            else:
                pass
            results = highlight.attribute(inputs, target=tgt_idx, additional_forward_args=additional_forward_args, return_convergence_delta=True) #return_convergence_delta is a must-pass arg
            attr = results[0] if len(results) == 2 else results #if result returns errors then tuple of length 2
            delta = results[1]        

            vis, vis_dict = self._layer_attr_logic(attr, delta, inputs, tgt_idx, additional_forward_args) #not a list yet
            vis_list.append(vis) ###appending individual VisualizationDataRecord
        self._visualize_layer_attr_logic(vis_list) #vis_list is a  list!

        if normed and (attribute in ["saliency", "integrated_gradients"]):
            return attr.norm(p=2, dim=-1).detach().cpu().numpy()  #B,L
        elif not normed and (attribute in ["saliency", "integrated_gradients"]):
            return attr.detach().cpu().numpy()        
        
    def _layer_attr_logic(self, attr: "L by D for ONE sequence from captum", delta: "error from captum", inputs: torch.Tensor =None, tgt_idx: int = 0, additional_forward_args = None):
        #Compute visualizer
        #https://captum.ai/tutorials/Bert_SQUAD_Interpret#:~:text=start_position_vis%20%3D%20viz.VisualizationDataRecord(
        from captum.attr import visualization
        attributions = attr.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions) * 3
        vis_dict = dict()
        vis_dict["convergence_score"] = delta
        pred, pred_idx = self.model(inputs, *additional_forward_args).softmax(dim=-1).max(-1)       
        pred = pred[0].item(); pred_idx = pred_idx[0].item(); tgt_idx = tgt_idx[0].item()
        vis_dict["pred_prob"] = pred
        vis_dict["pred_class"] = pred_idx
        vis_dict["true_class"] = tgt_idx
        vis_dict["attr_class"] = str(tgt_idx)
        #vis_dict["raw_input_ids"] = list(filter(lambda inp: inp != "[PAD]", self.tokenizer.convert_ids_to_tokens(inputs[0].tolist()) ))
        vis_dict["raw_input_ids"] = self.tokenizer.convert_ids_to_tokens(inputs[0].tolist()) #list of decoded tokens
        vis_dict["word_attributions"] = attributions        
        vis_dict["attr_score"] = attributions.sum()
        vis = visualization.VisualizationDataRecord(**vis_dict) #datarecords instance (not in doc!)
        return vis, vis_dict #not a list yet

    def _visualize_layer_attr_logic(self, vis: List[visualization.VisualizationDataRecord]):
        assert isinstance(vis, list), "vis must be an instance of list of VisualizationDataRecord..."
        html = visualization.visualize_text([*vis]) #one or multple sequence visualization!!
        path_to_captum_html = os.path.join(self.hparam.load_model_directory, "captum_figure.html")
        data = html.data
        with open(path_to_captum_html, 'w') as f:
            f.write(data)
        table = wandb.Table(columns = ["captum_figure"])
        table.add_data(wandb.Html(path_to_captum_html))
        wandb.log({"Captum Importance": table})

    def show_head_view(self, vis_dict: dict, inputs: torch.Tensor =None, tgt_idx: int = 0, additional_forward_args=None, filename: str=None):
        assert isinstance(vis_dict, dict), "vis_dict must be only one dictionary of ONE index..."
        tokens_array_nopads, attention = self._truncate_pad_and_attn(vis_dict, inputs, tgt_idx, additional_forward_args)
        self._head_view_logic(tokens_array_nopads, attention, log=False, filename=filename)

    def show_model_view(self, vis_dict: dict, inputs: torch.Tensor =None, tgt_idx: int = 0, additional_forward_args=None, filename: str=None):
        assert isinstance(vis_dict, dict), "vis_dict must be only one dictionary of ONE index..."
        tokens_array_nopads, attention = self._truncate_pad_and_attn(vis_dict, inputs, tgt_idx, additional_forward_args)
        self._model_view_logic(tokens_array_nopads, attention, log=False, filename=filename)

    def _truncate_pad_and_attn(self, vis_dict, inputs: torch.Tensor =None, tgt_idx: int = 0, additional_forward_args = None):
        tokens_array = np.array(vis_dict["raw_input_ids"]) #numpy of decoded tokens
        tokens_index = list(np.where(tokens_array != "[PAD]")[0]) #list for index
        tokens_array_nopads = list(tokens_array[tokens_index])
        kwargs = dict(input_ids=inputs, attention_mask=additional_forward_args[1], token_type_ids=additional_forward_args[0], output_attentions=True)
        attention = self.model.model(**kwargs)["attentions"] #list of tensors
        attention = tuple(map(lambda inp: inp[...,:len(tokens_array_nopads), :len(tokens_array_nopads)], attention))
        return tokens_array_nopads, attention #truncated

    def _head_view_logic(self, tokens_array_nopads: "by _truncate_pad_and_attn", attention: "by _truncate_pad_and_attn", log: "WANDB logging" =False, filename: str=None):
        html = head_view(attention, tokens_array_nopads, html_action="return")
        path_to_head_html = os.path.join(self.hparam.load_model_directory, f"head_figure_{filename}.html")
        data = html.data
        with open(path_to_head_html, 'w') as f:
            f.write(data)
        if log:
            wandb.log({"Head+Layer Importance": wandb.Html(open(path_to_head_html))})
        sys.stdout.write(f"Head view logic head_figure_{filename}.html is HTML saved...")

    def _model_view_logic(self, tokens_array_nopads: "by _truncate_pad_and_attn", attention: "by _truncate_pad_and_attn", log: "WANDB logging" =False, filename: str=None):
        html = model_view(attention, tokens_array_nopads, display_mode="light", html_action="return")
        path_to_model_html = os.path.join(self.hparam.load_model_directory, f"model_figure_{filename}.html")
        data = html.data
        with open(path_to_model_html, 'w') as f:
            f.write(data)
        if log:
            wandb.log({"Model Importance": wandb.Html(open(path_to_model_html))})
        sys.stdout.write(f"Model view logic model_figure_{filename}.html is HTML saved...")

    def get_statistics_latents(self, ):
        """Torchmetric or WANDB classification metrics & reduce dim
        """
        trainer = pl.Trainer(gpus=self.hparam.ngpus)
        trainer.predict(self.model, self.dmo) #Get loss and acc


if __name__ == "__main__":
    from main import get_args
    hparam = get_args()
    modelanalyzer = ModelAnalyzer(hparam=hparam)
    modelanalyzer.get_predictions()
#     batch, target = next(iter(modelanalyzer.dmo))
#     input_ids = batch.pop("input_ids", None)
    #print(modelanalyzer.model.model.config.max_position_embeddings)
#     ans = modelanalyzer.get_highlights(attribute="layer_integrated_gradients", inputs=input_ids, tgt_idx=target["labels"].view(-1,), additional_forward_args=(batch["token_type_ids"], batch["attention_mask"], False))
#     modelanalyzer.get_statistics_latents()
    #python -m utils -ls 0.1 -b 512 -ckpt epoch=143-train_loss_mean=0.66-val_loss_mean=0.76.ckpt --finetune --accelerator gpu --ngpus auto

####[Jan 9th 2024]####
# export WANDB_DIR=/Scr/hyunpark/DL_Sequence_Collab/pfcrt_project
# export WANDB_CACHE_DIR=/Scr/hyunpark/DL_Sequence_Collab/pfcrt_project
# export TRANSFORMERS_CACHE=/Scr/hyunpark/DL_Sequence_Collab/pfcrt_project
# git pull && python -m utils --ngpus 1 --accelerator gpu --strategy auto -b 512 --finetune --loss_weights 0 1 0 --load_model_directory output_finetune_l1 --load_model_checkpoint best_pretrained.ckpt

