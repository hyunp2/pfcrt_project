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
import model as Model
import dataset as dl
from typing import *


#https://github.com/HelloJocelynLu/t5chem/blob/main/t5chem/archived/MultiTask.py for more info


def get_args():
    parser = argparse.ArgumentParser(description='Training')

    #Model related
    parser.add_argument('--load-model-directory', "-dirm", type=str, default="/Scr/hyunpark/DL_Sequence_Collab/output", help='This is where model is/will be located...')  
    parser.add_argument('--load-model-checkpoint', "-ckpt", type=str, default=None, help='which checkpoint...')  
    parser.add_argument('--model-name', type=str, default='Rostlab/prot_bert', help='HUGGINGFACE Backbone model name card')

    #Molecule (Dataloader) related
    parser.add_argument('--load-data-directory', "-dird", default="/Scr/hyunpark/DL_Sequence_Collab/data", help='This is where data is located...')  
    parser.add_argument('--dataset', type=str, default="yarongef/human_proteome_triplets", help='pass dataset...')  

    #Optimizer related
    parser.add_argument('--max-epochs', default=60, type=int, help='number of epochs max')
    parser.add_argument('--min-epochs', default=1, type=int, help='number of epochs min')
    parser.add_argument('--batch-size', '-b', default=2048, type=int, help='batch size')
    parser.add_argument('--learning-rate', '-lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--ngpus', type=int, default=-1, help='Number of GPUs, -1 use all available. Use CUDA_VISIBLE_DEVICES=1, to decide gpus')
    parser.add_argument('--num-nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--warm-up-split', type=int, default=5, help='warmup times')
    parser.add_argument('--scheduler', type=str, default="cosine", help='scheduler type')
    parser.add_argument('--accelerator', "-accl", type=str, default="gpu", help='accelerator type')

    #Misc.
    parser.add_argument('--seed', type=int, default=42, help='seeding number')
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32], help='Floating point precision')
    parser.add_argument('--monitor', type=str, default="val_acc_mean", help='metric to watch')
    parser.add_argument('--loss', '-l', type=str, default="classification", choices=['classification', 'contrastive', 'ner'], help='loss for training')
    parser.add_argument('--save_top_k', type=int, default="5", help='num of models to save')
    parser.add_argument('--patience', type=int, default=10, help='patience for stopping')
    parser.add_argument('--metric_mode', type=str, default="max", help='mode of monitor')
    parser.add_argument('--distributed-backend', default='ddp', help='Distributed backend: dp, ddp, ddp2')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data prefetch')
    parser.add_argument('--amp-backend', type=str, default="native", help='Torch vs NVIDIA AMP')
    parser.add_argument('--max_length', type=int, default=1536, help='length for padding seqs')
    parser.add_argument('--label-smoothing', '-ls', type=float, default=0., help='CE loss regularization')
    parser.add_argument('--sanity-checks', '-sc', type=int, default=2, help='Num sanity checks..')
    parser.add_argument('--z_dim', '-zd', type=int, default=1024, help='CURL purpose.., SAME as self.encoder_features')
    parser.add_argument('--ner', '-ner', type=bool, default=False, help='NER training')
    parser.add_argument('--ner-config', '-nc', type=str, default=None, help='NER config')

    args = parser.parse_args()
    return args

def _main():
    hparams = get_args()

    pl.seed_everything(hparams.seed)

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = Model.ProtBertClassifier(hparams)

    # ------------------------
    # 2 INIT EARLY STOPPING
    # ------------------------
    early_stop_callback = pl.callbacks.EarlyStopping(
    monitor=hparams.monitor,
    min_delta=0.0,
    patience=hparams.patience,
    verbose=True,
    mode=hparams.metric_mode,
    )

    # --------------------------------
    # 3 INIT MODEL CHECKPOINT CALLBACK
    #  -------------------------------
    # initialize Model Checkpoint Saver
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filename="{epoch}-{train_acc_mean:.2f}-{val_acc_mean:.2f}",
    save_top_k=hparams.save_top_k,
    verbose=True,
    monitor=hparams.monitor,
    period=1,
    mode=hparams.metric_mode,
    dirpath=hparams.load_model_directory,
    )

    # --------------------------------
    # 4 INIT SWA CALLBACK
    #  -------------------------------
    # Stochastic Weight Averaging
    swa_callback = pl.callbacks.StochasticWeightAveraging(swa_epoch_start=0.8, swa_lrs=None, annealing_epochs=10, annealing_strategy='cos', avg_fn=None)

    # --------------------------------
    # 5 INIT SWA CALLBACK
    #  -------------------------------
    # Stochastic Weight Averaging
    #rsummary_callback = pl.callbacks.RichModelSummary() #Not in this PL version

    # --------------------------------
    # 6 INIT MISC CALLBACK
    #  -------------------------------
    # MISC
    progbar_callback = pl.callbacks.ProgressBar()
    timer_callback = pl.callbacks.Timer()
    #tqdmbar_callback = pl.callbacks.TQDMProgressBar()

    # ------------------------
    # N INIT TRAINER
    # ------------------------
    #tb_logger = pl.loggers.TensorBoardLogger("tb_logs", name="my_model")

    trainer = pl.Trainer(
    gpus=hparams.ngpus,
    max_epochs=hparams.max_epochs,
    min_epochs=hparams.min_epochs,
    callbacks = [early_stop_callback, checkpoint_callback, swa_callback, progbar_callback, timer_callback],
    precision=hparams.precision,
    amp_backend=hparams.amp_backend,
    deterministic=False,
    default_root_dir=hparams.load_model_directory,
    num_sanity_val_steps = hparams.sanity_checks,
    resume_from_checkpoint=os.path.join(hparams.load_model_directory, hparams.load_model_checkpoint) if hparams.load_model_checkpoint else None
    )

    trainer.fit(model)

if __name__ == "__main__":
    _main()
    #CUDA_VISIBLE_DEVICES=0 python main.py -ls 0.1 -b 512 -ckpt epoch=4-val_loss=0.30-val_acc=0.94.ckpt
