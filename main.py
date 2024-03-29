import torch
# import pytorch_lightning as pl
import lightning as L
import transformers
import numpy as np
import matplotlib.pyplot as plt
import pdb
from bertviz import head_view
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer, BertConfig, BertForSequenceClassification
# from pytorch_lightning.plugins import DDPPlugin
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
import finetune as FModel
import dataset as dl
from typing import *
import pathlib


#https://github.com/HelloJocelynLu/t5chem/blob/main/t5chem/archived/MultiTask.py for more info


def get_args():
    parser = argparse.ArgumentParser(description='Training')

    #Model related
    parser.add_argument('--load_model_directory', "-dirm", type=str, default="output", help='This is where model is/will be located...')  
    parser.add_argument('--load_model_checkpoint', "-ckpt", type=str, default=None, help='which checkpoint...')  
    parser.add_argument('--artifact_checkpoint', "-ackpt", type=str, default=None,  
                        help='name to save/load atfifact checkpoint..., this is the name of SAVED model (e.g. epoch=50-val_loss=20.ckpt) py PL')  
    parser.add_argument('--model_name', type=str, default='Rostlab/prot_bert', help='HUGGINGFACE Backbone model name card')
    parser.add_argument('--finetune', action="store_true")
    parser.add_argument('--pred', action="store_true")

    #Molecule (Dataloader) related
    parser.add_argument('--load_data_directory', "-dird", default="data", help='This is where data is located...')  
    parser.add_argument('--dataset', type=str, default="yarongef/human_proteome_triplets", help='pass dataset...')  

    #Optimizer related
    parser.add_argument('--optimizer', default="adamw", type=str, help='optimizer')
    parser.add_argument('--max_epochs', default=60, type=int, help='number of epochs max')
    parser.add_argument('--min_epochs', default=1, type=int, help='number of epochs min')
    parser.add_argument('--batch_size', '-b', default=2048, type=int, help='batch size')
    parser.add_argument('--learning_rate', '-lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--ngpus', default="auto", help='Number of GPUs, -1 use all available. Use CUDA_VISIBLE_DEVICES=1, to decide gpus')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--warm_up_split', type=int, default=5, help='warmup times')
    parser.add_argument('--scheduler', type=str, default="cosine", help='scheduler type')
    parser.add_argument('--accelerator', "-accl", type=str, default="gpu", help='accelerator type', choices=["cpu","gpu","tpu"])
    parser.add_argument('--strategy', "-st", default="ddp", help='accelerator type', choices=["ddp_spawn","ddp","dp","ddp2","horovod","none", "auto"]) 
    parser.add_argument('--loss_weights', type=float, nargs="*", required=True, help='which loss to ignore during finetuing')

    #Misc.
    parser.add_argument('--seed', type=int, default=42, help='seeding number')
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32], help='Floating point precision')
    parser.add_argument('--monitor', type=str, default="val_acc_mean", help='metric to watch')
    parser.add_argument('--loss', '-l', type=str, default="classification", choices=['classification', 'contrastive', 'ner'], help='loss for training')
    parser.add_argument('--save_top_k', type=int, default="5", help='num of models to save')
    parser.add_argument('--patience', type=int, default=10, help='patience for stopping')
    parser.add_argument('--metric_mode', type=str, default="max", help='mode of monitor')
    parser.add_argument('--distributed_backend', default='ddp', help='Distributed backend: dp, ddp, ddp2')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data prefetch')
    parser.add_argument('--amp_backend', type=str, default="native", help='Torch vs NVIDIA AMP')
    parser.add_argument('--max_length', type=int, default=1536, help='length for padding seqs')
    parser.add_argument('--label_smoothing', '-ls', type=float, default=0., help='CE loss regularization')
    parser.add_argument('--sanity_checks', '-sc', type=int, default=2, help='Num sanity checks..')
    parser.add_argument('--z_dim', '-zd', type=int, default=1024, help='CURL purpose.., SAME as self.encoder_features')
    parser.add_argument('--ner', '-ner', type=bool, default=False, help='NER training')
    parser.add_argument('--ner_config', '-nc', type=str, default=None, help='NER config')
    parser.add_argument('--fillna_val', '-fv', type=int, default=100, help='Dataset ignore index')
    parser.add_argument('--train_frac', type=float, default=0.8, help='data split')
    parser.add_argument('--nonuniform_weight', action="store_true", help='Weighted CE loss')
    parser.add_argument('--use_ce', action="store_true", help='CE vs Focal loss')
    parser.add_argument('--debias', action="store_true", help='Balance out 0s and 1s')
    parser.add_argument('--basis', type=int, default=0, choices=[0,1,2], help='Which target to use')
    parser.add_argument('--aug', action="store_true", help='Use data augmentation')
    parser.add_argument('--log', action="store_true", help='To log into W&B or not')


    args = parser.parse_args()
    return args

def _main():
    hparams = get_args()

    L.seed_everything(hparams.seed)

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = Model.ProtBertClassifier(hparams) if not hparams.finetune else FModel.ProtBertClassifierFinetune.load_from_checkpoint( os.path.join(hparams.load_model_directory, hparams.load_model_checkpoint), hparam=hparams, strict=False )

    # ------------------------
    # 2 INIT EARLY STOPPING
    # ------------------------
    early_stop_callback = L.pytorch.callbacks.EarlyStopping(
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
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
    filename="{epoch}-{val_acc_mean:.2f}" if "acc" in hparams.monitor else "{epoch}-{val_loss_mean:.2f}",
    save_top_k=hparams.save_top_k,
    verbose=True,
    monitor=hparams.monitor,
    every_n_epochs=1,
    mode=hparams.metric_mode,
    dirpath=hparams.load_model_directory,
    )

    # --------------------------------
    # 4 INIT SWA CALLBACK
    #  -------------------------------
    # Stochastic Weight Averaging
    swa_callback = L.pytorch.callbacks.StochasticWeightAveraging(swa_epoch_start=0.8, swa_lrs=0.001, annealing_epochs=10, annealing_strategy='cos', avg_fn=None)

    # --------------------------------
    # 5 INIT SWA CALLBACK
    #  -------------------------------
    # Stochastic Weight Averaging
    #rsummary_callback = L.callbacks.RichModelSummary() #Not in this PL version

    # --------------------------------
    # 6 INIT MISC CALLBACK
    #  -------------------------------
    # MISC
#     progbar_callback = L.callbacks.ProgressBar()
    timer_callback = L.pytorch.callbacks.Timer()
    tqdmbar_callback = L.pytorch.callbacks.TQDMProgressBar()

    # ------------------------
    # N INIT TRAINER
    # ------------------------
#     tb_logger = L.loggers.TensorBoardLogger("tb_logs", name="my_model")
    csv_logger = L.pytorch.loggers.CSVLogger(save_dir=hparams.load_model_directory)
#     plugins = DDPPlugin(find_unused_parameters=False) if hparams.accelerator == "ddp" else None
    
    
    # ------------------------
    # MISC.
    # ------------------------
    if hparams.load_model_checkpoint and not hparams.finetune:
        resume_ckpt = os.path.join(hparams.load_model_directory, hparams.load_model_checkpoint)
    elif hparams.load_model_checkpoint and hparams.finetune:
        # resume_ckpt = None
        resume_ckpt = os.path.join(hparams.load_model_directory, hparams.load_model_checkpoint)
    else:
        resume_ckpt = None
        
    if hparams.strategy in ["none", None]:
        hparams.strategy = None

    pathlib.Path(hparams.load_model_directory).mkdir(exist_ok=True)
    
    trainer = L.Trainer(
        logger=[csv_logger],
        max_epochs=hparams.max_epochs,
        min_epochs=hparams.min_epochs,
        callbacks = [early_stop_callback, checkpoint_callback, swa_callback, tqdmbar_callback, timer_callback],
        precision=hparams.precision,
        deterministic=False,
        default_root_dir=hparams.load_model_directory,
        num_sanity_val_steps = hparams.sanity_checks,
        log_every_n_steps=4,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.,
        devices=hparams.ngpus,
        strategy=hparams.strategy,
        accelerator=hparams.accelerator,
    )

    trainer.fit(model) #, ckpt_path=resume_ckpt) #New API!

if __name__ == "__main__":
    _main()
    #CUDA_VISIBLE_DEVICES=0 python -m train -ls 0.1 -b 512 -ckpt epoch=4-val_loss=0.30-val_acc=0.94.ckpt
    #python -m train --ngpus "auto" --accelerator gpu --strategy ddp -b 512 
    #CUDA_VISIBLE_DEVICES=0 python -m train -ls 0.1 -b 8 --ngpus "auto" --accelerator gpu --strategy none --finetune -ckpt ckpt_for_finetune.ckpt
    #python -m train --ngpus "auto" --accelerator gpu --strategy ddp -b 512 
    #python -m train --ngpus "auto" --accelerator gpu --strategy none -b 8 --finetune -ckpt ckpt_for_finetune.ckpt --use_ce --nonuniform_weight

    #[Dec 19th 2023] git pull && python -m main --ngpus auto --accelerator gpu --strategy auto -b 512 --loss_weights 0 0 0
    #[Dec 30th 2023] git pull && python -m main --ngpus auto --accelerator gpu --strategy auto -b 512 --finetune --loss_weights 0 1 0 --load_model_directory output_finetune --load_model_checkpoint best_pretrained.ckpt
