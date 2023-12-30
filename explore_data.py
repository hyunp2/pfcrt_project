import pandas as pd
import numpy as np
import torch
import lightning as L
import attrs, dataclasses
from typing import *
import os, sys, shutil, subprocess, pathlib
# from train import get_args
import argparse
# import model as Model
# import finetune as FModel
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer, BertConfig, BertForSequenceClassification, get_linear_schedule_with_warmup, Adafactor, AdamW


@dataclasses.dataclass
class DataParser(object):
    filename: str = dataclasses.field(default=None)
    
    def __post_init__(self, ):
        self.data = DataParser.read_file(self.filename)
        
    @classmethod
    def get_data(cls, filename):
        return cls(filename)
        
    @staticmethod
    def read_file(filename: str):
        ext = os.path.splitext(filename)[-1]
        if ext in [".xlsx"]:
            data = pd.read_excel(f"{filename}")
        elif ext in [".csv"]:
            data = pd.read_csv(f"{filename}")        
        return data
    
    def select_columns(self, col_names: List[str]=["PfCRT Isoform", "Amino Acid Sequence", 
                                                               "PPQ Resistance", "CQ Resistance", "Fitness"], drop_duplicate_on="Amino Acid Sequence", fill_na=None, dropna=True):
        if drop_duplicate_on == None:
            data = self.data.loc[:,col_names]
            data.fillna(value=fill_na, inplace=True)
            data.dropna(axis=0, inplace=dropna)
            return data
        elif drop_duplicate_on != None:
            data = self.data.loc[:,col_names].drop_duplicates(drop_duplicate_on)
            data.fillna(value=fill_na, inplace=True) 
            data.dropna(axis=0, inplace=dropna)
            return data
        
    
if __name__ == "__main__":
    parser = DataParser(filename="pfcrt.xlsx")
    data = parser.data
#     data_trunc = parser.select_columns()
#     print(data_trunc)
    data_trunc = parser.select_columns(fill_na=100) #arbitrary ignore val
    print(data_trunc)

    # data_trunc.describe()
    #        PPQ Resistance  CQ Resistance    Fitness
    # count       48.000000      58.000000  44.000000
    # mean         0.458333       0.465517   0.590909
    # std          0.824062       0.706251   0.497350
    # min          0.000000       0.000000   0.000000
    # 25%          0.000000       0.000000   0.000000
    # 50%          0.000000       0.000000   1.000000
    # 75%          0.250000       1.000000   1.000000
    # max          2.000000       2.000000   1.000000
    
    """
    def get_args():
        parser = argparse.ArgumentParser(description='Training')

        #Model related
        parser.add_argument('--load-model-directory', "-dirm", type=str, default="/Scr/hyunpark/DL_Sequence_Collab/pfcrt_project/output", help='This is where model is/will be located...')  
        parser.add_argument('--load-model-checkpoint', "-ckpt", type=str, default=None, help='which checkpoint...')  
        parser.add_argument('--model-name', type=str, default='Rostlab/prot_bert', help='HUGGINGFACE Backbone model name card')
        parser.add_argument('--finetune', action="store_true")

        #Molecule (Dataloader) related
        parser.add_argument('--load-data-directory', "-dird", default="/Scr/hyunpark/DL_Sequence_Collab/data", help='This is where data is located...')  
        parser.add_argument('--dataset', type=str, default="yarongef/human_proteome_triplets", help='pass dataset...')  

        #Optimizer related
        parser.add_argument('--optimizer', default="adamw", type=str, help='optimizer')
        parser.add_argument('--max-epochs', default=60, type=int, help='number of epochs max')
        parser.add_argument('--min-epochs', default=1, type=int, help='number of epochs min')
        parser.add_argument('--batch-size', '-b', default=2048, type=int, help='batch size')
        parser.add_argument('--learning-rate', '-lr', default=1e-3, type=float, help='learning rate')
        parser.add_argument('--ngpus', default="auto", help='Number of GPUs, -1 use all available. Use CUDA_VISIBLE_DEVICES=1, to decide gpus')
        parser.add_argument('--num-nodes', type=int, default=1, help='Number of nodes')
        parser.add_argument('--warm-up-split', type=int, default=5, help='warmup times')
        parser.add_argument('--scheduler', type=str, default="cosine", help='scheduler type')
        parser.add_argument('--accelerator', "-accl", type=str, default="gpu", help='accelerator type', choices=["cpu","gpu","tpu"])
        parser.add_argument('--strategy', "-st", default="ddp", help='accelerator type', choices=["ddp_spawn","ddp","dp","ddp2","horovod","none"])

        #Misc.
        parser.add_argument('--seed', type=int, default=42, help='seeding number')
        parser.add_argument('--precision', type=int, default=32, choices=[16, 32], help='Floating point precision')
        parser.add_argument('--monitor', type=str, default="val_loss_mean", help='metric to watch')
        parser.add_argument('--loss', '-l', type=str, default="classification", choices=['classification', 'contrastive', 'ner'], help='loss for training')
        parser.add_argument('--save_top_k', type=int, default="5", help='num of models to save')
        parser.add_argument('--patience', type=int, default=10, help='patience for stopping')
        parser.add_argument('--metric_mode', type=str, default="min", help='mode of monitor')
        parser.add_argument('--distributed-backend', default='ddp', help='Distributed backend: dp, ddp, ddp2')
        parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data prefetch')
        parser.add_argument('--amp-backend', type=str, default="native", help='Torch vs NVIDIA AMP')
        parser.add_argument('--max_length', type=int, default=1536, help='length for padding seqs')
        parser.add_argument('--label-smoothing', '-ls', type=float, default=0., help='CE loss regularization')
        parser.add_argument('--sanity-checks', '-sc', type=int, default=2, help='Num sanity checks..')
        parser.add_argument('--z_dim', '-zd', type=int, default=1024, help='CURL purpose.., SAME as self.encoder_features')
        parser.add_argument('--ner', '-ner', type=bool, default=False, help='NER training')
        parser.add_argument('--ner-config', '-nc', type=str, default=None, help='NER config')
        parser.add_argument('--fillna-val', '-fv', type=int, default=100, help='Dataset ignore index')
        parser.add_argument('--train_frac', type=float, default=0.8, help='data split')
        parser.add_argument('--nonuniform_weight', action="store_true", help='Weighted CE loss')
        parser.add_argument('--use_ce', action="store_true", help='CE vs Focal loss')

        args = parser.parse_args()
        return args
    hparams = get_args()
    
    pl.seed_everything(hparams.seed)
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    hparams.load_model_checkpoint = "epoch=53-train_loss_mean=0.00-val_loss_mean=0.19.ckpt" #"epoch=16-train_loss_mean=0.00-val_loss_mean=0.32.ckpt"
    hparams.batch_size = 4
    
    model = Model.ProtBertClassifier.load_from_checkpoint( os.path.join(hparams.load_model_directory, hparams.load_model_checkpoint), hparam=hparams )
#     model = FModel.ProtBertClassifier.load_from_checkpoint( os.path.join(hparams.load_model_directory, hparams.load_model_checkpoint), hparam=hparams, strict=False )

#     trainer = pl.Trainer(
#         strategy=hparams.strategy,
#         accelerator=hparams.accelerator,
#         devices=hparams.ngpus,
#         default_root_dir=hparams.load_model_directory,
#         )
    
    import dataset as ds
#     dataset = load_dataset("yarongef/human_proteome_triplets", cache_dir=hparams.load_data_directory)
    tokenizer=BertTokenizer.from_pretrained("Rostlab/prot_bert",do_lower_case=False, return_tensors="pt", cache_dir=hparams.load_model_directory)

    x = []
    for i in range(len(data_trunc)):
        x.append(' '.join(data_trunc.iloc[i,1])) #AA Sequence
    proper_inputs = x
    inputs = tokenizer.batch_encode_plus(proper_inputs,
                                      add_special_tokens=True,
                                      padding=True, truncation=True, return_tensors="pt",
                                      max_length=hparams.max_length) #Tokenize inputs as a dict type of Tensors
    targets = data_trunc.iloc[:,2:].values #list type including nans; (B,3)
    targets = torch.from_numpy(targets).view(len(targets), -1).long() #target is originally list -> change to Tensor (B,1)
    
    
#     valid_targets = (targets < hparams.fillna_val) #B,3
# #     print(valid_targets.any(dim=-1), targets, targets.size())
    
#     valid_targets0 = targets[valid_targets[:,0]][:,0].to(targets) #only for targ0
#     valid_targets1 = targets[valid_targets[:,1]][:,1].to(targets) #only for targ1
#     valid_targets2 = targets[valid_targets[:,2]][:,2].to(targets) #only for targ2
# #     print(valid_targets0, valid_targets1, valid_targets2)

#     weight0 = (1 / (torch.nn.functional.one_hot(valid_targets0).sum(dim=0) / valid_targets0.size(0) + torch.finfo(torch.float32).eps)).to(targets)
#     weight1 = (1 / (torch.nn.functional.one_hot(valid_targets1).sum(dim=0) / valid_targets1.size(0) + torch.finfo(torch.float32).eps)).to(targets)
#     weight2 = (1 / (torch.nn.functional.one_hot(valid_targets2).sum(dim=0) / valid_targets2.size(0) + torch.finfo(torch.float32).eps)).to(targets)
# #     print(weight0, weight1, weight2)
    
    
    dataset = ds.SequenceDataset(inputs=inputs, targets=targets)
    custom_dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=hparams.batch_size)
    
    batch = iter(custom_dataloader).next()
    inputs_, targets_ = batch
#     print(inputs_, targets_)
    out = model.forward(**inputs_) #B3
#     print(out)
#     print(model.loss(out, targets_))
    
    model.make_hook()
    dll = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=len(dataset))
    inputs_, targets_ = iter(dll).next()
    model.forward(**inputs_)
    ext = model.fhook["encoded_feats"]
    print(ext, ext.shape)
#     trainer.predict(model, dataloaders=custom_dataloader)
    
#     python -m explore_data --ngpus 1 --accelerator gpu --strategy ddp --load-model-checkpoint epoch=16-train_loss_mean=0.00-val_loss_mean=0.32.ckpt -b 4

    
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.over_sampling import RandomOverSampler
    smote_enn = RandomOverSampler(random_state=0)
    X = ext.detach().cpu().numpy() #B,dim
    y = targets_["labels"][:,0].detach().cpu().numpy()
#     print(X.shape, y.shape)
    X_resampled, y_resampled = smote_enn.fit_resample(X, y)
#     print(X.shape, X_resampled.shape)
    print(smote_enn.sample_indices_)
    """

