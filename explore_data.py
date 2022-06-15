import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
import attrs, dataclasses
from typing import *
import os, sys, shutil, subprocess, pathlib
from train import get_args
import model as Model

@dataclasses.dataclass
class DataParser(object):
    def __init__(self, filename:str = None):
        data = self.read_file(filename)
        self.data = data
        
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
                                                               "PPQ Resistance", "CQ Resistance", "Fitness"], drop_duplicate_on="Amino Acid Sequence"):
        if drop_duplicate_on == None:
            return self.data.loc[:,col_names]
        elif drop_duplicate_on != None:
            return self.data.loc[:,col_names].drop_duplicates(drop_duplicate_on)
        
    
if __name__ == "__main__":
    parser = DataParser(filename="pfcrt.csv")
    data = parser.data
    data_trunc = parser.select_columns()
    
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
    
    hparams = get_args()
    pl.seed_everything(hparams.seed)
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = Model.ProtBertClassifier.load_from_checkpoint( os.path.join(hparams.load_model_directory, hparams.load_model_checkpoint), hparam=hparams )
    
    trainer = pl.Trainer(
        accelerator=hparams.accelerator,
        gpus=hparams.ngpus,
        default_root_dir=hparams.load_model_directory,
        )
    trainer.predict(model)
    
#     python -m explore_data --ngpus 1 --accelerator ddp --load-model-checkpoint epoch=59-train_loss_mean=0.95-val_loss_mean=0.18.ckpt

    
    


