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
        strategy=hparams.strategy,
        accelerator=hparams.accelerator,
        devices=hparams.ngpus,
        default_root_dir=hparams.load_model_directory,
        )
    
    import dataset as ds
#     dataset = load_dataset("yarongef/human_proteome_triplets", cache_dir=hparams.load_data_directory)
    tokenizer=BertTokenizer.from_pretrained("Rostlab/prot_bert",do_lower_case=False, return_tensors="pt",cache_dir=hparams.load_model_directory)

    x = []
    for i in range(len(data_trunc)):
        x.append(' '.join(data_trunc.iloc[i,1])) #AA Sequence
    proper_inputs = x
    inputs = tokenizer.batch_encode_plus(proper_inputs,
                                      add_special_tokens=True,
                                      padding=True, truncation=True, return_tensors="pt",
                                      max_length=hparams.max_length) #Tokenize inputs as a dict type of Tensors
    targets = data_trunc.iloc[:,2:].values #list type including nans; (B,3)
    targets = torch.from_numpy(targets).view(len(targets), -1) #target is originally list -> change to Tensor (B,1)
    dataset = ds.SequenceDataset(inputs=inputs, targets=targets)
    custom_dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=hparams.batch_size)
    
    batch = iter(custom_dataloader).next()
    model.forward(batch) #BL
#     trainer.predict(model, dataloaders=custom_dataloader)
    
#     python -m explore_data --ngpus 1 --accelerator gpu --strategy ddp --load-model-checkpoint epoch=16-train_loss_mean=0.00-val_loss_mean=0.32.ckpt -b 4

    
    


