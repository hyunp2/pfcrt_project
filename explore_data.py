import pandas as pd
import numpy as np
import torch
import attrs, dataclasses
from typing import *
import os, sys, shutil, subprocess, pathlib

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
                                                               "PPQ Resistance", "CQ Resistance", "Fitness"]):
        return self.data.loc[:,col_names]
    
if __name__ == "__main__":
    parser = DataParser(filename="pfcrt.csv")
    data = parser.data
    data_trunc = select_columns()

    
    
    


