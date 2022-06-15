import pandas as pd
import numpy as np
import torch
import attrs
from typing import *

@attrs.define
class DataParser(object):
    def __init__(self, filename:str = None):
        self.data = self.read_file(filename)
        
    @classmethod
    def get_data(cls, filename):
        return cls(filename)
        
    @staticmethod
    def read_file(filename: str):
        data = pd.read_excel(f"{filename}")
        return data
    
    @staticmethod
    def select_columns(df: pd.DataFrame, col_names: List[str]=["PfCRT Isoform", "Amino Acid Sequence", 
                                                               "PPQ Resistance", "CQ Resistance", "Fitness"]):
        return df.loc[:,col_names]
    
    

    
    
    


