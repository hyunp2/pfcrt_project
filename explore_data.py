import pandas as pd
import numpy as np
import torch
import attrs

@attrs.define
class DataParser(object):
    def __init__(self, filename:str = None)
        self.data = self.read_file(filename)
        
    @classmethod
    def get_data(cls, filename):
        return cls(filename)
        
    @staticmethod
    def read_file(filename: str):
        data = pd.read_xlsx(f"{filename}")
        return data
    

    
    
    


