import pandas as pd
import numpy as np
import torch

class DataParser(object):
    def read_file(filename: str):
        data = pd.read_xlsx(f"{filename}")
        return data


