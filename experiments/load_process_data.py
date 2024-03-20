import os
import torch
import pandas as pd
from typing import Tuple, Optional
from torch.utils.data import TensorDataset

class LoadDatasetForTask():
    def __init__(self, Xpath:str, ypath:Optional[str], representation='mcw') -> Tuple[TensorDataset, TensorDataset]:
        self.X = []
        assert len(Xpath) > 0 and type(Xpath) is str
        self.xp = Xpath
        self.rep = representation
        if ypath != '':
            self.y = torch.load(ypath)

    def load(self):
        if self.rep == 'mcw':
            xdir = os.listdir(self.xp)
            for x in xdir:
                t = torch.load(f'{self.xp}/{x}')[0]
                self.X.append(t)
            return self.X, self.y
        elif self.rep == 'smiles':
            df = pd.read_csv(self.xp)
            if not len(df.columns) == 2:
                raise Exception("invalid experimental input")
            target = df.columns[1]
            self.X = df['smiles']
            self.y = df[target]
            return self.X, self.y
        elif self.rep == 'selfies':
            df = pd.read_csv(self.xp)
            if not len(df.columns) == 2:
                raise Exception("invalid experimental input")
            target = df.columns[1]
            self.X = df['selfies']
            self.y = df[target]
            return self.X, self.y
        else:
            print("INVALID REPRESENTATION / INPUT")
            raise Exception("unsupported")


class ProcessDataForGP():
    def __init__(self, X, y, representation='mcw'):
        self.X = X
        self.y = y
        self.rep = representation
    
    def handle_data(self):
        if self.rep == 'mcw':
            return
        elif self.rep == 'smiles':
            return
        elif self.rep == 'selfies':
            return