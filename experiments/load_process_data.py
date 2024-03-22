import os
import torch
import dill
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from torch.utils.data import TensorDataset

class LoadDatasetForTask():
    def __init__(self, X, y, repn):
        self.X = X
        self.y = y
        self.repn = repn
    
    def load(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.repn == 'complexes':
            with open(self.X, 'rb') as f:
                x_data = dill.load(f)
            X = []
            for x in x_data:
                rep = x_data[x][0]
                t = torch.tensor(rep)
                X.append(t)
            max_len = max([x.squeeze().numel() for x in X])
            data = [torch.nn.functional.pad(x, pad=(0, max_len - x.numel()), mode='constant', value=0) for x in X]
            X = torch.stack(data)
            ydata = pd.read_csv(self.y)
            y = ydata['E isomer n-pi* wavelength in nm']
            mean_value = y.mean()
            y.fillna(value=mean_value, inplace=True)
            y = torch.tensor(y.values).view(len(y), 1)
            assert len(X) == len(y) and isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor)
            return tuple([X, y])


if __name__ == '__main__':
    ds = LoadDatasetForTask(X='dataset/photoswitches/fast_complex_lookup_repn.pkl', y='dataset/photoswitches/photoswitches.csv', repn='complexes')
    res = ds.load()