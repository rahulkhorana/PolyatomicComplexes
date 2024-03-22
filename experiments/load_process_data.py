import torch
import dill
import pandas as pd
from typing import Tuple
from gauche.dataloader import MolPropLoader

class LoadDatasetForTask():
    def __init__(self, X, y, y_column, repn):
        self.X = X
        self.y = y
        self.y_column = y_column
        self.repn = repn
    
    def load_photoswitches(self) -> Tuple[torch.Tensor, torch.Tensor]:
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
            y = ydata[self.y_column]
            mean_value = y.mean()
            y.fillna(value=mean_value, inplace=True)
            y = torch.tensor(y.values).view(len(y), 1)
            assert len(X) == len(y) and isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor)
            return tuple([X, y])
        elif self.repn == 'fingerprints':
            loader = MolPropLoader()
            loader.validate = lambda: False
            loader.load_benchmark("Photoswitch")
            loader.featurize('ecfp_fragprints')
            X = loader.features
            X = torch.from_numpy(X).type(torch.float64)
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            mean_value = y.mean()
            y.fillna(value=mean_value, inplace=True)
            y = torch.tensor(y.values).view(len(y), 1)
            assert len(X) == len(y) and isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor)
            return tuple([X, y])
        elif self.repn == 'SELFIES':
            loader = MolPropLoader()
            loader.validate = lambda: False
            loader.load_benchmark("Photoswitch")
            loader.featurize('bag_of_selfies')
            X = loader.features
            X = torch.from_numpy(X).type(torch.float64)
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            mean_value = y.mean()
            y.fillna(value=mean_value, inplace=True)
            y = torch.tensor(y.values).view(len(y), 1)
            assert len(X) == len(y) and isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor)
            return tuple([X, y])
        elif self.repn == 'GRAPHS':
            loader = MolPropLoader()
            loader.validate = lambda: False
            loader.load_benchmark("Photoswitch")
            loader.featurize('molecular_graphs')
            X = loader.features
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            mean_value = y.mean()
            y.fillna(value=mean_value, inplace=True)
            y = torch.tensor(y.values).view(len(y), 1)
            assert len(X) == len(y) and isinstance(y, torch.Tensor)
            return tuple([X, y])