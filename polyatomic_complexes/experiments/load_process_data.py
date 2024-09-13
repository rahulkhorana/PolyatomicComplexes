import torch
import dill
import joblib
import numpy as np
import pandas as pd
from typing import Tuple
from gauche.dataloader import MolPropLoader
from sklearn.preprocessing import StandardScaler

import os

os.chdir("../")


class LoadDatasetForTask:
    def __init__(self, X, y, y_column, repn):
        self.X = X
        self.y = y
        self.y_column = y_column
        self.repn = repn
        self.data_root = os.getcwd() + "/polyatomic_complexes/"

    def load_photoswitches(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.repn == "complexes":
            with open(self.X, "rb") as f:
                x_data = dill.load(f)
            X = []
            for x in x_data:
                rep = x_data[x][0]
                t = torch.tensor(rep)
                X.append(t)
            max_len = max([x.squeeze().numel() for x in X])
            data = [
                torch.nn.functional.pad(
                    x, pad=(0, max_len - x.numel()), mode="constant", value=0
                )
                for x in X
            ]
            X = torch.stack(data)
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            mean_value = y.mean()
            y.fillna(value=mean_value, inplace=True)
            y = torch.tensor(y.values).view(len(y), 1)
            assert (
                len(X) == len(y)
                and isinstance(X, torch.Tensor)
                and isinstance(y, torch.Tensor)
            )
            return tuple([X, y])
        elif self.repn == "deep_complexes":
            with open(self.X, "rb") as f:
                x_data = dill.load(f)
            X = []
            for x in x_data:
                rep0 = x_data[x][0]
                rep1 = x_data[x][1]
                rep0.flatten()
                rep1.flatten()
                r = np.concatenate([rep0, rep1], axis=0)
                t = torch.tensor(r)
                X.append(t)
            max_len = max([x.squeeze().numel() for x in X])
            data = [
                torch.nn.functional.pad(
                    x, pad=(0, max_len - x.numel()), mode="constant", value=0
                )
                for x in X
            ]
            X = torch.stack(data)
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            mean_value = y.mean()
            y.fillna(value=mean_value, inplace=True)
            y = torch.tensor(y.values).view(len(y), 1)
            assert (
                len(X) == len(y)
                and isinstance(X, torch.Tensor)
                and isinstance(y, torch.Tensor)
            )
            return tuple([X, y])
        elif self.repn == "stacked_complexes":
            with open(self.X, "rb") as f:
                x_data = dill.load(f)
            X = []
            for x in x_data:
                X.append(x_data[x][0][0])
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            mean_value = y.mean()
            y.fillna(value=mean_value, inplace=True)
            y = torch.tensor(y.values).view(len(y), 1)
            assert len(X) == len(y) and isinstance(y, torch.Tensor)
            return tuple([X, y])
        elif self.repn == "fingerprints":
            loader = MolPropLoader()
            loader.validate = lambda: False
            loader.load_benchmark(
                "Photoswitch",
                path=self.data_root + "dataset/photoswitches/photoswitches.csv",
            )
            loader.featurize("ecfp_fragprints")
            X = loader.features
            X = torch.from_numpy(X).type(torch.float64)
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            mean_value = y.mean()
            y.fillna(value=mean_value, inplace=True)
            y = torch.tensor(y.values).view(len(y), 1)
            assert (
                len(X) == len(y)
                and isinstance(X, torch.Tensor)
                and isinstance(y, torch.Tensor)
            )
            return tuple([X, y])
        elif self.repn == "SELFIES":
            loader = MolPropLoader()
            loader.validate = lambda: False
            loader.load_benchmark(
                "Photoswitch",
                path=self.data_root + "dataset/photoswitches/photoswitches.csv",
            )
            loader.featurize("bag_of_selfies")
            X = loader.features
            X = torch.from_numpy(X).type(torch.float64)
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            mean_value = y.mean()
            y.fillna(value=mean_value, inplace=True)
            y = torch.tensor(y.values).view(len(y), 1)
            assert (
                len(X) == len(y)
                and isinstance(X, torch.Tensor)
                and isinstance(y, torch.Tensor)
            )
            return tuple([X, y])
        elif self.repn == "GRAPHS":
            loader = MolPropLoader()
            loader.validate = lambda: False
            loader.load_benchmark(
                "Photoswitch",
                path=self.data_root + "dataset/photoswitches/photoswitches.csv",
            )
            loader.featurize("molecular_graphs")
            X = loader.features
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            mean_value = y.mean()
            y.fillna(value=mean_value, inplace=True)
            y = torch.tensor(y.values).view(len(y), 1)
            assert len(X) == len(y) and isinstance(y, torch.Tensor)
            return tuple([X, y])
        elif self.repn == "SMILES":
            loader = MolPropLoader()
            loader.validate = lambda: False
            loader.load_benchmark(
                "Photoswitch",
                path=self.data_root + "dataset/photoswitches/photoswitches.csv",
            )
            loader.featurize("bag_of_smiles")
            X = loader.features
            X = torch.from_numpy(X).type(torch.float64)
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            mean_value = y.mean()
            y.fillna(value=mean_value, inplace=True)
            y = torch.tensor(y.values).view(len(y), 1)
            assert len(X) == len(y) and isinstance(y, torch.Tensor)
            return tuple([X, y])

    def load_esol(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.repn == "complexes":
            with open(self.X, "rb") as f:
                x_data = dill.load(f)
            X = []
            for x in x_data:
                rep = x_data[x][0]
                t = torch.tensor(rep)
                X.append(t)
            max_len = max([x.squeeze().numel() for x in X])
            data = [
                torch.nn.functional.pad(
                    x, pad=(0, max_len - x.numel()), mode="constant", value=0
                )
                for x in X
            ]
            X = torch.stack(data)
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            y = torch.tensor(y.values).view(len(y), 1)
            assert (
                len(X) == len(y)
                and isinstance(X, torch.Tensor)
                and isinstance(y, torch.Tensor)
            )
            return tuple([X, y])
        elif self.repn == "deep_complexes":
            with open(self.X, "rb") as f:
                x_data = dill.load(f)
            X = []
            for x in x_data:
                rep0 = x_data[x][0]
                rep1 = x_data[x][1]
                rep0.flatten()
                rep1.flatten()
                r = np.concatenate([rep0, rep1], axis=0)
                t = torch.tensor(r)
                X.append(t)
            max_len = max([x.squeeze().numel() for x in X])
            data = [
                torch.nn.functional.pad(
                    x, pad=(0, max_len - x.numel()), mode="constant", value=0
                )
                for x in X
            ]
            X = torch.stack(data)
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            y = torch.tensor(y.values).view(len(y), 1)
            assert (
                len(X) == len(y)
                and isinstance(X, torch.Tensor)
                and isinstance(y, torch.Tensor)
            )
            return tuple([X, y])
        elif self.repn == "stacked_complexes":
            with open(self.X, "rb") as f:
                x_data = dill.load(f)
            X = []
            for x in x_data:
                X.append(x_data[x][0][0])
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            y = torch.tensor(y.values).view(len(y), 1)
            assert len(X) == len(y) and isinstance(y, torch.Tensor)
            return tuple([X, y])
        elif self.repn == "fingerprints":
            loader = MolPropLoader()
            loader.load_benchmark("ESOL", path=self.data_root + "dataset/esol/ESOL.csv")
            loader.featurize("ecfp_fragprints")
            X = loader.features
            X = torch.from_numpy(X).type(torch.float64)
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            y = torch.tensor(y.values).view(len(y), 1)
            assert (
                len(X) == len(y)
                and isinstance(X, torch.Tensor)
                and isinstance(y, torch.Tensor)
            )
            return tuple([X, y])
        elif self.repn == "SELFIES":
            loader = MolPropLoader()
            loader.load_benchmark("ESOL", path=self.data_root + "dataset/esol/ESOL.csv")
            loader.featurize("bag_of_selfies")
            X = loader.features
            X = torch.from_numpy(X).type(torch.float64)
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            y = torch.tensor(y.values).view(len(y), 1)
            assert (
                len(X) == len(y)
                and isinstance(X, torch.Tensor)
                and isinstance(y, torch.Tensor)
            )
            return tuple([X, y])
        elif self.repn == "GRAPHS":
            loader = MolPropLoader()
            loader.load_benchmark("ESOL", path=self.data_root + "dataset/esol/ESOL.csv")
            loader.featurize("molecular_graphs")
            X = loader.features
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            y = torch.tensor(y.values).view(len(y), 1)
            assert len(X) == len(y) and isinstance(y, torch.Tensor)
            return tuple([X, y])
        elif self.repn == "SMILES":
            loader = MolPropLoader()
            loader.load_benchmark("ESOL", path=self.data_root + "dataset/esol/ESOL.csv")
            loader.featurize("bag_of_smiles")
            X = loader.features
            X = torch.from_numpy(X).type(torch.float64)
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            y = torch.tensor(y.values).view(len(y), 1)
            assert (
                len(X) == len(y)
                and isinstance(X, torch.Tensor)
                and isinstance(y, torch.Tensor)
            )
            return tuple([X, y])

    def load_freesolv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.repn == "complexes":
            with open(self.X, "rb") as f:
                x_data = dill.load(f)
            X = []
            for x in x_data:
                rep = x_data[x][0]
                t = torch.tensor(rep)
                X.append(t)
            max_len = max([x.squeeze().numel() for x in X])
            data = [
                torch.nn.functional.pad(
                    x, pad=(0, max_len - x.numel()), mode="constant", value=0
                )
                for x in X
            ]
            X = torch.stack(data)
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            y = torch.tensor(y.values).view(len(y), 1)
            assert (
                len(X) == len(y)
                and isinstance(X, torch.Tensor)
                and isinstance(y, torch.Tensor)
            )
            return tuple([X, y])
        elif self.repn == "deep_complexes":
            with open(self.X, "rb") as f:
                x_data = dill.load(f)
            X = []
            for x in x_data:
                rep0 = x_data[x][0]
                rep1 = x_data[x][1]
                rep0.flatten()
                rep1.flatten()
                r = np.concatenate([rep0, rep1], axis=0)
                t = torch.tensor(r)
                X.append(t)
            max_len = max([x.squeeze().numel() for x in X])
            data = [
                torch.nn.functional.pad(
                    x, pad=(0, max_len - x.numel()), mode="constant", value=0
                )
                for x in X
            ]
            X = torch.stack(data)
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            y = torch.tensor(y.values).view(len(y), 1)
            assert (
                len(X) == len(y)
                and isinstance(X, torch.Tensor)
                and isinstance(y, torch.Tensor)
            )
            return tuple([X, y])
        elif self.repn == "stacked_complexes":
            with open(self.X, "rb") as f:
                x_data = dill.load(f)
            X = []
            for x in x_data:
                X.append(x_data[x][0][0])
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            y = torch.tensor(y.values).view(len(y), 1)
            assert len(X) == len(y) and isinstance(y, torch.Tensor)
            return tuple([X, y])
        elif self.repn == "fingerprints":
            loader = MolPropLoader()
            loader.load_benchmark(
                "FreeSolv", path=self.data_root + "dataset/free_solv/FreeSolv.csv"
            )
            loader.featurize("ecfp_fragprints")
            X = loader.features
            X = torch.from_numpy(X).type(torch.float64)
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            y = torch.tensor(y.values).view(len(y), 1)
            assert (
                len(X) == len(y)
                and isinstance(X, torch.Tensor)
                and isinstance(y, torch.Tensor)
            )
            return tuple([X, y])
        elif self.repn == "SELFIES":
            loader = MolPropLoader()
            loader.load_benchmark(
                "FreeSolv", path=self.data_root + "dataset/free_solv/FreeSolv.csv"
            )
            loader.featurize("bag_of_selfies")
            X = loader.features
            X = torch.from_numpy(X).type(torch.float64)
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            y = torch.tensor(y.values).view(len(y), 1)
            assert (
                len(X) == len(y)
                and isinstance(X, torch.Tensor)
                and isinstance(y, torch.Tensor)
            )
            return tuple([X, y])
        elif self.repn == "GRAPHS":
            loader = MolPropLoader()
            loader.load_benchmark(
                "FreeSolv", path=self.data_root + "dataset/free_solv/FreeSolv.csv"
            )
            loader.featurize("molecular_graphs")
            X = loader.features
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            y = torch.tensor(y.values).view(len(y), 1)
            assert len(X) == len(y) and isinstance(y, torch.Tensor)
            return tuple([X, y])
        elif self.repn == "SMILES":
            loader = MolPropLoader()
            loader.load_benchmark(
                "FreeSolv", path=self.data_root + "dataset/free_solv/FreeSolv.csv"
            )
            loader.featurize("bag_of_smiles")
            X = loader.features
            X = torch.from_numpy(X).type(torch.float64)
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            y = torch.tensor(y.values).view(len(y), 1)
            assert (
                len(X) == len(y)
                and isinstance(X, torch.Tensor)
                and isinstance(y, torch.Tensor)
            )
            return tuple([X, y])

    def load_lipophilicity(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.repn == "complexes":
            with open(self.X, "rb") as f:
                x_data = dill.load(f)
            X = []
            for x in x_data:
                rep = x_data[x][0]
                t = torch.tensor(rep)
                X.append(t)
            max_len = max([x.squeeze().numel() for x in X])
            data = [
                torch.nn.functional.pad(
                    x, pad=(0, max_len - x.numel()), mode="constant", value=0
                )
                for x in X
            ]
            X = torch.stack(data)
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            y = torch.tensor(y.values).view(len(y), 1)
            assert (
                len(X) == len(y)
                and isinstance(X, torch.Tensor)
                and isinstance(y, torch.Tensor)
            )
            return tuple([X, y])
        elif self.repn == "deep_complexes":
            with open(self.X, "rb") as f:
                x_data = dill.load(f)
            X = []
            for x in x_data:
                rep0 = x_data[x][0]
                rep1 = x_data[x][1]
                rep0.flatten()
                rep1.flatten()
                r = np.concatenate([rep0, rep1], axis=0)
                t = torch.tensor(r)
                X.append(t)
            max_len = max([x.squeeze().numel() for x in X])
            data = [
                torch.nn.functional.pad(
                    x, pad=(0, max_len - x.numel()), mode="constant", value=0
                )
                for x in X
            ]
            X = torch.stack(data)
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            y = torch.tensor(y.values).view(len(y), 1)
            assert (
                len(X) == len(y)
                and isinstance(X, torch.Tensor)
                and isinstance(y, torch.Tensor)
            )
            return tuple([X, y])
        elif self.repn == "stacked_complexes":
            with open(self.X, "rb") as f:
                x_data = dill.load(f)
            X = []
            for x in x_data:
                X.append(x_data[x][0][0])
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            y = torch.tensor(y.values).view(len(y), 1)
            assert len(X) == len(y) and isinstance(y, torch.Tensor)
            return tuple([X, y])
        elif self.repn == "fingerprints":
            loader = MolPropLoader()
            loader.load_benchmark(
                "Lipophilicity",
                path=self.data_root + "dataset/lipophilicity/Lipophilicity.csv",
            )
            loader.featurize("ecfp_fragprints")
            X = loader.features
            X = torch.from_numpy(X).type(torch.float64)
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            y = torch.tensor(y.values).view(len(y), 1)
            assert (
                len(X) == len(y)
                and isinstance(X, torch.Tensor)
                and isinstance(y, torch.Tensor)
            )
            return tuple([X, y])
        elif self.repn == "SELFIES":
            loader = MolPropLoader()
            loader.load_benchmark(
                "Lipophilicity",
                path=self.data_root + "dataset/lipophilicity/Lipophilicity.csv",
            )
            loader.featurize("bag_of_selfies")
            X = loader.features
            X = torch.from_numpy(X).type(torch.float64)
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            y = torch.tensor(y.values).view(len(y), 1)
            assert (
                len(X) == len(y)
                and isinstance(X, torch.Tensor)
                and isinstance(y, torch.Tensor)
            )
            return tuple([X, y])
        elif self.repn == "GRAPHS":
            loader = MolPropLoader()
            loader.load_benchmark(
                "Lipophilicity",
                path=self.data_root + "dataset/lipophilicity/Lipophilicity.csv",
            )
            loader.featurize("molecular_graphs")
            X = loader.features
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            y = torch.tensor(y.values).view(len(y), 1)
            assert len(X) == len(y) and isinstance(y, torch.Tensor)
            return tuple([X, y])
        elif self.repn == "SMILES":
            loader = MolPropLoader()
            loader.load_benchmark(
                "Lipophilicity",
                path=self.data_root + "dataset/lipophilicity/Lipophilicity.csv",
            )
            loader.featurize("bag_of_smiles")
            X = loader.features
            X = torch.from_numpy(X).type(torch.float64)
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            y = torch.tensor(y.values).view(len(y), 1)
            assert (
                len(X) == len(y)
                and isinstance(X, torch.Tensor)
                and isinstance(y, torch.Tensor)
            )
            return tuple([X, y])

    def load_mp(self):
        if self.repn == "complexes":
            with open(self.X, "rb") as f:
                x_data = dill.load(f)
            X = []
            for i, x in enumerate(x_data):
                rep = x_data[x][0]
                t = torch.tensor(rep).view((len(rep), 1))
                X.append(t.flatten(0))
            max_len = max([x.squeeze().numel() for x in X])
            data = [
                torch.nn.functional.pad(
                    x, pad=(0, max_len - x.numel()), mode="constant", value=0
                )
                for x in X
            ]
            X = torch.stack(data)
            X = torch.linalg.vector_norm(X, ord=2, dim=(-1))
            X = X.view(len(X), 1)
            ydata = pd.read_csv(self.y, low_memory=False)
            y = ydata[self.y_column]
            y = torch.tensor(y.values, dtype=torch.float32).view(len(y), 1)
            assert (
                len(X) == len(y)
                and isinstance(X, torch.Tensor)
                and isinstance(y, torch.Tensor)
            )
            return tuple([X, y])
        elif self.repn == "stacked_complexes":
            with open(self.X, "rb") as f:
                x_data = joblib.load(f)
            X = []
            for x in x_data:
                X.append(x_data[x][0][0])
            ydata = pd.read_csv(self.y, low_memory=False)
            y = ydata[self.y_column]
            y = torch.tensor(y.values, dtype=torch.float32).view(len(y), 1)
            assert len(X) == len(y) and isinstance(y, torch.Tensor)
            return tuple([X, y])

    def load_matbench(self):
        if self.repn == "complexes":
            with open(self.X, "rb") as f:
                x_data = dill.load(f)
            X = []
            ydata = pd.read_csv(self.y, low_memory=False)
            y = ydata[self.y_column]
            y_data = []
            for pair in zip(x_data, y):
                x, yi = pair
                try:
                    yc = np.float32(yi)
                    y_data.append(yc)
                except Exception:
                    continue
                rep = x_data[x][0]
                t = torch.tensor(rep).view((len(rep), 1))
                X.append(t.flatten(0))

            max_len = max([x.squeeze().numel() for x in X])
            data = [
                torch.nn.functional.pad(
                    x, pad=(0, max_len - x.numel()), mode="constant", value=0
                )
                for x in X
            ]
            X = torch.stack(data)
            X = torch.linalg.vector_norm(X, ord=2, dim=(-1))
            X = X.view(len(X), 1)
            y = torch.tensor(y_data, dtype=torch.float32).view(len(y_data), 1)
            assert (
                len(X) == len(y)
                and isinstance(X, torch.Tensor)
                and isinstance(y, torch.Tensor)
            )
            return tuple([X, y])
        elif self.repn == "stacked_complexes":
            with open(self.X, "rb") as f:
                x_data = dill.load(f)
            X = []
            for x in x_data:
                X.append(x_data[x][0][0])
            ydata = pd.read_csv(self.y, low_memory=False)
            y = ydata[self.y_column]
            y = torch.tensor(y.values, dtype=torch.float32).view(len(y), 1)
            assert len(X) == len(y) and isinstance(y, torch.Tensor)
            return tuple([X, y])

    def load_jdft2d(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.repn == "complexes":
            with open(self.X, "rb") as f:
                x_data = dill.load(f)
            X = []
            for x in x_data:
                rep = x_data[x][0]
                t = torch.tensor(rep)
                X.append(t)
            max_len = max([x.squeeze().numel() for x in X])
            data = [
                torch.nn.functional.pad(
                    x, pad=(0, max_len - x.numel()), mode="constant", value=0
                )
                for x in X
            ]
            X = torch.stack(data)
            X = X.numpy()
            X = StandardScaler().fit_transform(X)
            X = torch.from_numpy(X)
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            y = torch.tensor(y.values).view(len(y), 1)
            assert (
                len(X) == len(y)
                and isinstance(X, torch.Tensor)
                and isinstance(y, torch.Tensor)
            )
            return tuple([X, y])
        elif self.repn == "deep_complexes":
            with open(self.X, "rb") as f:
                x_data = dill.load(f)
            X = []
            for x in x_data:
                rep0 = x_data[x][0]
                rep1 = x_data[x][1]
                rep1 = np.asarray(rep1)
                rep0.flatten()
                rep1.flatten()
                r = np.concatenate([rep0, rep1], axis=0)
                t = torch.tensor(r)
                X.append(t)
            max_len = max([x.squeeze().numel() for x in X])
            data = [
                torch.nn.functional.pad(
                    x, pad=(0, max_len - x.numel()), mode="constant", value=0
                )
                for x in X
            ]
            X = torch.stack(data)
            X = X.numpy()
            X = StandardScaler().fit_transform(X)
            X = torch.from_numpy(X)
            ydata = pd.read_csv(self.y)
            y = ydata[self.y_column]
            y = torch.tensor(y.values).view(len(y), 1)
            assert (
                len(X) == len(y)
                and isinstance(X, torch.Tensor)
                and isinstance(y, torch.Tensor)
            )
            return tuple([X, y])
        elif self.repn == "stacked_complexes":
            with open(self.X, "rb") as f:
                x_data = dill.load(f)
            X = []
            for x in x_data:
                X.append(x_data[x][0][0])
            ydata = pd.read_csv(self.y, low_memory=False)
            y = ydata[self.y_column]
            y = torch.tensor(y.values, dtype=torch.float32).view(len(y), 1)
            assert len(X) == len(y) and isinstance(y, torch.Tensor)
            return tuple([X, y])
