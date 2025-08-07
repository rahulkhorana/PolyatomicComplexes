import os
import torch
import pandas as pd
import dill
import joblib
import numpy as np
from typing import Tuple, List, Any
from sklearn.preprocessing import StandardScaler
from gauche.dataloader import MolPropLoader
from gauche import SIGP, NonTensorialInputs


class LoadDatasetForTask:
    def __init__(self, X: str, y: str, y_column: str, repn: str, project_root: str):
        self.X = X
        self.y = y
        self.y_column = y_column
        self.repn = repn
        self.data_root = project_root

    def _load_from_pkl(
        self, is_stacked: bool = False, is_deep_complex: bool = False
    ) -> Tuple[Any, torch.Tensor]:
        try:
            with open(self.X, "rb") as f:
                x_data = joblib.load(f) if "matbench" in self.X else dill.load(f)
        except FileNotFoundError:
            print(f"Error: Pickle file not found at {self.X}. Returning empty data.")
            return [], torch.empty(0, 1, dtype=torch.float32)

        X_list = []
        for x in x_data:
            if is_stacked:
                X_list.append(x_data[x][0][0])
            elif is_deep_complex:
                rep0 = x_data[x][0]
                rep1 = x_data[x][1]
                r = np.concatenate([rep0.flatten(), rep1.flatten()], axis=0)
                t = torch.tensor(r, dtype=torch.float32)
                X_list.append(t)
            else:
                rep = x_data[x][0]
                t = torch.tensor(rep, dtype=torch.float32)
                X_list.append(t)

        if is_stacked:
            X = X_list
        else:
            if X_list:
                max_len = max([t.squeeze().numel() for t in X_list])
                padded_data = [
                    torch.nn.functional.pad(
                        t, pad=(0, max_len - t.numel()), mode="constant", value=0
                    )
                    for t in X_list
                ]
                X = torch.stack(padded_data)
            else:
                X = torch.empty(0, 0)

        ydata = pd.read_csv(self.y, low_memory=False)
        y = ydata[self.y_column]

        if "photoswitches" in self.y.lower():
            mean_value = y.mean()
            y.fillna(value=mean_value, inplace=True)

        y = torch.tensor(y.values, dtype=torch.float32).view(len(y), 1)

        if len(X) != len(y):
            print(
                f"Warning: Mismatched number of samples in X ({len(X)}) and y ({len(y)})."
            )

        return X, y

    def _load_gauche_data(
        self, benchmark_name: str, featurizer_name: str
    ) -> Tuple[Any, torch.Tensor]:
        loader = MolPropLoader()
        loader.validate = lambda: False  # type: ignore
        loader.load_benchmark(benchmark_name, path=self.y)
        loader.featurize(featurizer_name)
        X = loader.features

        if featurizer_name != "molecular_graphs":
            X = torch.from_numpy(X).type(torch.float32)

        ydata = pd.read_csv(self.y, low_memory=False)
        y = ydata[self.y_column]
        y = torch.tensor(y.values, dtype=torch.float32).view(len(y), 1)

        return X, y

    def load_dataset(self, dataset_name: str) -> Tuple[Any, torch.Tensor]:
        if self.repn in ["complexes", "deep_complexes", "stacked_complexes"]:
            root_enc_path = os.path.join(
                self.data_root,
                "polyatomic_complexes",
                "dataset",
                dataset_name.lower().replace(" ", "_"),
            )
            if self.repn == "complexes":
                self.X = os.path.join(root_enc_path, "fast_complex_lookup_repn.pkl")
            elif self.repn == "deep_complexes":
                self.X = os.path.join(root_enc_path, "deep_complex_lookup_repn.pkl")
            elif self.repn == "stacked_complexes":
                self.X = os.path.join(root_enc_path, "stacked_complex_lookup_repn.pkl")

        if self.repn == "complexes":
            X, y = self._load_from_pkl()
        elif self.repn == "deep_complexes":
            X, y = self._load_from_pkl(is_deep_complex=True)
        elif self.repn == "stacked_complexes":
            X, y = self._load_from_pkl(is_stacked=True)
        elif self.repn == "fingerprints":
            X, y = self._load_gauche_data(dataset_name, "ecfp_fragprints")
        elif self.repn == "SELFIES":
            X, y = self._load_gauche_data(dataset_name, "bag_of_selfies")
        elif self.repn == "GRAPHS":
            X, y = self._load_gauche_data(dataset_name, "molecular_graphs")
        elif self.repn == "SMILES":
            X, y = self._load_gauche_data(dataset_name, "bag_of_smiles")
        else:
            raise ValueError(f"Unknown representation: {self.repn}")

        if dataset_name.lower() == "jdft2d" and isinstance(X, torch.Tensor):
            X = X.numpy()
            X = StandardScaler().fit_transform(X)
            X = torch.from_numpy(X)

        return X, y
