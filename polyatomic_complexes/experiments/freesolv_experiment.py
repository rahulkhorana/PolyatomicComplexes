import os
import torch
import time
import numpy as np
from .load_process_data import LoadDatasetForTask

# botorch specific
from botorch.models.gp_regression import ExactGP

# gpytorch specific
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, Kernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

# kernels + gp
from .kernels import TanimotoKernel
from .gaussian_process import evaluate_model, evaluate_graph_model
from gauche import SIGP, NonTensorialInputs
from gauche.kernels.graph_kernels import (
    ShortestPathKernel,
    VertexHistogramKernel,
    GraphletSamplingKernel,
)

from matplotlib import pyplot as plt

if torch.cuda.is_available():
    dev = "cuda:0"
    torch.cuda.empty_cache()
else:
    dev = "cpu"
device = torch.device(dev)


class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(TanimotoKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class GraphGP(SIGP):
    def __init__(self, train_x, train_y, likelihood, kernel, **kernel_kwargs):
        super().__init__(train_x, train_y, likelihood)
        self.mean = ConstantMean()
        self.covariance = kernel(**kernel_kwargs)

    def forward(self, x):
        """
        A forward pass through the model.
        """
        mean = self.mean(torch.zeros(len(x), 1)).float()
        covariance = self.covariance(x)

        # because graph kernels operate over discrete inputs it is beneficial
        # to add some jitter for numerical stability
        jitter = max(covariance.diag().mean().detach().item() * 1e-4, 1e-4)
        covariance += torch.eye(len(x)) * jitter
        return MultivariateNormal(mean, covariance)


class StackedGP(SIGP):
    def __init__(
        self,
        train_x: NonTensorialInputs,
        train_y: torch.Tensor,
        likelihood: Likelihood,
        kernel: Kernel,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean = ConstantMean()
        self.covariance = kernel()

    def forward(self, x):
        """
        A forward pass through the model.
        """
        mean = self.mean(torch.zeros(len(x), 1)).float()
        covariance = self.covariance(x)

        # because graph kernels operate over discrete inputs it is beneficial
        # to add some jitter for numerical stability
        jitter = max(covariance.diag().mean().detach().item() * 1e-4, 1e-4)
        covariance += torch.eye(len(x)) * jitter
        return MultivariateNormal(mean, covariance)


def initialize_model(train_x: torch.Tensor, train_obj: torch.Tensor, likelihood):
    model = ExactGPModel(train_x, train_obj, likelihood).to(train_x)
    return model


def initialize_graph_gp(train_x, train_obj, likelihood, kernel, **kernel_kwargs):
    model = GraphGP(train_x, train_obj, likelihood, kernel, **kernel_kwargs)
    return model


def initialize_stacked_gp(train_x, train_obj, likelihood, kernel, **kwargs):
    model = StackedGP(train_x, train_obj, likelihood, kernel)
    return model


def one_experiment(
    target,
    encoding,
    n_trials,
    n_iters,
    encoding_path=None,
    data_path=None,
    fig_path=None,
    holdout_set_size=0.33,
):
    X, y = [], []
    if data_path is None:
        data_path = os.getcwd() + "/polyatomic_complexes/dataset/free_solv/FreeSolv.csv"
    if fig_path is None:
        fig_path = (
            os.getcwd()
            + f"/polyatomic_complexes/results/FreeSolv/{encoding}_Results/confidence_mae_model_{encoding}_{target}.png"
        )
    if encoding_path is None:
        root_enc_path = os.getcwd() + "/polyatomic_complexes/dataset/free_solv"
        if encoding == "complexes":
            encoding_path = root_enc_path + "/fast_complex_lookup_repn.pkl"
        elif encoding == "deep_complexes":
            encoding_path = root_enc_path + "/deep_complex_lookup_repn.pkl"
        elif encoding == "stacked_complexes":
            encoding_path = root_enc_path + "/stacked_complex_lookup_repn.pkl"
    if encoding == "complexes":
        X, y = LoadDatasetForTask(
            X=encoding_path,
            y=data_path,
            repn=encoding,
            y_column=target,
        ).load_freesolv()
    elif encoding == "deep_complexes":
        X, y = LoadDatasetForTask(
            X=encoding_path,
            y=data_path,
            repn=encoding,
            y_column=target,
        ).load_freesolv()
    elif encoding == "stacked_complexes":
        X, y = LoadDatasetForTask(
            X=encoding_path,
            y=data_path,
            repn=encoding,
            y_column=target,
        ).load_freesolv()
    elif encoding == "fingerprints":
        X, y = LoadDatasetForTask(
            X="gauche_ecfp",
            y=data_path,
            repn=encoding,
            y_column=target,
        ).load_freesolv()
    elif encoding == "SELFIES":
        X, y = LoadDatasetForTask(
            X="gauche_selfies",
            y=data_path,
            repn=encoding,
            y_column=target,
        ).load_freesolv()
    elif encoding == "GRAPHS":
        X, y = LoadDatasetForTask(
            X="gauche_graphs",
            y=data_path,
            repn=encoding,
            y_column=target,
        ).load_freesolv()
    elif encoding == "SMILES":
        X, y = LoadDatasetForTask(
            X="gauche_smiles",
            y=data_path,
            repn=encoding,
            y_column=target,
        ).load_freesolv()

    if encoding != "GRAPHS" and encoding != "stacked_complexes":
        (
            r2_list,
            rmse_list,
            mae_list,
            crps_list,
            confidence_percentiles,
            mae_mean,
            mae_std,
        ) = evaluate_model(
            initialize_model=initialize_model,
            n_trials=n_trials,
            n_iters=n_iters,
            test_set_size=holdout_set_size,
            X=X,
            y=y,
            figure_path=fig_path,
        )
    elif encoding == "stacked_complexes":
        (
            r2_list,
            rmse_list,
            mae_list,
            crps_list,
            confidence_percentiles,
            mae_mean,
            mae_std,
        ) = evaluate_graph_model(
            initialize_stacked_gp,
            n_trials=n_trials,
            n_iters=n_iters,
            test_set_size=holdout_set_size,
            X=X,
            y=y,
            figure_path=fig_path,
            kernel=GraphletSamplingKernel,
        )
    else:
        (
            r2_list,
            rmse_list,
            mae_list,
            crps_list,
            confidence_percentiles,
            mae_mean,
            mae_std,
        ) = evaluate_graph_model(
            initialize_graph_gp,
            n_trials=n_trials,
            n_iters=n_iters,
            test_set_size=holdout_set_size,
            X=X,
            y=y,
            figure_path=fig_path,
        )

    mean_r2 = "\nmean R^2: {:.4f} +- {:.4f}".format(
        np.mean(r2_list), np.std(r2_list) / np.sqrt(len(r2_list))
    )
    mean_rmse = "mean RMSE: {:.4f} +- {:.4f}".format(
        np.mean(rmse_list), np.std(rmse_list) / np.sqrt(len(rmse_list))
    )
    mean_mae = "mean MAE: {:.4f} +- {:.4f}\n".format(
        np.mean(mae_list), np.std(mae_list) / np.sqrt(len(mae_list))
    )
    mean_crps = "mean CRPS: {:.4f} +- {:.4f}\n".format(
        np.mean(crps_list), np.std(crps_list) / np.sqrt(len(crps_list))
    )
    return mean_r2, mean_rmse, mean_mae, mean_crps


if __name__ == "__main__":
    EXPERIMENT_TYPE = "FreeSolv"
    ENCODING = "stacked_complexes"
    N_TRIALS = 20
    N_ITERS = 5
    holdout_set_size = 0.33
    # dataset processing
    X, y = [], []
    # dataset loading
    possible_target_cols = ["expt", "calc"]

    results = []

    for col in possible_target_cols:
        mean_r2, mean_rmse, mean_mae, mean_crps = one_experiment(
            col, ENCODING, N_TRIALS, N_ITERS
        )
        results.append([col, mean_r2, mean_rmse, mean_mae, mean_crps])

    if type(EXPERIMENT_TYPE) is str:
        trial_num = len(os.listdir(f"results/{EXPERIMENT_TYPE}"))
        results_path = f"results/{EXPERIMENT_TYPE}/{ENCODING}_{time.time()}.txt"

        with open(results_path, "w") as f:
            f.write(EXPERIMENT_TYPE + ":")
            f.write("\n")
            f.write(ENCODING + ":")
            for result in results:
                col, mean_r2, mean_rmse, mean_mae, mean_crps = result
                f.write(
                    f"column: {col}, {mean_r2}, {mean_rmse}, {mean_mae}, {mean_crps}"
                )
                f.write("\n")
        f.close()
        print("CONCLUDED")
