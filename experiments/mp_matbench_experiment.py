import os
import sys
import torch
import time
import numpy as np
from load_process_data import LoadDatasetForTask
from multiprocessing.pool import ThreadPool as Pool

# botorch specific
from botorch.models.gp_regression import ExactGP

# gpytorch specific
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

# kernels + gp
from kernels import TanimotoKernel
from gaussian_process import evaluate_model

from matplotlib import pyplot as plt

plt.switch_backend("Agg")


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
        self.covar_module = ScaleKernel(MaternKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def initialize_model(train_x: torch.Tensor, train_obj: torch.Tensor, likelihood):
    model = ExactGPModel(train_x, train_obj, likelihood).to(train_x)
    return model


def one_experiment(target, encoding, n_trials, n_iters):
    X, y = [], []
    if encoding == "complexes":
        X, y = LoadDatasetForTask(
            X="dataset/materials_project/fast_complex_lookup_repn.pkl",
            y="dataset/materials_project/materials_data.csv",
            repn=encoding,
            y_column=target,
        ).load_matbench()

    if ENCODING != "GRAPHS":
        r2_list, rmse_list, mae_list, confidence_percentiles, mae_mean, mae_std = (
            evaluate_model(
                initialize_model=initialize_model,
                n_trials=n_trials,
                n_iters=n_iters,
                test_set_size=holdout_set_size,
                X=X,
                y=y,
                figure_path=f"results/{EXPERIMENT_TYPE}/confidence_mae_model_{ENCODING}_{target}.png",
            )
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
    return mean_r2, mean_rmse, mean_mae


if __name__ == "__main__":
    EXPERIMENT_TYPE = "MatBench"
    ENCODING = "complexes"
    N_TRIALS = 20
    N_ITERS = 5
    holdout_set_size = 0.33
    # dataset processing
    X, y = [], []
    # dataset loading
    possible_target_cols = ["g_vrh", "k_voigt", "k_reuss", "k_vrh", "g_voigt", "efermi"]

    results = []

    def helper(column):
        if column == "efermi":
            global holdout_set_size
            holdout_set_size = 0.9
        print(f"column processed {column}")
        mean_r2, mean_rmse, mean_mae = one_experiment(
            column, ENCODING, N_TRIALS, N_ITERS
        )
        results.append([column, mean_r2, mean_rmse, mean_mae])
        print(f"current results {results}")

    with Pool(2) as p:
        p.map(
            func=helper,
            iterable=possible_target_cols,
        )

    if type(EXPERIMENT_TYPE) is str:
        trial_num = len(os.listdir(f"results/{EXPERIMENT_TYPE}"))
        results_path = f"results/{EXPERIMENT_TYPE}/{ENCODING}_{time.time()}.txt"

        with open(results_path, "w") as f:
            f.write(EXPERIMENT_TYPE + ":")
            f.write("\n")
            f.write(ENCODING + ":")
            for result in results:
                col, mean_r2, mean_rmse, mean_mae = result
                f.write(f"column: {col}, {mean_r2}, {mean_rmse}, {mean_mae}")
                f.write("\n")
        f.close()
        print("CONCLUDED")
