import os
import torch
import time
import numpy as np
import warnings
from typing import Tuple, List, Any, Dict, Callable
from multiprocessing.pool import ThreadPool as Pool

# gpytorch specific
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel, RFFKernel, Kernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import Likelihood, GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

# botorch specific
from botorch.models.gp_regression import ExactGP
from botorch.fit import fit_gpytorch_mll

# gauche imports
from gauche import SIGP, NonTensorialInputs
from gauche.kernels.graph_kernels import (
    GraphletSamplingKernel,
)

# Imports from other files in the same directory (non-relative)
from load_process_data import LoadDatasetForTask
from gaussian_process import evaluate_model, evaluate_graph_model
from kernels import TanimotoKernel
from metrics import CRPS


# --- Reusable GP Model Definitions (Consolidated) ---
class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_type: str = "tanimoto"):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()

        if kernel_type == "jdft2d":
            self.covar_module = ScaleKernel(
                RFFKernel(2500) + TanimotoKernel() + RBFKernel() + MaternKernel()
            )
        else:
            self.covar_module = ScaleKernel(TanimotoKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)  # type: ignore


class GraphGP(SIGP):
    def __init__(self, train_x, train_y, likelihood, kernel, **kernel_kwargs):
        super().__init__(train_x, train_y, likelihood)
        self.mean = ConstantMean()
        self.covariance = kernel(**kernel_kwargs)

    def forward(self, x):
        mean = self.mean(torch.zeros(len(x), 1)).float()  # type: ignore
        covariance = self.covariance(x)
        jitter = max(covariance.diag().mean().detach().item() * 1e-4, 1e-4)
        covariance += torch.eye(len(x)) * jitter
        return MultivariateNormal(mean, covariance)  # type: ignore

    def transform_inputs(self, X):
        """Identity transform for non-tensorial inputs to satisfy BoTorch API."""
        return X


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
        self.covariance = kernel

    def forward(self, x):
        mean = self.mean(torch.zeros(len(x), 1)).float()  # type: ignore
        covariance = self.covariance(x)
        jitter = max(covariance.diag().mean().detach().item() * 1e-4, 1e-4)  # type: ignore
        covariance += torch.eye(len(x)) * jitter
        return MultivariateNormal(mean, covariance)  # type: ignore

    def transform_inputs(self, X):
        """Identity transform for non-tensorial inputs to satisfy BoTorch API."""
        return X


# --- Model Initialization Functions (Consolidated) ---
def initialize_model(
    train_x: torch.Tensor, train_obj: torch.Tensor, likelihood, **kwargs
):
    model = ExactGPModel(train_x, train_obj, likelihood, **kwargs).to(train_x)
    return model


def initialize_graph_gp(train_x, train_obj, likelihood, kernel, **kernel_kwargs):
    model = GraphGP(train_x, train_obj, likelihood, kernel, **kernel_kwargs)
    return model


def initialize_stacked_gp(train_x, train_obj, likelihood, kernel, **kwargs):
    """Initializes a StackedGP model.

    This function instantiates the provided kernel class with any given kwargs
    before passing the kernel instance to the StackedGP model constructor.
    """
    kernel_instance = kernel(**kwargs)
    model = StackedGP(train_x, train_obj, likelihood, kernel_instance)
    return model


# --- Unified Experiment Runner Function ---
def run_experiment(
    experiment_type: str,
    encoding: str,
    target_cols: List[str],
    n_trials: int,
    n_iters: int,
    holdout_set_size: float,
    num_workers: int = 1,
    custom_holdout_col=None,
) -> Dict[str, Any]:
    # Use Path to correctly find the project root
    base_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_path, "..", ".."))

    # Standardize data and figure paths
    data_file = f"{experiment_type.lower().replace(' ', '_')}.csv"
    data_path = os.path.join(
        project_root,
        "polyatomic_complexes",
        "dataset",
        experiment_type.lower().replace(" ", "_"),
        data_file,
    )
    results_dir = os.path.join(
        project_root, "polyatomic_complexes", "results", experiment_type
    )
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{encoding}_{time.time()}.txt")

    results = []

    def run_single_column(target):
        current_holdout_size = holdout_set_size
        if custom_holdout_col and target == custom_holdout_col:
            current_holdout_size = 0.9

        print(
            f"Starting experiment for: {experiment_type}, Encoding: {encoding}, Target: {target}"
        )

        loader = LoadDatasetForTask(
            X="", y=data_path, y_column=target, repn=encoding, project_root=project_root
        )
        X, y = loader.load_dataset(experiment_type)

        if not isinstance(X, list) and (X.shape[0] == 0 or y.shape[0] == 0):
            print(f"Skipping experiment for {target} due to empty data.")
            return [target, "Error", "Error", "Error", "Error"]

        if encoding in ["GRAPHS", "stacked_complexes"]:
            evaluate_func = evaluate_graph_model
            initialize_func = (
                initialize_stacked_gp
                if encoding == "stacked_complexes"
                else initialize_graph_gp
            )
            eval_kwargs = {"kernel": GraphletSamplingKernel}
        else:
            evaluate_func = evaluate_model
            initialize_func = initialize_model
            eval_kwargs = (
                {"kernel_type": "jdft2d"} if experiment_type == "JDFT2D" else {}
            )

        r2_list, rmse_list, mae_list, crps_list, _, _, _ = evaluate_func(  # type: ignore
            initialize_model=initialize_func,
            n_trials=n_trials,
            n_iters=n_iters,
            test_set_size=current_holdout_size,
            X=X,  # type: ignore
            y=y,
            figure_path="",
            **eval_kwargs,  # type: ignore
        )

        mean_r2 = f"mean R^2: {np.mean(r2_list):.4f} +- {np.std(r2_list) / np.sqrt(len(r2_list)):.4f}"
        mean_rmse = f"mean RMSE: {np.mean(rmse_list):.4f} +- {np.std(rmse_list) / np.sqrt(len(rmse_list)):.4f}"
        mean_mae = f"mean MAE: {np.mean(mae_list):.4f} +- {np.std(mae_list) / np.sqrt(len(mae_list)):.4f}"
        mean_crps = f"mean CRPS: {np.mean(crps_list):.4f} +- {np.std(crps_list) / np.sqrt(len(crps_list)):.4f}"

        return [target, mean_r2, mean_rmse, mean_mae, mean_crps]

    if num_workers > 1:
        with Pool(num_workers) as p:
            results = p.map(run_single_column, target_cols)
    else:
        for col in target_cols:
            results.append(run_single_column(col))

    with open(results_path, "w") as f:
        f.write(f"{experiment_type}:\n")
        f.write(f"{encoding}:\n")
        for result in results:
            col, mean_r2, mean_rmse, mean_mae, mean_crps = result
            f.write(f"column: {col}, {mean_r2}, {mean_rmse}, {mean_mae}, {mean_crps}\n")

    print(f"Results written to {results_path}")
    return {"experiment_type": experiment_type, "results": results}
