import os
import torch
import time
import numpy as np
import jax.numpy as jnp
from load_process_data import LoadDatasetForTask

#botorch specific
from botorch.models.gp_regression import SingleTaskGP

#gpytorch specific
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

# kernels + gp
from kernels import TanimotoKernel, MolecularKernel
from gaussian_process import run_training_loop

if torch.cuda.is_available():
    dev = "cuda:0"
    torch.cuda.empty_cache()
else: 
    dev = "cpu" 
device = torch.device(dev)


class GP(SingleTaskGP):
    def __init__(self, train_X, train_Y):
        super().__init__(train_X, train_Y, None)
        assert type(MODE) is str and len(MODE) > 0

        if MODE == 'mcw':
            norm = jnp.linalg.norm(train_X[0])
            self.mean_module = ConstantMean(constant_prior=norm)
            self.covar_module = ScaleKernel(base_kernel=MolecularKernel())
            self.to(device)
        else:
            self.mean_module = ConstantMean()
            self.covar_module = ScaleKernel(base_kernel=TanimotoKernel())
            self.to(device)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def initialize_model(train_x, train_obj, state_dict=None):
    print("start init")
    model = GP(train_x, train_obj).to(device)
    print("gp complete")
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    print("mll complete")
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


if __name__ == '__main__':
    MODE = "mcw"
    N_TRIALS = 5
    N_ITERS = 5
    holdout_set_size = 0.33
    # dataset processing
    X,y = [], []
    # dataset loading
    if MODE == "mcw":
        X,y = LoadDatasetForTask(Xpath='dataset/small/X', ypath='dataset/small/Y/dataset_molecular_y_small.pt', representation=MODE).load()
        #X, Y = ProcessDataForGP()
    elif MODE == "smiles":
        X,y = LoadDatasetForTask(Xpath='dataset/small/dataset_smiles.csv', ypath='', representation='smiles').load()
    elif MODE == "selfies":
        X,y = LoadDatasetForTask(Xpath='dataset/small/dataset_smiles.csv', ypath='', representation='selfies').load()
    else:
        raise Exception("unsupported")

    # training
    best_observed_all_ei, best_random_all = [], []
    best_observed_all_ei, best_random_all = run_training_loop(initialize_model=initialize_model,n_trials=N_TRIALS, n_iters=N_ITERS, holdout_size=holdout_set_size, X=X, y=y)

    if type(EXPERIMENT_SIZE) is str and EXPERIMENT_SIZE in set(["small_experiment", "large_experiment"]):
        trial_num = len(os.listdir(f'results/{EXPERIMENT_SIZE}'))
        results_path = f"results/{EXPERIMENT_SIZE}/result_trial_{trial_num}_{time.time()}.npy"

        with open(results_path, 'wb') as f:
            np.save(f, np.asarray(best_observed_all_ei))
            np.save(f, np.asarray(best_random_all))
        
        print("CONCLUDED")