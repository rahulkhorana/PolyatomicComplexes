import os
import torch
import time
import numpy as np
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
from kernels import TanimotoKernel, WalkKernel, GraphKernel
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
        assert type(ENCODING) is str and len(ENCODING) > 0

        if ENCODING == 'complexes':
            self.mean_module = ConstantMean()
            self.covar_module = ScaleKernel(base_kernel=WalkKernel())
            self.to(device)
        elif ENCODING != 'graphs':
            self.mean_module = ConstantMean()
            self.covar_module = ScaleKernel(base_kernel=TanimotoKernel())
            self.to(device)
        else:
            self.mean_module = ConstantMean()
            self.covar_module = ScaleKernel(base_kernel=GraphKernel())
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
    emll.append(mll)
    print("mll complete")
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


if __name__ == '__main__':
    EXPERIMENT_TYPE = 'Photoswitches'
    ENCODING = 'complexes'
    N_TRIALS = 5
    N_ITERS = 5
    holdout_set_size = 0.33
    # dataset processing
    X,y = [], []
    # dataset loading
    if ENCODING == "complexes":
        ds = LoadDatasetForTask(X='dataset/photoswitches/fast_complex_lookup_repn.pkl', y='dataset/photoswitches/photoswitches.csv', repn=ENCODING)
        X, y = ds.load()
    elif ENCODING == "SELFIES" or ENCODING == 'fingerprints' or ENCODING == 'GRAPHS':
        X,y = LoadDatasetForTask(Xpath='', yPath='', repn=ENCODING).load()
    else:
        raise Exception("unsupported")
    
    """    
    # training
    best_observed_all_ei, best_random_all, emll = [], [], []
    best_observed_all_ei, best_random_all = run_training_loop(initialize_model=initialize_model,n_trials=N_TRIALS, n_iters=N_ITERS, holdout_size=holdout_set_size, X=X, y=y)

    if type(EXPERIMENT_TYPE) is str:
        trial_num = len(os.listdir(f'results/{EXPERIMENT_TYPE}'))
        results_path = f"results/{EXPERIMENT_TYPE}/result_trial_{trial_num}_{time.time()}.npy"

        with open(results_path, 'wb') as f:
            np.save(f, np.asarray(best_observed_all_ei))
            np.save(f, np.asarray(best_random_all))
            np.save(f, np.asarray(emll))
        
        print("CONCLUDED")
    """