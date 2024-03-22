import os
import sys
import torch
import time
import numpy as np
from load_process_data import LoadDatasetForTask

#botorch specific
from botorch.models.gp_regression import ExactGP

#gpytorch specific
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

# kernels + gp
from kernels import TanimotoKernel, WalkKernel, GraphKernel
from gaussian_process import run_training_loop, evaluate_model

# gauche libraries
sys.path.insert(1, 'gauche_utils')
from gauche_utils.data_utils import transform_data

from matplotlib import pyplot as plt

if torch.cuda.is_available():
    dev = "cuda:0"
    torch.cuda.empty_cache()
else: 
    dev = "cpu" 
device = torch.device(dev)


class GP(ExactGP):
    def __init__(self, train_X, train_Y):
        super(ExactGP, self).__init__(train_X, train_Y, likelihood=GaussianLikelihood())
        assert type(ENCODING) is str and len(ENCODING) > 0
        if ENCODING == 'complexes':
            self.mean_module = ConstantMean()
            self.covar_module = ScaleKernel(TanimotoKernel())

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
    model = GP(train_x, train_obj).to(train_x)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    emll.append(mll)
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
        X,y = ds.load()
    elif ENCODING == "SELFIES" or ENCODING == 'fingerprints' or ENCODING == 'GRAPHS':
        X,y = LoadDatasetForTask(Xpath='', yPath='', repn=ENCODING).load()
    else:
        raise Exception("unsupported")
    
    print(f'tpes {type(X)}')
    print(f'tpes {type(y)}')
    # training
    best_observed_all_ei, best_random_all, emll = [], [], []
    best_observed_all_ei, best_random_all = evaluate_model(initialize_model=initialize_model,n_trials=N_TRIALS, n_iters=N_ITERS, test_set_size=holdout_set_size, X=X, y=y)

    if type(EXPERIMENT_TYPE) is str:
        trial_num = len(os.listdir(f'results/{EXPERIMENT_TYPE}'))
        results_path = f"results/{EXPERIMENT_TYPE}/result_trial_{trial_num}_{time.time()}.npy"


        def ci(y):
            return 1.96 * y.std(axis=0) / np.sqrt(N_TRIALS)

        iters = np.arange(N_ITERS + 1)
        y_ei = np.asarray(best_observed_all_ei)
        y_rnd = np.asarray(best_random_all)

        y_rnd_mean = y_rnd.mean(axis=0)
        y_ei_mean = y_ei.mean(axis=0)
        y_rnd_std = y_rnd.std(axis=0)
        y_ei_std = y_ei.std(axis=0)

        lower_rnd = y_rnd_mean - y_rnd_std
        upper_rnd = y_rnd_mean + y_rnd_std
        lower_ei = y_ei_mean - y_ei_std
        upper_ei = y_ei_mean + y_ei_std

        plt.plot(iters, y_rnd_mean, label='Random')
        plt.fill_between(iters, lower_rnd, upper_rnd, alpha=0.2)
        plt.plot(iters, y_ei_mean, label='EI')
        plt.fill_between(iters, lower_ei, upper_ei, alpha=0.2)
        plt.xlabel('Number of Iterations')
        plt.ylabel('Best Objective Value')
        plt.legend(loc="lower right")
        plt.xticks(list(np.arange(1, 21)))
        plt.show()

        with open(results_path, 'wb') as f:
            print(f'stats best obs all ei : {best_observed_all_ei}')
            print(f'stats best rand all : {best_random_all}')
            print(f'stats emll {emll[0].likelihood}')
            np.save(f, np.asarray(best_observed_all_ei))
            np.save(f, np.asarray(best_random_all))
            np.save(f, np.asarray(emll))
        
        print("CONCLUDED")