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
sys.path.append('.')
from gauche_utils.data_utils import transform_data

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

class GP(ExactGP):
    def __init__(self, train_X, train_Y, likelihood):
        assert type(ENCODING) is str and len(ENCODING) > 0
        super(ExactGP).__init__(train_X, train_Y, likelihood)
        if ENCODING == 'complexes':
            return
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


def initialize_model(train_x:torch.Tensor, train_obj:torch.Tensor, likelihood, state_dict=None):
    model = ExactGPModel(train_x, train_obj, likelihood).to(train_x)
    return model


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
    r2_list,rmse_list, mae_list, confidence_percentiles, mae_mean, mae_std = evaluate_model(initialize_model=initialize_model,n_trials=N_TRIALS, n_iters=N_ITERS, test_set_size=holdout_set_size, X=X, y=y, figure_path=f'results/{EXPERIMENT_TYPE}/{time.time()}_confidence_mae_model_{ENCODING}.png')

    if type(EXPERIMENT_TYPE) is str:
        trial_num = len(os.listdir(f'results/{EXPERIMENT_TYPE}'))
        results_path = f"results/{EXPERIMENT_TYPE}/result_trial_{trial_num}_{time.time()}.txt"
        
        mean_r2 = "\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list), np.std(r2_list)/np.sqrt(len(r2_list)))
        mean_rmse = "mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)/np.sqrt(len(rmse_list)))
        mean_mae = "mean MAE: {:.4f} +- {:.4f}\n".format(np.mean(mae_list), np.std(mae_list)/np.sqrt(len(mae_list)))

        with open(results_path, 'w') as f:
            f.write(mean_r2)
            f.write('\n')
            f.write(mean_rmse)
            f.write('\n')
            f.write(mean_mae)
            f.write('\n')
            f.write('r^2 list: \n')
            np.savetxt(f, r2_list, delimiter=',')
            f.write('\n')
            f.write('rmse list: \n')
            np.savetxt(f, rmse_list, delimiter=',')
            f.write('\n')
            f.write('mae list: ')
            np.savetxt(f, mae_list, delimiter=',')
            f.write('\n')
            f.write('confidence percentiles: \n')
            np.savetxt(f, confidence_percentiles, delimiter=',')
            f.write('\n')
            f.write('mae mean: \n')
            np.savetxt(f, mae_mean, delimiter=',')
            f.write('\n')
            f.write('mae std: \n')
            np.savetxt(f, mae_std, delimiter=',')
        f.close()
        
        print("CONCLUDED")