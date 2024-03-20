import torch

# GP specific
from gaussian_process import transform_batched_tensor, update_random_observations, optimize_acqf_and_get_observation
from sklearn.model_selection import train_test_split


#botorch specific
from botorch import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.models.gp_regression import SingleTaskGP

#gpytorch specific
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, LinearKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood


if torch.cuda.is_available(): 
    dev = "cuda:0"
    torch.cuda.empty_cache()
else: 
    dev = "cpu" 
device = torch.device(dev) 