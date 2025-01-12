"""
This file contains code that was adapted and modified from the Gauche Project.
We use their code per the MIT license in their repo.
All credits go to the original authors. We cite them in our manuscript.
# Credit: https://github.com/leojklarner/gauche
# Original Paper: https://arxiv.org/abs/2212.04450
"""

import time
import torch
from typing import Tuple

# data specific
import numpy as np
import matplotlib.pyplot as plt

# botorch specific
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
import warnings

# sklearn specific
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from .metrics import CRPS

# gp specific

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from gauche.dataloader.data_utils import transform_data
from gauche import NonTensorialInputs
from gauche.kernels.graph_kernels import WeisfeilerLehmanKernel

plt.switch_backend("Agg")

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


def optimize_acqf_and_get_observation(acq_func, heldout_inputs, heldout_outputs):
    # Loop over the discrete set of points to evaluate the acquisition function at.
    acq_vals = []
    for i in range(len(heldout_outputs)):
        obj = heldout_inputs[i].unsqueeze(-2)
        acq_vals.append(acq_func(obj))  # use unsqueeze to append batch dimension

    # observe new values
    acq_vals = torch.tensor(acq_vals)
    best_idx = torch.argmax(acq_vals)
    new_x = heldout_inputs[best_idx].unsqueeze(-2)  # add batch dimension
    new_obj = heldout_outputs[best_idx].unsqueeze(-1)  # add output dimension

    # Delete the selected input and value from the heldout set.
    heldout_inputs = torch.cat(
        (heldout_inputs[:best_idx], heldout_inputs[best_idx + 1 :]), axis=0
    )
    heldout_outputs = torch.cat(
        (heldout_outputs[:best_idx], heldout_outputs[best_idx + 1 :]), axis=0
    )

    return new_x, new_obj, heldout_inputs, heldout_outputs


def update_random_observations(best_random, heldout_inputs, heldout_outputs):
    # Take a random sample by permuting the indices and selecting the first element.
    index = torch.randperm(len(heldout_outputs))[0]
    next_random_best = heldout_outputs[index]
    best_random.append(max(best_random[-1], next_random_best))

    # Delete the selected input and value from the heldout set.
    heldout_inputs = torch.cat(
        (heldout_inputs[:index], heldout_inputs[index + 1 :]), axis=0
    )
    heldout_outputs = torch.cat(
        (heldout_outputs[:index], heldout_outputs[index + 1 :]), axis=0
    )

    return best_random, heldout_inputs, heldout_outputs


def transform_batched_tensor(tensors: list, max_len=None) -> Tuple[torch.Tensor, int]:
    arr = []
    for t in tensors:
        i, j, k, l = t
        assert (
            type(i) is torch.Tensor
            and type(j) is torch.Tensor
            and type(k) is torch.Tensor
            and type(l) is torch.Tensor
        )
        t = torch.concat([i.flatten(), j.flatten(), k.flatten(), l.flatten()])
        arr.append(t)
    if max_len is None:
        max_len = max([x.squeeze().numel() for x in arr])
    data = [
        torch.nn.functional.pad(
            x, pad=(0, max_len - x.numel()), mode="constant", value=0
        )
        for x in arr
    ]
    data = torch.stack(data)
    return data, max_len


def run_training_loop(
    initialize_model, n_trials, n_iters, holdout_size, X, y, verbose=False
):
    best_observed_all_ei, best_random_all = [], []

    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    for trial in range(1, n_trials + 1):
        print(f"\nTrial {trial:>2} of {n_trials} ", end="")
        best_observed_ei, best_random = [], []

        # Generate initial training data and initialize model
        train_x_ei, heldout_x_ei, train_y_ei, heldout_y_ei = train_test_split(
            X, y, test_size=holdout_size, random_state=trial
        )
        best_observed_value_ei = torch.max(train_y_ei)

        # Convert numpy arrays to PyTorch tensors and flatten the label vectors
        train_x_ei = train_x_ei.type(torch.float64)
        heldout_x_ei = heldout_x_ei.type(torch.float64)
        train_y_ei = train_y_ei.type(torch.float64)
        heldout_y_ei = heldout_y_ei.type(torch.float64)

        # The initial heldout set is the same for random search
        heldout_x_random = heldout_x_ei
        heldout_y_random = heldout_y_ei

        mll_ei, model_ei = initialize_model(train_x_ei, train_y_ei)

        best_observed_ei.append(best_observed_value_ei)
        best_random.append(best_observed_value_ei)

        # run N_ITERS rounds of BayesOpt after the initial random batch
        for iteration in range(1, n_iters + 1):

            t0 = time.time()

            # fit the model
            fit_gpytorch_mll(mll_ei)

            # Use analytic acquisition function for batch size of 1.
            EI = ExpectedImprovement(
                model=model_ei, best_f=(train_y_ei.to(train_y_ei)).max()
            )

            new_x_ei, new_obj_ei, heldout_x_ei, heldout_y_ei = (
                optimize_acqf_and_get_observation(EI, heldout_x_ei, heldout_y_ei)
            )

            # update training points
            train_x_ei = torch.cat([train_x_ei, new_x_ei])
            train_y_ei = torch.cat([train_y_ei, new_obj_ei])

            # update random search progress
            best_random, heldout_x_random, heldout_y_random = (
                update_random_observations(
                    best_random,
                    heldout_inputs=heldout_x_random,
                    heldout_outputs=heldout_y_random,
                )
            )
            best_value_ei = torch.max(new_obj_ei, best_observed_ei[-1])
            best_observed_ei.append(best_value_ei.squeeze())

            # reinitialise the model so it is ready for fitting on the next iteration
            # use the current state dict to speed up fitting
            mll_ei, model_ei = initialize_model(
                train_x_ei,
                train_y_ei,
                model_ei.state_dict(),
            )

            t1 = time.time()

            if verbose:
                print(
                    f"\nBatch {iteration:>2}: best_value (random, qEI) = "
                    f"({max(best_random):>4.2f}, {best_value_ei:>4.2f}), "
                    f"time = {t1 - t0:>4.2f}.",
                    end="",
                )
            else:
                print(".", end="")

    best_observed_all_ei.append(torch.hstack(best_observed_ei))
    best_random_all.append(torch.hstack(best_random))

    best_observed_all_ei.append(best_observed_ei)
    best_random_all.append(best_random)
    return best_observed_all_ei, best_random_all


def evaluate_model(
    initialize_model, n_trials, n_iters, test_set_size, X, y, figure_path
):
    # initialise performance metric lists
    r2_list = []
    rmse_list = []
    mae_list = []
    crps_list = []

    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # We pre-allocate array for plotting confidence-error curves

    _, _, _, y_test = train_test_split(
        X, y, test_size=test_set_size
    )  # To get test set size
    n_test = len(y_test)

    mae_confidence_list = np.zeros((n_trials, n_test))

    print("\nBeginning training loop...")

    for i in range(0, n_trials):

        print(f"Starting trial {i}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_set_size, random_state=i
        )
        #  We standardise the outputs
        _, y_train, _, y_test, y_scaler = transform_data(
            X_train, y_train, X_test, y_test
        )
        assert isinstance(X_train, torch.Tensor) and isinstance(X_test, torch.Tensor)
        X_train, X_test = X_train.float(), X_test.float()
        # Convert numpy arrays to PyTorch tensors and flatten the label vectors
        # print(f'types {type(X_train)}')
        y_train = torch.tensor(y_train.astype(np.float64)).flatten().float()
        y_test = torch.tensor(y_test.astype(np.float64)).flatten().float()
        assert (
            isinstance(X_train, torch.Tensor)
            and isinstance(y_train, torch.Tensor)
            and isinstance(X_test, torch.Tensor)
            and isinstance(y_test, torch.Tensor)
        )

        likelihood = GaussianLikelihood()
        # initialise GP likelihood and model
        model = initialize_model(X_train, y_train, likelihood)

        # Find optimal model hyperparameters
        # "Loss" for GPs - the marginal log likelihood
        likelihood = GaussianLikelihood()
        mll = ExactMarginalLogLikelihood(likelihood, model)

        print("init done")
        # Use the BoTorch utility for fitting GPs in order to use the LBFGS-B optimiser (recommended)
        fit_gpytorch_mll(mll)

        print("fitting done")

        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()

        print("eval:")
        # mean and variance GP prediction
        f_pred = model(X_test)

        y_pred = f_pred.mean
        y_var = f_pred.variance

        # Transform back to real data space to compute metrics and detach gradients. Must unsqueeze dimension
        # to make compatible with inverse_transform in scikit-learn version > 1
        y_pred = y_scaler.inverse_transform(y_pred.detach().unsqueeze(dim=1))
        y_test = y_scaler.inverse_transform(y_test.detach().unsqueeze(dim=1))

        # Compute scores for confidence curve plotting.

        ranked_confidence_list = np.argsort(y_var.detach(), axis=0).flatten()

        for k in range(len(y_test)):

            # Construct the MAE error for each level of confidence

            conf = ranked_confidence_list[0 : k + 1]
            mae = mean_absolute_error(y_test[conf], y_pred[conf])
            mae_confidence_list[i, k] = mae

        # Output Standardised RMSE and RMSE on Train Set
        y_train = y_train.detach()
        y_pred_train = model(X_train).mean.detach()
        train_rmse_stan = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_rmse = np.sqrt(
            mean_squared_error(
                y_scaler.inverse_transform(y_train.unsqueeze(dim=1)),
                y_scaler.inverse_transform(y_pred_train.unsqueeze(dim=1)),
            )
        )

        # Compute R^2, RMSE and MAE on Test set
        score = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        crps = CRPS(y_pred, y_test).crps().tolist()

        r2_list.append(score)
        rmse_list.append(rmse)
        mae_list.append(mae)
        crps_list.append(crps)

    r2_list = np.array(r2_list)
    rmse_list = np.array(rmse_list)
    mae_list = np.array(mae_list)
    crps_list = np.array(crps_list)
    # Plot confidence-error curves

    # 1e-14 instead of 0 to for numerical reasons!
    confidence_percentiles = np.arange(1e-14, 100, 100 / len(y_test))

    # We plot the Mean-absolute error confidence-error curves

    mae_mean = np.mean(mae_confidence_list, axis=0)
    mae_std = np.std(mae_confidence_list, axis=0)

    mae_mean = np.flip(mae_mean)
    mae_std = np.flip(mae_std)

    # 1 sigma errorbars

    lower = mae_mean - mae_std
    upper = mae_mean + mae_std

    plt.plot(confidence_percentiles, mae_mean, label="mean")
    plt.fill_between(confidence_percentiles, lower, upper, alpha=0.2)
    plt.xlabel("Confidence Percentile")
    plt.ylabel("MAE (nm)")
    plt.ylim([0, np.max(upper) + 1])
    plt.xlim([0, 100 * ((len(y_test) - 1) / len(y_test))])
    plt.yticks(np.arange(0, np.max(upper) + 1, 5.0))
    plt.savefig(figure_path)

    return (
        r2_list,
        rmse_list,
        mae_list,
        crps_list,
        confidence_percentiles,
        mae_mean,
        mae_std,
    )


def evaluate_graph_model(
    initialize_model,
    X,
    y,
    test_set_size,
    n_trials,
    n_iters,
    figure_path="",
    kernel=WeisfeilerLehmanKernel,
    **kernel_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    # Initialise performance metric lists
    r2_list = []
    rmse_list = []
    mae_list = []
    crps_list = []

    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # We pre-allocate array for plotting confidence-error curves
    n_test = int(len(y) * test_set_size) + 1
    mae_confidence_list = np.zeros((n_trials, n_test))

    for i in range(n_trials):
        print(f"Starting trial {i}")
        # Carry out the random split with the current random seed
        # and standardise the outputs
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_set_size, random_state=i
        )
        _, y_train, _, y_test, y_scaler = transform_data(
            np.zeros_like(y_train), y_train, np.zeros_like(y_test), y_test
        )

        # Convert graph-structured inputs to custom data class for
        # non-tensorial inputs and convert labels to PyTorch tensors
        X_train = NonTensorialInputs(X_train)
        X_test = NonTensorialInputs(X_test)
        y_train = torch.tensor(y_train).flatten().float()
        y_test = torch.tensor(y_test).flatten().float()

        # Initialise GP likelihood and model
        likelihood = GaussianLikelihood()
        model = initialize_model(
            X_train, y_train, likelihood, kernel, node_label="element"
        )
        print("model initialization done")
        # Define the marginal log likelihood used to optimise the model hyperparameters
        mll = ExactMarginalLogLikelihood(likelihood, model)

        # Use the BoTorch utility for fitting GPs in order
        # to use the LBFGS-B optimiser (recommended)
        fit_gpytorch_mll(mll)
        print("fitting done")
        # Get into evaluation (predictive posterior) mode and compute predictions
        model.eval()
        likelihood.eval()
        f_pred = model(X_test)
        y_pred = f_pred.mean
        y_var = f_pred.variance
        print("eval done")
        # Transform the predictions back to the original scale and calucalte eval metrics
        y_pred = y_scaler.inverse_transform(y_pred.detach().unsqueeze(dim=1))
        y_test = y_scaler.inverse_transform(y_test.detach().unsqueeze(dim=1))

        # Construct the MAE error for each level of confidence
        ranked_confidence_list = np.argsort(y_var.detach(), axis=0).flatten()
        for k in range(len(y_test)):
            conf = ranked_confidence_list[0 : k + 1]
            mae = mean_absolute_error(y_test[conf], y_pred[conf])
            mae_confidence_list[i, k] = mae

        # Compute R^2, RMSE and MAE on Test set
        score = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        crps = CRPS(y_pred, y_test).crps().tolist()

        r2_list.append(score)
        rmse_list.append(rmse)
        mae_list.append(mae)
        crps_list.append(crps)

    r2_list = np.array(r2_list)
    rmse_list = np.array(rmse_list)
    mae_list = np.array(mae_list)
    crps_list = np.array(crps_list)

    # Print mean and standard error of the mean for each metric

    print(
        "\nmean R^2: {:.4f} +- {:.4f}".format(
            np.mean(r2_list), np.std(r2_list) / np.sqrt(len(r2_list))
        )
    )
    print(
        "mean RMSE: {:.4f} +- {:.4f}".format(
            np.mean(rmse_list), np.std(rmse_list) / np.sqrt(len(rmse_list))
        )
    )
    print(
        "mean MAE: {:.4f} +- {:.4f}\n".format(
            np.mean(mae_list), np.std(mae_list) / np.sqrt(len(mae_list))
        )
    )
    print(
        "mean CRPS: {:.4f} +- {:.4f}\n".format(
            np.mean(crps_list), np.std(crps_list) / np.sqrt(len(crps_list))
        )
    )

    # Plot the mean-absolute error/confidence-error curves
    # with 1 sigma errorbars

    confidence_percentiles = np.arange(1e-14, 100, 100 / len(y_test))

    mae_mean = np.mean(mae_confidence_list, axis=0)
    mae_mean = np.flip(mae_mean)
    mae_std = np.std(mae_confidence_list, axis=0)
    mae_std = np.flip(mae_std)
    lower = mae_mean - mae_std
    upper = mae_mean + mae_std

    plt.plot(confidence_percentiles, mae_mean, label="mean")
    plt.fill_between(confidence_percentiles, lower, upper, alpha=0.2)
    plt.xlabel("Confidence Percentile")
    plt.ylabel("MAE (nm)")
    plt.ylim([0, np.max(upper) + 1])
    plt.xlim([0, 100 * ((len(y_test) - 1) / len(y_test))])
    plt.yticks(np.arange(0, np.max(upper) + 1, 5.0))
    plt.show()

    return (
        r2_list,
        rmse_list,
        mae_list,
        crps_list,
        confidence_percentiles,
        mae_mean,
        mae_std,
    )
