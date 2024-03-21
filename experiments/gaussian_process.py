import os
import time
import torch
from typing import Tuple

# data specific
import numpy as np
import jax.numpy as jnp

#botorch specific
from botorch import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement

#sklearn specific
from sklearn.model_selection import train_test_split

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
    heldout_inputs = torch.cat((heldout_inputs[:best_idx], heldout_inputs[best_idx+1:]), axis=0)
    heldout_outputs = torch.cat((heldout_outputs[:best_idx], heldout_outputs[best_idx+1:]), axis=0)

    return new_x, new_obj, heldout_inputs, heldout_outputs


def update_random_observations(best_random, heldout_inputs, heldout_outputs):
    # Take a random sample by permuting the indices and selecting the first element.
    index = torch.randperm(len(heldout_outputs))[0]
    next_random_best = heldout_outputs[index]
    best_random.append(max(best_random[-1], next_random_best))

    # Delete the selected input and value from the heldout set.
    heldout_inputs = torch.cat((heldout_inputs[:index], heldout_inputs[index+1:]), axis=0)
    heldout_outputs = torch.cat((heldout_outputs[:index], heldout_outputs[index+1:]), axis=0)

    return best_random, heldout_inputs, heldout_outputs

def transform_batched_tensor(tensors:list, max_len=None) -> Tuple[torch.Tensor, int]:
    arr = []
    for t in tensors:
        i, j, k, l = t
        assert type(i) is torch.Tensor and type(j) is torch.Tensor and type(k) is torch.Tensor and type(l) is torch.Tensor
        t = torch.concat([i.flatten(), j.flatten(), k.flatten(), l.flatten()])
        arr.append(t)
    if max_len is None:
        max_len = max([x.squeeze().numel() for x in arr])
    data = [torch.nn.functional.pad(x, pad=(0, max_len - x.numel()), mode='constant', value=0) for x in arr]
    data = torch.stack(data)
    return data, max_len

    
def run_training_loop(initialize_model, n_trials, n_iters, holdout_size, X, y, verbose=False):
    best_observed_all_ei, best_random_all = [], []
    for trial in range(1, n_trials + 1):
        print(f"\nTrial {trial:>2} of {n_trials} ", end="")
        best_observed_ei, best_random = [], []

        # Generate initial training data and initialize model
        train_x_ei, heldout_x_ei, train_y_ei, heldout_y_ei = train_test_split(X, y, test_size=holdout_size, random_state=trial)
        best_observed_value_ei = torch.tensor(np.max(train_y_ei))

        # Convert numpy arrays to PyTorch tensors and flatten the label vectors
        train_x_ei = transform_batched_tensor(train_x_ei).type(torch.float64).to(device)
        heldout_x_ei = transform_batched_tensor(heldout_x_ei).type(torch.float64).to(device)
        train_y_ei = torch.tensor(train_y_ei).type(torch.float64).to(device)
        heldout_y_ei = torch.tensor(heldout_y_ei).type(torch.float64).to(device)

        print("completed tensors")

        # The initial heldout set is the same for random search
        heldout_x_random = heldout_x_ei
        heldout_y_random = heldout_y_ei

        mll_ei, model_ei = initialize_model(train_x_ei, train_y_ei)

        print("completed init")

        best_observed_ei.append(best_observed_value_ei)
        best_random.append(best_observed_value_ei)

        print("completed append")

        # run N_ITERS rounds of BayesOpt after the initial random batch
        for iteration in range(1, n_iters + 1):
            t0 = time.time()

            # fit the model
            fit_gpytorch_model(mll_ei)
            # Use analytic acquisition function for batch size of 1.
            EI = ExpectedImprovement(model=model_ei, best_f=(train_y_ei.to(train_y_ei)).max())

            new_x_ei, new_obj_ei, heldout_x_ei, heldout_y_ei = optimize_acqf_and_get_observation(EI,heldout_x_ei,heldout_y_ei)
            # update training points
            train_x_ei = torch.cat([train_x_ei, new_x_ei])
            train_y_ei = torch.cat([train_y_ei, new_obj_ei])

            # update random search progress
            best_random, heldout_x_random, heldout_y_random = update_random_observations(best_random,heldout_inputs=heldout_x_random,heldout_outputs=heldout_y_random)
            best_value_ei = torch.max(new_obj_ei, best_observed_ei[-1])
            best_observed_ei.append(best_value_ei)

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
                    f"time = {t1 - t0:>4.2f}.", end=""
                )
            else:
                print(".", end="")

    best_observed_all_ei.append(best_observed_ei)
    best_random_all.append(best_random)
    return best_observed_all_ei, best_random_all




