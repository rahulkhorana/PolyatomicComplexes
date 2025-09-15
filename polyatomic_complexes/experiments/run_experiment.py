import os
import argparse
import time
from typing import Dict, List, Any
from multiprocessing import Pool
from .experiment_runner import run_experiment

EXPERIMENT_CONFIGS = {
    "esol": {
        "experiment_type": "ESOL",
        "encoding": "stacked_complexes",
        "n_trials": 20,
        "n_iters": 10,
        "holdout_set_size": 0.33,
        "target_cols": [
            "ESOL predicted log solubility in mols per litre",
            "Minimum Degree",
            "Molecular Weight",
            "Number of H-Bond Donors",
            "Number of Rings",
            "Number of Rotatable Bonds",
            "Polar Surface Area",
            "measured log solubility in mols per litre",
        ],
        "num_workers": 1,
    },
    "freesolv": {
        "experiment_type": "FreeSolv",
        "encoding": "stacked_complexes",
        "n_trials": 20,
        "n_iters": 5,
        "holdout_set_size": 0.33,
        "target_cols": ["expt", "calc"],
        "num_workers": 1,
    },
    "lipophilicity": {
        "experiment_type": "Lipophilicity",
        "encoding": "stacked_complexes",
        "n_trials": 20,
        "n_iters": 5,
        "holdout_set_size": 0.33,
        "target_cols": ["exp"],
        "num_workers": 1,
    },
    "jdft2d": {
        "experiment_type": "JDFT2D",
        "encoding": "stacked_complexes",
        "n_trials": 20,
        "n_iters": 10,
        "holdout_set_size": 0.2,
        "target_cols": ["exfoliation_en"],
        "num_workers": 2,
    },
    "materials_project": {
        "experiment_type": "Materials Project",
        "encoding": "complexes",
        "n_trials": 7,
        "n_iters": 5,
        "holdout_set_size": 0.9,
        "target_cols": [
            "uncorrected_energy_per_atom",
            "energy_per_atom",
            "formation_energy_per_atom",
            "equilibrium_reaction_energy_per_atom",
            "total_magnetization",
            "total_magnetization_normalized_vol",
        ],
        "num_workers": 1,
    },
    "mp_matbench": {
        "experiment_type": "MP_MatBench",
        "encoding": "stacked_complexes",
        "n_trials": 20,
        "n_iters": 5,
        "holdout_set_size": 0.33,
        "target_cols": ["g_vrh", "k_voigt", "k_reuss", "k_vrh", "g_voigt", "efermi"],
        "num_workers": 1,
        "custom_holdout_col": "efermi",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Run a specific experiment.")
    parser.add_argument(
        "experiment",
        type=str,
        choices=EXPERIMENT_CONFIGS.keys(),
        help="The name of the experiment to run.",
    )
    args = parser.parse_args()
    config = EXPERIMENT_CONFIGS[args.experiment]
    print(f"Running experiment: {config['experiment_type']}")
    run_experiment(**config)


if __name__ == "__main__":
    main()
