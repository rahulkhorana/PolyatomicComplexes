import os
import time
import json
import random
import pytest
import subprocess
import polyatomic_complexes
import polyatomic_complexes.src.complexes as complexes
import polyatomic_complexes.experiments as experiments


cwd = os.getcwd()
root_data = cwd + "/polyatomic_complexes/"


cases = []


def fuzz_test(n=20, k=15):
    with open(root_data + "dataset/construct/lookup_map.json") as data:
        lookup = json.load(data)
    assert isinstance(lookup, dict)
    for _ in range(n):
        atom_list = random.sample(list(lookup.keys()), k)
        p = complexes.polyatomic_complex.PolyAtomComplex(atom_list)
        case = (p, "general")
        cases.append(case)
        case = (p, "fast")
        cases.append(case)
        case = (p, "fast_stacked")
        cases.append(case)


fuzz_test(20, 1)
fuzz_test(10, 22)
fuzz_test(5, 27)


@pytest.mark.parametrize(
    "polyatom,build_type",
    cases,
)
def test_build(polyatom: complexes.polyatomic_complex.PolyAtomComplex, build_type: str):
    if build_type == "general":
        assert polyatom.general_build_complex()
    elif build_type == "fast_stacked":
        assert polyatom.fast_stacked_complex()
    else:
        assert polyatom.fast_build_complex()


cases2 = []


def fuzz_test(n=10):
    with open(root_data + "dataset/construct/lookup_map.json") as data:
        lookup = json.load(data)
    assert isinstance(lookup, dict)
    items = random.sample(list(lookup.items()), n)
    for k, it in items:
        p, n, e = it
        a = complexes.atomic_complex.AtomComplex(p, n, e, 5, 3, 3, 0)
        case = (a, p, n, e)
        cases2.append(case)


fuzz_test()


@pytest.mark.parametrize(
    "atom,ans_p,ans_n,ans_e",
    cases2,
)
def test_build(
    atom: complexes.atomic_complex.AtomComplex,
    ans_p: int,
    ans_n: int,
    ans_e: int,
):
    assert isinstance(atom.P, int) and atom.P >= 0
    assert isinstance(atom.N, int) and atom.N >= 0
    assert isinstance(atom.E, int) and atom.E >= 0
    assert max(atom.P, atom.N) >= 1
    assert min([atom.P, atom.N, atom.E]) >= 0
    assert isinstance(atom.d_p, list) or isinstance(atom.d_p, int)
    assert isinstance(atom.d_n, list) or isinstance(atom.d_n, int)
    assert isinstance(atom.d_e, list) or isinstance(atom.d_e, int)
    if isinstance(atom.d_p, int):
        assert atom.d_p > 0
    else:
        assert len(atom.d_p) == 2 and min(atom.d_p) >= 0
    if isinstance(atom.d_n, int):
        assert atom.d_n > 0
    else:
        assert len(atom.d_n) == 2 and min(atom.d_n) >= 0
    assert isinstance(atom.d_e, int) and atom.d_e >= 0
    assert isinstance(atom.cutoff, int)
    assert atom.P == ans_p and atom.N == ans_n and atom.E == ans_e
    assert atom.fast_build_complex()
    if (
        isinstance(atom.d_p, int)
        and isinstance(atom.d_n, int)
        and isinstance(atom.d_e, int)
    ):
        assert atom.general_build_complex()


all_paths = [
    ("data/esol", "dataset/esol/"),
    ("data/free_solv", "dataset/free_solv/"),
    ("data/lipophilicity", "dataset/lipophilicity/"),
    ("data/materials_project", "dataset/materials_project/"),
    ("data/mp_matbench_jdft2d", "dataset/mp_matbench_jdft2d/"),
    ("data/photoswitches", "dataset/photoswitches/"),
]

paths = [all_paths[0]]


def run_git_lfs_pull():
    result = subprocess.run(["git", "lfs", "pull"], capture_output=True, text=True)

    if result.returncode == 0:
        print("Git LFS pull successful.")
    else:
        print("Git LFS pull failed.")
        print(result.stderr)


run_git_lfs_pull()


@pytest.mark.parametrize(
    "build,root",
    paths,
)
def test_data_processing(build: str, root: str):
    if build not in os.listdir(cwd + "/sanity_testing_build"):
        os.makedirs(cwd + "/sanity_testing_build/" + build)
    src_path = root_data + root
    dummy_path = cwd + "/sanity_testing_build/" + build + "/"
    if "esol" in set(build.split("/")):
        e = polyatomic_complexes.complexes.process_esol.ProcessESOL(
            src_path, dummy_path
        )
        e.process()
        e.process_deep_complexes()
        e.process_stacked()
    elif "free_solv" in set(build.split("/")):
        e = polyatomic_complexes.complexes.process_freesolv.ProcessFreeSolv(
            src_path, dummy_path
        )
        e.process()
        e.process_deep_complexes()
        e.process_stacked()
    elif "lipophilicity" in set(build.split("/")):
        e = polyatomic_complexes.complexes.process_lipophilicity.ProcessLipophilicity(
            src_path, dummy_path
        )
        e.process()
        e.process_deep_complexes()
        e.process_stacked()
    elif "materials_project" in set(build.split("/")):
        e = polyatomic_complexes.complexes.process_materials_project.ProcessMP(
            src_path, dummy_path
        )
        e.process()
        e.process_deep_complexes()
        e.process_stacked()
    elif "mp_matbench_jdft2d" in set(build.split("/")):
        e = polyatomic_complexes.complexes.process_mp_jdft2d.ProcessJDFT(
            src_path, dummy_path
        )
        e.process()
        e.process_deep_complexes()
        e.process_stacked()
    elif "photoswitches" in set(build.split("/")):
        e = polyatomic_complexes.complexes.process_photoswitches.ProcessPhotoswitches(
            src_path, dummy_path
        )
        e.process()
        e.process_deep_complexes()
        e.process_stacked()
    assert len(os.listdir(dummy_path)) == 3


def run_experiment(
    possible_target_cols,
    one_experiment_fn,
    ENCODING,
    N_TRIALS,
    N_ITERS,
    EXPERIMENT_TYPE,
    destination_path,
    x_path,
    y_path,
    fig_path,
):
    results = []
    try:
        for col in possible_target_cols:
            print(f"column: {col}")
            mean_r2, mean_rmse, mean_mae, mean_crps = one_experiment_fn(
                target=col,
                encoding=ENCODING,
                n_trials=N_TRIALS,
                n_iters=N_ITERS,
                encoding_path=x_path,
                data_path=y_path,
                fig_path=fig_path,
            )
            print("finished")
            results.append([col, mean_r2, mean_rmse, mean_mae, mean_crps])

        if type(EXPERIMENT_TYPE) is str:
            results_path = destination_path

            with open(results_path, "w") as f:
                f.write(EXPERIMENT_TYPE + ":")
                f.write("\n")
                f.write(ENCODING + ":")
                for result in results:
                    col, mean_r2, mean_rmse, mean_mae, mean_crps = result
                    f.write(
                        f"column: {col}, {mean_r2}, {mean_rmse}, {mean_mae}, {mean_crps}"
                    )
                    f.write("\n")
            f.close()
            return "SUCCESS"
    except Exception as e:
        print(e)
        return "FAILED"


experimental = [
    (
        "esol",
        {
            "target_columns": [
                "ESOL predicted log solubility in mols per litre",
                "Minimum Degree",
                "Molecular Weight",
            ],
            "root": "dataset/esol/",
        },
    ),
    ("free_solv", {"target_columns": ["expt", "calc"], "root": "dataset/free_solv/"}),
    (
        "lipophilicity",
        {
            "target_columns": ["exp"],
            "root": "dataset/lipophilicity/",
        },
    ),
    (
        "photoswitches",
        {
            "target_columns": [
                "Z PhotoStationaryState",
                "PBE0 DFT E isomer pi-pi* wavelength in nm",
            ],
            "root": "dataset/photoswitches/",
        },
    ),
    (
        "mp_matbench_jdft2d",
        {
            "target_columns": ["exfoliation_en"],
            "root": "dataset/mp_matbench_jdft2d/",
        },
    ),
    (
        "materials_project",
        {
            "target_columns": [
                "formation_energy_per_atom",
            ],
            "root": "dataset/materials_project/",
        },
    ),
]

path_mappings = {
    "deep_complexes": "deep_complex_lookup_repn.pkl",
    "fast_complexes": "fast_complex_lookup_repn.pkl",
    "stacked_complexes": "stacked_complex_lookup_repn.pkl",
}


def run_exp_over(sample_encs, tgt_cols, prefix, params, f, name):
    statuses = True
    for t in tgt_cols:
        for e in sample_encs:
            _dest = (
                cwd + "/sanity_testing_build/" + prefix + "/" + f"{e}_{time.time()}.txt"
            )
            src_path = root_data + params["root"]
            y_path = src_path + f"{name}.csv"
            fig_path = (
                cwd
                + "/sanity_testing_build/"
                + prefix
                + f"/confidence_mae_model_{e}_{t}.png"
            )
            if e in path_mappings.keys():
                x_path = src_path + path_mappings[e]
            else:
                x_path = None
            status = run_experiment(
                tgt_cols,
                f,
                e,
                2,
                2,
                name,
                _dest,
                x_path,
                y_path,
                fig_path,
            )
            statuses = statuses and (status == "SUCCESS")
    return statuses


@pytest.mark.parametrize(
    "name,params",
    experimental,
)
def test_run_experiments(name: str, params: dict):
    if name == "esol":
        sample_encs = ["deep_complexes", "fingerprints", "GRAPHS", "SMILES"]
        one_experiment_fn = experiments.esol_experiment.one_experiment
        tgt_cols = params["target_columns"]
        prefix = "results/esol"
        if prefix not in os.listdir(cwd + "/sanity_testing_build") and not (
            os.path.exists(cwd + "/sanity_testing_build/" + prefix)
        ):
            os.makedirs(cwd + "/sanity_testing_build/" + prefix)
        status = run_exp_over(
            sample_encs, tgt_cols, prefix, params, one_experiment_fn, "ESOL"
        )
        assert status
    elif name == "free_solv":
        sample_encs = ["deep_complexes", "fingerprints", "GRAPHS", "SMILES"]
        one_experiment_fn = experiments.freesolv_experiment.one_experiment
        tgt_cols = params["target_columns"]
        prefix = "results/free_solv"
        if prefix not in os.listdir(cwd + "/sanity_testing_build") and not (
            os.path.exists(cwd + "/sanity_testing_build/" + prefix)
        ):
            os.makedirs(cwd + "/sanity_testing_build/" + prefix)
        status = run_exp_over(
            sample_encs, tgt_cols, prefix, params, one_experiment_fn, "FreeSolv"
        )
        assert status
    elif name == "materials_project":
        sample_encs = ["complexes"]
        one_experiment_fn = experiments.materials_project_experiment.one_experiment
        tgt_cols = params["target_columns"]
        prefix = "results/materials_project"
        if prefix not in os.listdir(cwd + "/sanity_testing_build") and not (
            os.path.exists(cwd + "/sanity_testing_build/" + prefix)
        ):
            os.makedirs(cwd + "/sanity_testing_build/" + prefix)
        status = run_exp_over(
            sample_encs, tgt_cols, prefix, params, one_experiment_fn, "materials_data"
        )
        assert status
    elif name == "lipophilicity":
        sample_encs = ["complexes", "fingerprints", "GRAPHS", "SMILES"]
        one_experiment_fn = experiments.lipophilicity_experiment.one_experiment
        tgt_cols = params["target_columns"]
        prefix = "results/lipophilicity"
        if prefix not in os.listdir(cwd + "/sanity_testing_build") and not (
            os.path.exists(cwd + "/sanity_testing_build/" + prefix)
        ):
            os.makedirs(cwd + "/sanity_testing_build/" + prefix)
        status = run_exp_over(
            sample_encs, tgt_cols, prefix, params, one_experiment_fn, "Lipophilicity"
        )
        assert status
    elif name == "photoswitches":
        sample_encs = ["complexes", "fingerprints", "GRAPHS", "SMILES"]
        one_experiment_fn = experiments.photoswitches_experiment.one_experiment
        tgt_cols = params["target_columns"]
        prefix = "results/photoswitches"
        if prefix not in os.listdir(cwd + "/sanity_testing_build") and not (
            os.path.exists(cwd + "/sanity_testing_build/" + prefix)
        ):
            os.makedirs(cwd + "/sanity_testing_build/" + prefix)
        status = run_exp_over(
            sample_encs, tgt_cols, prefix, params, one_experiment_fn, "photoswitches"
        )
        assert status
    elif name == "mp_matbench_jdft2d":
        sample_encs = ["complexes"]
        one_experiment_fn = experiments.jdft2d_matbench_experiment.one_experiment
        tgt_cols = params["target_columns"]
        prefix = "results/mp_matbench_jdft2d"
        if prefix not in os.listdir(cwd + "/sanity_testing_build") and not (
            os.path.exists(cwd + "/sanity_testing_build/" + prefix)
        ):
            os.makedirs(cwd + "/sanity_testing_build/" + prefix)
        status = run_exp_over(
            sample_encs, tgt_cols, prefix, params, one_experiment_fn, "jdft2d"
        )
        assert status
    else:
        print(f"invalid experiment name {name}")
        assert True
