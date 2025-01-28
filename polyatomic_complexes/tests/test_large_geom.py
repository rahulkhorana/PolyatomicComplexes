import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

BASE_PATH = Path(__file__)


from polyatomic_complexes.src.complexes.polyatomic_geometry import (
    PolyatomicGeometrySMILE,
)

from polyatomic_complexes.src.complexes.force_complex import ForceComplex
from polyatomic_complexes.src.complexes.quantum_complex import QuantumComplex
from polyatomic_complexes.src.complexes.abstract_complex import AbstractComplex
from polyatomic_complexes.src.complexes.quantum_theor_complex import QuantumWavesComplex


from .test_geom import (
    check_abstract_complex_test,
    check_force_complex_test,
    check_quantum_complex_test,
    check_quantum_waves_complex_test,
)

from .test_geom_unique import (
    check_adjacency_lists_unique,
    check_betti_numbers_unique,
    check_incidence_unique,
    check_dirac_unique,
    check_laplacians_unique,
    check_persistence_unique,
    check_skeleta_unique,
    check_coadjacency_unique,
)

modes = ["abstract", "force-field"]

#### LARGE SANITY ####
parent_path = BASE_PATH.parent.parent.__str__()
datapath_esol = parent_path + "/dataset/esol/ESOL.csv"
datapath_freesolv = parent_path + "/dataset/free_solv/FreeSolv.csv"
datapath_lipo = parent_path + "/dataset/lipophilicity/Lipophilicity.csv"
datapath_photo = parent_path + "/dataset/photoswitches/photoswitches.csv"
smiles_esol = pd.read_csv(datapath_esol)["smiles"].tolist()
smiles_freesolv = pd.read_csv(datapath_freesolv)["smiles"].tolist()
smiles_lipo = pd.read_csv(datapath_lipo)["smiles"].tolist()
smiles_photo = pd.read_csv(datapath_photo)["SMILES"].tolist()
ALL_SMILES = smiles_esol + smiles_freesolv + smiles_lipo + smiles_photo
ALL_SMILES = np.random.choice(a=ALL_SMILES, size=5, replace=False).tolist()

large_cases = []
for smile in ALL_SMILES:
    for mode in modes:
        large_cases.append((smile, mode))


@pytest.mark.parametrize("smile,mode", large_cases)
def test_large_polyatomic_geometry(smile, mode):
    pgs = PolyatomicGeometrySMILE(smile=smile, target_dimension=3, mode=mode)
    pgs = pgs.smiles_to_geom_complex()
    if mode == "abstract":
        assert isinstance(pgs, AbstractComplex)
        assert check_abstract_complex_test(pgs)
    elif mode == "force-field":
        assert isinstance(pgs, ForceComplex)
        assert check_force_complex_test(pgs)
    else:
        raise Exception("INVALID + UNSUPPORTED")


modes_quantum = ["quantum", "quantum-waves"]


@pytest.mark.parametrize("smile,mode", [])
def test_quantum_polyatomic_geometry(smile, mode):
    pgs = PolyatomicGeometrySMILE(smile=smile, target_dimension=3, mode=mode)
    pgs = pgs.smiles_to_geom_complex()
    if mode == "quantum":
        assert isinstance(pgs, QuantumComplex)
        assert check_quantum_complex_test(pgs)
    elif mode == "quantum-waves":
        assert isinstance(pgs, QuantumWavesComplex)
        assert check_quantum_waves_complex_test(pgs)
    else:
        raise Exception("INVALID + UNSUPPORTED")


#### LARGE UNIQUE ####
large_cases_pair = []
for smile1 in ALL_SMILES[:5]:
    for smile2 in ALL_SMILES[:5]:
        if smile1 != smile2:
            for mode in modes:
                large_cases_pair.append((smile1, smile2, mode))


@pytest.mark.parametrize("smile1,smile2,mode", large_cases_pair)
def test_large_unique_battery(smile1, smile2, mode):
    assert smile1 != smile2 and mode in set(modes)
    pgs1 = PolyatomicGeometrySMILE(smile=smile1, target_dimension=3, mode=mode)
    complex1 = pgs1.smiles_to_geom_complex()
    pgs2 = PolyatomicGeometrySMILE(smile=smile2, target_dimension=3, mode=mode)
    complex2 = pgs2.smiles_to_geom_complex()
    assert check_adjacency_lists_unique(complex1, complex2)
    assert check_betti_numbers_unique(complex1, complex2)
    assert check_incidence_unique(complex1, complex2)
    assert check_dirac_unique(complex1, complex2)
    assert check_laplacians_unique(complex1, complex2)
    assert check_persistence_unique(complex1, complex2)
    assert check_skeleta_unique(complex1, complex2)
    assert check_coadjacency_unique(complex1, complex2)
