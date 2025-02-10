import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from pathlib import Path

BASE_PATH = Path(__file__)

from polyatomic_complexes.src.complexes.polyatomic_geometry import (
    PolyatomicGeometrySMILE,
)
from polyatomic_complexes.src.complexes.abstract_complex import AbstractComplex

import pytest
import pandas as pd
import numpy as np


smiles = [
    "OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O",
    "Cc1occc1C(=O)Nc2ccccc2",
    "c1ccc2c(c1)ccc3c2ccc4c5ccccc5ccc43",
    "CCc1ccccn1",
    "COc1ccc(cc1)N2CCN(CC2)C(=O)[C@@H]3CCCC[C@H]3C(=O)NCC#N",
    "Cn1c(CN2CCN(CC2)c3ccc(Cl)cc3)nc4ccccc14",
]
modes = ["abstract"]

cases_small = []
for s in smiles:
    for m in modes:
        cases_small.append((s, m))


@pytest.mark.parametrize("smile,mode", cases_small)
def test_single_kchain(smile, mode):
    pgs = PolyatomicGeometrySMILE(smile=smile, target_dimension=3, mode=mode)
    pgs = pgs.smiles_to_geom_complex()
    if mode == "abstract":
        assert isinstance(pgs, AbstractComplex)
        assert check_kchains(pgs)
    else:
        raise Exception("INVALID + UNSUPPORTED")


cases_pair = []
for s1 in smiles:
    for s2 in smiles:
        for m in modes:
            if s1 != s2:
                cases_pair.append((s1, s2, m))


@pytest.mark.parametrize("smile1,smile2,mode", cases_pair)
def test_small_kchain_pair(smile1, smile2, mode):
    pgs1 = PolyatomicGeometrySMILE(smile=smile1, target_dimension=3, mode=mode)
    pgs2 = PolyatomicGeometrySMILE(smile=smile2, target_dimension=3, mode=mode)
    pgs1 = pgs1.smiles_to_geom_complex()
    pgs2 = pgs2.smiles_to_geom_complex()
    if mode == "abstract":
        assert isinstance(pgs1, AbstractComplex)
        assert isinstance(pgs2, AbstractComplex)
        assert check_pair_kchains(pgs1, pgs2)
    else:
        raise Exception("INVALID + UNSUPPORTED")


def check_kchains(complex):
    assert isinstance(complex, AbstractComplex)
    k_chains = complex.get_spectral_k_chains()
    for i, ch1 in enumerate(k_chains.values()):
        for j, ch2 in enumerate(k_chains.values()):
            if i != j:
                assert isinstance(ch1, np.ndarray)
                assert isinstance(ch2, np.ndarray)
                assert not np.array_equal(ch1, ch2)
    k_chains_raw = complex.get_raw_k_chains()
    for i, ch1 in enumerate(k_chains_raw.values()):
        for j, ch2 in enumerate(k_chains_raw.values()):
            if i != j:
                assert isinstance(ch1, np.ndarray)
                assert isinstance(ch2, np.ndarray)
                assert not np.array_equal(ch1, ch2) or (
                    np.array_equal(ch1, np.zeros(ch1.shape))
                    and np.array_equal(ch2, np.zeros(ch1.shape))
                )
    formal_sums = complex.k_chains_formal_sum()
    dataframe_chains = complex.get_df_k_chains()
    assert isinstance(dataframe_chains, pd.DataFrame)
    return True


def check_pair_kchains(complex1, complex2):
    assert isinstance(complex1, AbstractComplex)
    assert isinstance(complex2, AbstractComplex)
    k_chains1 = complex1.get_spectral_k_chains()
    k_chains_raw1 = complex1.get_raw_k_chains()
    formal_sums1 = complex1.k_chains_formal_sum()
    dataframe_chains1 = complex1.get_df_k_chains()
    k_chains2 = complex2.get_spectral_k_chains()
    k_chains_raw2 = complex2.get_raw_k_chains()
    formal_sums2 = complex2.k_chains_formal_sum()
    dataframe_chains2 = complex2.get_df_k_chains()
    for ch1 in k_chains1.values():
        for ch2 in k_chains2.values():
            try:
                assert isinstance(ch1, np.ndarray)
                assert isinstance(ch2, np.ndarray)
                assert not np.array_equal(ch1, ch2)
            except Exception as e:
                print(f"error is {e}")
                return False
    for ch1 in k_chains_raw1.values():
        for ch2 in k_chains_raw2.values():
            try:
                assert isinstance(ch1, np.ndarray)
                assert isinstance(ch2, np.ndarray)
                # assert not np.array_equal(ch1, ch2)
                assert not np.array_equal(ch1, ch2) or (
                    np.array_equal(ch1, np.zeros(ch1.shape))
                    and np.array_equal(ch2, np.zeros(ch1.shape))
                )
            except Exception as e:
                print(f"error is {e}")
                return False
    try:
        assert isinstance(dataframe_chains1, pd.DataFrame)
        assert isinstance(dataframe_chains2, pd.DataFrame)
    except:
        return False
    return True


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
ALL_SMILES = np.random.choice(a=ALL_SMILES, size=50, replace=False).tolist()


cases_pair_large = []
for s1 in ALL_SMILES:
    for s2 in ALL_SMILES:
        for m in modes:
            if s1 != s2:
                cases_pair_large.append((s1, s2, m))


@pytest.mark.parametrize("smile1,smile2,mode", cases_pair_large)
def test_small_kchain_pair_large(smile1, smile2, mode):
    pgs1 = PolyatomicGeometrySMILE(smile=smile1, target_dimension=3, mode=mode)
    pgs2 = PolyatomicGeometrySMILE(smile=smile2, target_dimension=3, mode=mode)
    pgs1 = pgs1.smiles_to_geom_complex()
    pgs2 = pgs2.smiles_to_geom_complex()
    if mode == "abstract":
        assert isinstance(pgs1, AbstractComplex)
        assert isinstance(pgs2, AbstractComplex)
        assert check_pair_kchains(pgs1, pgs2)
    else:
        raise Exception("INVALID + UNSUPPORTED")
