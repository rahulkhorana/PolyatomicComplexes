import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


from polyatomic_complexes.src.complexes.polyatomic_geometry import (
    PolyatomicGeometrySMILE,
)

from polyatomic_complexes.src.complexes.force_complex import ForceComplex
from polyatomic_complexes.src.complexes.quantum_complex import QuantumComplex
from polyatomic_complexes.src.complexes.abstract_complex import AbstractComplex
from polyatomic_complexes.src.complexes.quantum_theor_complex import QuantumWavesComplex


from typing import List
import os
import pytest
import json
import random
import numpy as np
from collections import defaultdict
from scipy.sparse import coo_matrix


def check_adjacency_lists_unique(complex1, complex2):
    assert (
        isinstance(complex1, AbstractComplex)
        or isinstance(complex1, ForceComplex)
        or isinstance(complex1, QuantumComplex)
        or isinstance(complex1, QuantumWavesComplex)
    )
    assert (
        isinstance(complex2, AbstractComplex)
        or isinstance(complex2, ForceComplex)
        or isinstance(complex2, QuantumComplex)
        or isinstance(complex2, QuantumWavesComplex)
    )
    adj1, adj2 = complex1.get_adjacencies(), complex2.get_adjacencies()
    assert isinstance(adj1, defaultdict)
    assert isinstance(adj2, defaultdict)
    assert "molecule_adjacencies" in adj1
    assert "molecule_adjacencies" in adj2
    mol_adj1 = adj1["molecule_adjacencies"]
    mol_adj2 = adj2["molecule_adjacencies"]
    all_terms = set()
    for sub_adj1, sub_adj2 in zip(mol_adj1, mol_adj2):
        for term1, term2 in zip(sub_adj1, sub_adj2):
            assert isinstance(term1[0], str)
            assert isinstance(term2[0], str)
            assert isinstance(term1[1], np.ndarray)
            assert isinstance(term2[1], np.ndarray)
            arr_1, arr_2 = term1[1], term2[1]
            v = np.array_equal(arr_1, arr_2)
            all_terms.add(v)
            if False in all_terms:
                return True
    return False


def check_betti_numbers_unique(complex1, complex2):
    """
    NOTE: default mode is not persistence features -> so these will be empty
    - to check persistence -> have to manually set the mode to persistence inside .get_comb_features()
    - this is not default because it is computationally expensive.
    """
    assert (
        isinstance(complex1, AbstractComplex)
        or isinstance(complex1, ForceComplex)
        or isinstance(complex1, QuantumComplex)
        or isinstance(complex1, QuantumWavesComplex)
    )
    assert (
        isinstance(complex2, AbstractComplex)
        or isinstance(complex2, ForceComplex)
        or isinstance(complex2, QuantumComplex)
        or isinstance(complex2, QuantumWavesComplex)
    )
    get_betti_1 = complex1.get_betti_numbers()
    get_betti_2 = complex2.get_betti_numbers()
    assert isinstance(get_betti_1, defaultdict)
    assert isinstance(get_betti_2, defaultdict)
    assert (
        "molecule_betti_numbers" in get_betti_1
        and "molecule_betti_numbers" in get_betti_2
    )
    complex_1_betti = get_betti_1["molecule_betti_numbers"]
    complex_2_betti = get_betti_2["molecule_betti_numbers"]

    assert complex_1_betti is None or isinstance(complex_1_betti, list)
    assert complex_2_betti is None or isinstance(complex_2_betti, list)
    if (
        isinstance(complex_2_betti, list)
        and isinstance(complex_2_betti, list)
        and len(complex_1_betti) > 0
        and len(complex_2_betti) > 0
    ):
        to_np_1 = np.asarray(complex_1_betti)
        to_np_2 = np.asarray(complex_2_betti)
        if to_np_1.shape == to_np_2.shape and 0 not in set(to_np_1.shape):
            v = np.array_equal(to_np_1, to_np_2)
            assert not v
        elif 0 not in set(to_np_1.shape) and 0 not in set(to_np_2.shape):
            v = np.array_equal(to_np_1, to_np_2)
            assert not v
    return True


def check_incidence_unique(complex1, complex2):
    assert (
        isinstance(complex1, AbstractComplex)
        or isinstance(complex1, ForceComplex)
        or isinstance(complex1, QuantumComplex)
        or isinstance(complex1, QuantumWavesComplex)
    )
    assert (
        isinstance(complex2, AbstractComplex)
        or isinstance(complex2, ForceComplex)
        or isinstance(complex2, QuantumComplex)
        or isinstance(complex2, QuantumWavesComplex)
    )
    incidence_1 = complex1.get_incidence()
    incidence_2 = complex2.get_incidence()
    assert isinstance(incidence_1, defaultdict)
    assert isinstance(incidence_2, defaultdict)
    assert "molecule_incidence" in incidence_1
    assert "molecule_incidence" in incidence_2
    for im1, im2 in zip(
        incidence_1["molecule_incidence"], incidence_2["molecule_incidence"]
    ):
        assert isinstance(im1, dict) and len(im1.keys()) > 0
        assert isinstance(im2, dict) and len(im2.keys()) > 0
        for key1, key2 in zip(im1.keys(), im2.keys()):
            value1 = im1[key1]
            value2 = im2[key2]
            assert isinstance(value1, dict)
            assert isinstance(value2, dict)
            for particles1, particles2 in zip(value1, value2):
                assert isinstance(particles1, frozenset)
                assert isinstance(particles2, frozenset)
                assert len(particles1) > 0
                assert len(particles2) > 0
                for p1, p2 in zip(particles1, particles2):
                    assert len(p1) == 2
                    assert len(p2) == 2
                    assert isinstance(p1[0], str)
                    assert isinstance(p2[0], str)
                    assert p1[0].split("_")[0] in set(["E", "P", "N"])
                    assert p2[0].split("_")[0] in set(["E", "P", "N"])
                    assert isinstance(p1[1], tuple)
                    assert isinstance(p2[1], tuple)
                    if len(p1[1]) == 2:
                        assert p1[1][0] in set(["electron", "proton", "neutron"])
                        assert p2[1][0] in set(["electron", "proton", "neutron"])
                        try:
                            arr1 = np.frombuffer(p1[1][1], dtype=np.uint8)
                            arr2 = np.frombuffer(p2[1][1], dtype=np.uint8)
                            assert arr1.shape != 0 and arr2.shape != 0
                            assert not np.array_equal(arr1, arr2)
                        except:
                            raise Exception("invalid incidence")
                    elif len(p1[1]) == 3:
                        assert p1[1][0] in set(["electron", "proton", "neutron"])
                        assert p2[1][0] in set(["electron", "proton", "neutron"])
                        try:
                            arr1 = np.frombuffer(p1[1][1], dtype=np.uint8)
                            arr2 = np.frombuffer(p2[1][1], dtype=np.uint8)
                            assert arr1.shape != 0 and arr2.shape != 0
                            assert not np.array_equal(arr1, arr2)
                            try:
                                arr1_w = np.frombuffer(p1[1][2], dtype=np.uint8)
                                arr2_w = np.frombuffer(p2[1][2], dtype=np.uint8)
                                assert arr2_w.shape != 0 and arr1_w.shape != 0
                                assert not np.array_equal(arr1_w, arr2_w)
                            except IndexError:
                                print("not electron pair")
                                pass
                        except:
                            raise Exception("invalid incidence")
                    else:
                        assert p1[1][0] in set(["electron", "proton", "neutron"])
                        assert p2[1][0] in set(["electron", "proton", "neutron"])
                        try:
                            for _, items in enumerate(p1[1][1:]):
                                arr1 = np.frombuffer(items, dtype=np.uint8)
                                assert arr1.shape != 0
                            for _, items in enumerate(p2[1][1:]):
                                arr2 = np.frombuffer(items, dtype=np.uint8)
                                assert arr2.shape != 0
                        except:
                            raise Exception("INVALID")
    return True


def check_dirac_unique(complex1, complex2):
    assert (
        isinstance(complex1, AbstractComplex)
        or isinstance(complex1, ForceComplex)
        or isinstance(complex1, QuantumComplex)
        or isinstance(complex1, QuantumWavesComplex)
    )
    assert (
        isinstance(complex2, AbstractComplex)
        or isinstance(complex2, ForceComplex)
        or isinstance(complex2, QuantumComplex)
        or isinstance(complex2, QuantumWavesComplex)
    )
    dirac_1 = complex1.get_dirac()
    dirac_2 = complex2.get_dirac()
    assert isinstance(dirac_1, defaultdict)
    assert isinstance(dirac_2, defaultdict)
    assert "molecule_dirac" in dirac_1
    assert "molecule_dirac" in dirac_2
    mol_dirac_1 = dirac_1["molecule_dirac"]
    mol_dirac_2 = dirac_2["molecule_dirac"]
    assert (
        mol_dirac_1 is None
        or isinstance(mol_dirac_1, np.ndarray)
        or isinstance(mol_dirac_1, coo_matrix)
        or isinstance(mol_dirac_1, list)
    )
    assert (
        mol_dirac_2 is None
        or isinstance(mol_dirac_2, np.ndarray)
        or isinstance(mol_dirac_2, coo_matrix)
        or isinstance(mol_dirac_2, list)
    )
    if isinstance(mol_dirac_1, list):
        try:
            mol_dirac_1 = np.asarray(mol_dirac_1[0])
            assert mol_dirac_1.shape != 0
        except:
            raise Exception("unconverted")

    if isinstance(mol_dirac_1, coo_matrix):
        try:
            mol_dirac_1 = mol_dirac_1.toarray()
            assert isinstance(mol_dirac_1, np.ndarray)
            assert mol_dirac_1.shape != 0
        except:
            raise Exception("unconverted")
    if isinstance(mol_dirac_2, list):
        try:
            mol_dirac_2 = np.asarray(mol_dirac_2[0])
            assert mol_dirac_2.shape != 0
        except:
            raise Exception("unconverted")
    if isinstance(mol_dirac_2, coo_matrix):
        try:
            mol_dirac_2 = mol_dirac_2.toarray()
            assert isinstance(mol_dirac_2, np.ndarray)
            assert mol_dirac_2.shape != 0
        except:
            raise Exception("unconverted")
    if isinstance(mol_dirac_1, np.ndarray) and isinstance(mol_dirac_2, np.ndarray):
        assert not np.array_equal(mol_dirac_1, mol_dirac_2)
        assert not np.array_equal(mol_dirac_1, np.zeros(mol_dirac_1.shape))
        assert not np.array_equal(mol_dirac_2, np.zeros(mol_dirac_2.shape))
    return True


def check_laplacians_unique(complex1, complex2):
    assert (
        isinstance(complex1, AbstractComplex)
        or isinstance(complex1, ForceComplex)
        or isinstance(complex1, QuantumComplex)
        or isinstance(complex1, QuantumWavesComplex)
    )
    assert (
        isinstance(complex2, AbstractComplex)
        or isinstance(complex2, ForceComplex)
        or isinstance(complex2, QuantumComplex)
        or isinstance(complex2, QuantumWavesComplex)
    )
    lap_1 = complex1.get_laplacians()
    lap_2 = complex2.get_laplacians()
    assert isinstance(lap_1, defaultdict)
    assert isinstance(lap_2, defaultdict)
    assert "molecule_laplacians" in lap_1
    assert "molecule_laplacians" in lap_2
    mol_lap_1 = lap_1["molecule_laplacians"]
    mol_lap_2 = lap_2["molecule_laplacians"]
    for sub_lap1, sub_lap2 in zip(mol_lap_1, mol_lap_2):
        for term1, term2 in zip(sub_lap1, sub_lap2):
            assert isinstance(term1[0], str)
            assert isinstance(term2[0], str)
            assert isinstance(term1[1], np.ndarray)
            assert isinstance(term2[1], np.ndarray)
            assert not np.array_equal(term1[1], term2[1])
    return True


def check_persistence_unique(complex1, complex2):
    """
    NOTE: by default mode is not persistence features -> so these will be empty
    - to check persistence -> have to manually set the mode to persistence inside .get_comb_features()
    - this is not default because it is computationally expensive.
    """
    assert (
        isinstance(complex1, AbstractComplex)
        or isinstance(complex1, ForceComplex)
        or isinstance(complex1, QuantumComplex)
        or isinstance(complex1, QuantumWavesComplex)
    )
    assert (
        isinstance(complex2, AbstractComplex)
        or isinstance(complex2, ForceComplex)
        or isinstance(complex2, QuantumComplex)
        or isinstance(complex2, QuantumWavesComplex)
    )
    pers1 = complex1.get_persistence()
    pers2 = complex2.get_persistence()
    assert isinstance(pers1, defaultdict)
    assert isinstance(pers2, defaultdict)
    assert "molecule_persistence" in pers1
    assert "molecule_persistence" in pers2
    get_pers1 = pers1["molecule_persistence"]
    get_pers2 = pers2["molecule_persistence"]
    assert get_pers1 is None or isinstance(get_pers1, list)
    assert get_pers2 is None or isinstance(get_pers2, list)
    try:
        get_pers1 = np.asarray(get_pers1)
        get_pers2 = np.asarray(get_pers2)
        if 0 not in set(get_pers2.shape) and 0 not in set(get_pers1.shape):
            assert len(get_pers1) > 0 and len(get_pers2) > 0
            assert not np.array_equal(get_pers1, get_pers2)
    except:
        return False
    return True


def check_skeleta_unique(complex1, complex2):
    assert (
        isinstance(complex1, AbstractComplex)
        or isinstance(complex1, ForceComplex)
        or isinstance(complex1, QuantumComplex)
        or isinstance(complex1, QuantumWavesComplex)
    )
    assert (
        isinstance(complex2, AbstractComplex)
        or isinstance(complex2, ForceComplex)
        or isinstance(complex2, QuantumComplex)
        or isinstance(complex2, QuantumWavesComplex)
    )
    sk1 = complex1.get_skeleta()
    sk2 = complex2.get_skeleta()
    assert isinstance(sk1, defaultdict)
    assert isinstance(sk2, defaultdict)
    assert "molecule_skeleta" in sk1
    assert "molecule_skeleta" in sk2
    mol_sk_1 = sk1["molecule_skeleta"]
    mol_sk_2 = sk2["molecule_skeleta"]
    assert isinstance(mol_sk_1, list) and isinstance(mol_sk_2, list)
    assert len(mol_sk_1) > 0 and len(mol_sk_2) > 0
    assert len(mol_sk_1[0]) > 0 and len(mol_sk_2[0]) > 0
    for sub_sk1, sub_sk2 in zip(mol_sk_1, mol_sk_2):
        assert not np.array_equal(sub_sk1, sub_sk2)
    return True


def check_coadjacency_unique(complex1, complex2):
    assert (
        isinstance(complex1, AbstractComplex)
        or isinstance(complex1, ForceComplex)
        or isinstance(complex1, QuantumComplex)
        or isinstance(complex1, QuantumWavesComplex)
    )
    assert (
        isinstance(complex2, AbstractComplex)
        or isinstance(complex2, ForceComplex)
        or isinstance(complex2, QuantumComplex)
        or isinstance(complex2, QuantumWavesComplex)
    )
    coadj1 = complex1.get_coadjacencies()
    coadj2 = complex2.get_coadjacencies()
    assert isinstance(coadj1, defaultdict)
    assert isinstance(coadj2, defaultdict)
    assert "molecule_co_adjacencies" in coadj1
    assert "molecule_co_adjacencies" in coadj2
    mol_coadj1 = coadj1["molecule_co_adjacencies"]
    mol_coadj2 = coadj2["molecule_co_adjacencies"]
    terms = set()
    is_zero = set()
    for sub_coadj1, sub_coadj2 in zip(mol_coadj1, mol_coadj2):
        for term1, term2 in zip(sub_coadj1, sub_coadj2):
            assert isinstance(term1[0], str)
            assert isinstance(term2[0], str)
            assert isinstance(term1[1], np.ndarray)
            assert isinstance(term2[1], np.ndarray)
            v = np.array_equal(term1[1], term2[1])
            terms.add(v)
            if v:
                is_zero.add(np.array_equal(term1[1], np.zeros(term1[1].shape)))
    assert False in terms or (len(is_zero) == 1 and True in is_zero)
    return True


smiles = [
    "OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O",
    "Cc1occc1C(=O)Nc2ccccc2",
    "c1ccc2c(c1)ccc3c2ccc4c5ccccc5ccc43",
    "CCc1ccccn1",
    "COc1ccc(cc1)N2CCN(CC2)C(=O)[C@@H]3CCCC[C@H]3C(=O)NCC#N",
    "Cn1c(CN2CCN(CC2)c3ccc(Cl)cc3)nc4ccccc14",
]
modes = ["abstract", "force-field", "quantum", "quantum-waves"]


cases_pair = []
for sm1 in smiles:
    for sm2 in smiles:
        if sm1 != sm2:
            for mde in modes:
                _case = (sm1, sm2, mde)
                cases_pair.append(_case)


@pytest.mark.parametrize("smile1,smile2,mode", cases_pair)
def test_small_unique_battery(smile1, smile2, mode):
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


sm1 = "CCOC(=O)c1ccccc1c2csc(NS(=O)(=O)c3ccc(Cl)cc3)n2"
sm2 = "C[C@H]1O[C@H]([C@H](O)[C@@H]1O)n2cnc3c(N)nc(OC4CC5CC5C4)nc23"
test_small_unique_battery(sm1, sm2, "abstract")
