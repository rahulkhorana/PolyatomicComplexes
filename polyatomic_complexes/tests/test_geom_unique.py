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

    all_terms = set()
    for sub_adj1, sub_adj2 in zip(
        adj1["molecule_adjacencies"], adj2["molecule_adjacencies"]
    ):
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
