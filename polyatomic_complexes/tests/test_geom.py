from polyatomic_complexes.src.complexes.polyatomic_geometry import (
    PolyatomicGeometrySMILE,
)

from polyatomic_complexes.src.complexes.atomic_complex import AtomComplex
from polyatomic_complexes.src.complexes.force_complex import ForceComplex
from polyatomic_complexes.src.complexes.quantum_complex import QuantumComplex
from polyatomic_complexes.src.complexes.abstract_complex import AbstractComplex
from polyatomic_complexes.src.complexes.quantum_theor_complex import QuantumWavesComplex


from polyatomic_complexes.src.complexes.space_utils import (
    geometricPolyatomicComplex,
    geometricAtomicComplex,
)

from typing import List
import os
import pytest
import json
import random
import numpy as np
from collections import defaultdict
from toponetx import CombinatorialComplex


def nice_print(argname, arg):
    print("*" * 20)
    print(f"The {argname}")
    print("*" * 20)


def check_abstract_complex_test(complex: AbstractComplex):
    # simple check abstract complex
    complex_ac = complex.abstract_complex()
    assert isinstance(complex_ac, defaultdict)
    for k in complex_ac.keys():
        assert len(k) == 2 and isinstance(k, tuple)
        assert isinstance(k[0], AtomComplex) and isinstance(k[1], str)
    # simple check rank order complex
    complex_roc = complex.rank_order_complex()
    assert isinstance(complex_roc, defaultdict)
    assert "molecule" in complex_roc and len(complex_roc["molecule"]) == 2
    assert (
        "nuclear_structure" in complex_roc
        and len(complex_roc["nuclear_structure"]) == 2
    )
    assert (
        "electronic_structure" in complex_roc
        and len(complex_roc["electronic_structure"]) == 2
    )
    required_features = [
        "incidence",
        "laplacians",
        "adjacencies",
        "co_adjacencies",
        "skeleta",
        "all_cell_coadj",
        "dirac",
    ]
    check_features = lambda column: set(
        [
            required_features[i] in complex_roc[column][1]
            for i in range(len(required_features))
        ]
    )
    assert isinstance(complex_roc["molecule"][0], CombinatorialComplex)
    assert isinstance(complex_roc["molecule"][1], defaultdict)
    assert True in check_features("molecule") and len(check_features("molecule")) == 1
    assert isinstance(complex_roc["nuclear_structure"][0], CombinatorialComplex)
    assert isinstance(complex_roc["nuclear_structure"][1], defaultdict)
    assert (
        True in check_features("nuclear_structure")
        and len(check_features("nuclear_structure")) == 1
    )
    assert isinstance(complex_roc["electronic_structure"][0], CombinatorialComplex)
    assert isinstance(complex_roc["electronic_structure"][1], defaultdict)
    assert (
        True in check_features("electronic_structure")
        and len(check_features("electronic_structure")) == 1
    )
    # simple check atomic structure
    check_types = lambda column: set(
        [
            len(x) == 3
            and isinstance(x[0], defaultdict)
            and isinstance(x[2], defaultdict)
            for x in column
        ]
    )
    atm_struct = complex.atomic_structure()
    assert isinstance(atm_struct, list)
    assert True in check_types(atm_struct) and len(check_types(atm_struct)) == 1
    # simple check bonds
    bonds = complex.bonds()
    check_bonds = lambda bnds: set(
        [
            len(b) == 3
            and isinstance(b, tuple)
            and isinstance(b[0], str)
            and isinstance(b[1], str)
            and isinstance(b[2], list)
            for b in bnds
        ]
    )
    assert isinstance(bonds, list)
    assert True in check_bonds(bonds) and len(check_bonds(bonds)) == 1
    # simple check not implemented methods
    expected_to_NI_methods = [
        complex.forces,
        complex.electrostatics,
        complex.get_forces,
        complex.get_electrostatics,
        complex.wavefunctions,
    ]
    for method in expected_to_NI_methods:
        with pytest.raises(NotImplementedError) as excinfo:
            method()
        assert (
            str(excinfo.value)
            == "This is not defined behavior for an Abstract Complex!"
        )
    # simple check atomic topology
    ato_top = complex.atomic_topology()
    rk, feats = ato_top
    assert isinstance(rk, list) and isinstance(feats, dict)
    assert set(
        [isinstance(r, tuple) and len(r) == 2 and isinstance(r[1], list) for r in rk]
    ) == set([True])
    # simple check atomic structure
    ato_struct = complex.atomic_structure()
    assert isinstance(ato_struct, list)
    adj = complex.get_adjacencies()
    assert isinstance(adj, defaultdict)
    assert "molecule_adjacencies" in adj
    for sub_adj1 in adj["molecule_adjacencies"]:
        for term1 in sub_adj1:
            assert isinstance(term1[0], str)
            assert isinstance(term1[1], np.ndarray)
    # simple check coadjacency (all cell)
    all_cell_coaj = complex.get_all_cell_coadj()
    nice_print("all_cell_coaj", all_cell_coaj)
    print(f"all cell is: {all_cell_coaj}")
    assert isinstance(all_cell_coaj, defaultdict)
    assert "molecule_all_cell_coadj" in all_cell_coaj
    assert isinstance(all_cell_coaj["molecule_all_cell_coadj"], list)
    assert isinstance(all_cell_coaj["molecule_all_cell_coadj"][0], list)
    try:
        all_cell = all_cell_coaj["molecule_all_cell_coadj"]
        data = [np.asarray(row) for row in all_cell]
        as_arr = np.asarray(data, dtype=object)
        assert isinstance(as_arr, np.ndarray)
        for c in as_arr:
            assert isinstance(c, np.ndarray)
    except Exception as e:
        raise e
    # simple check atomic structure
    get_structure = complex.get_atomic_structure()
    assert isinstance(get_structure, list) and len(get_structure) != 1
    # simple check atotop
    get_ato = complex.get_atomic_topology()
    nice_print("get_ato", get_ato)

    get_bonds = complex.get_bonds()
    nice_print("get_bonds", get_bonds)

    get_betti = complex.get_betti_numbers()
    nice_print("get_betti", get_betti)

    get_incidences = complex.get_incidence()
    nice_print("get_incidences", get_incidences)

    get_abs_complex = complex.get_complex("abstract_complex")
    nice_print("get_abs_complex", get_abs_complex)

    get_rk_complex = complex.get_complex("rank_order")
    nice_print("get_rk_complex", get_rk_complex)

    get_dirac = complex.get_dirac()
    nice_print("get_dirac", get_dirac)

    get_def_or = complex.get_default_orientations()
    nice_print("get_def_or", get_def_or)

    get_lap = complex.get_laplacians()
    nice_print("get_lap", get_lap)

    get_pc = complex.get_pc_matrix("stacked")
    nice_print("get_pc", get_pc)

    get_pers = complex.get_persistence()
    nice_print("get_pers", get_pers)

    get_sk = complex.get_skeleta()
    nice_print("get_sk", get_sk)
    return True


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
    for sub_adj1, sub_adj2 in zip(
        adj1["molecule_adjacencies"], adj2["molecule_adjacencies"]
    ):
        for term1, term2 in zip(sub_adj1, sub_adj2):
            assert isinstance(term1[0], str)
            assert isinstance(term2[0], str)
            assert isinstance(term1[1], np.ndarray)
            assert isinstance(term2[1], np.ndarray)
            arr_1, arr_2 = term1[1], term2[1]
            assert not np.array_equal(arr_1, arr_2)
    return True


def check_all_cell_unique(complex1, complex2):
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

    def convert_nd(coaj):
        data = [np.asarray(row) for row in coaj]
        as_arr = np.asarray(data, dtype=object)
        return as_arr

    all_cell_coaj1 = complex1.get_all_cell_coadj()
    all_cell_coaj2 = complex2.get_all_cell_coadj()
    assert isinstance(all_cell_coaj1, defaultdict)
    assert "molecule_all_cell_coadj" in all_cell_coaj1
    assert isinstance(all_cell_coaj2, defaultdict)
    assert "molecule_all_cell_coadj" in all_cell_coaj2
    all_c_1 = all_cell_coaj1["molecule_all_cell_coadj"]
    all_c_2 = all_cell_coaj1["molecule_all_cell_coadj"]
    arr_1, arr_2 = convert_nd(all_c_1), convert_nd(all_c_2)
    assert isinstance(arr_1, np.ndarray)
    assert isinstance(arr_2, np.ndarray)
    for row1, row2 in zip(arr_1, arr_2):
        assert isinstance(row1, np.ndarray)
        assert isinstance(row2, np.ndarray)
        assert not np.array_equal(arr_1, arr_2)
    return True


def check_force_complex_test(complex: ForceComplex):
    return True


def check_quantum_complex_test(complex: QuantumComplex):
    return True


def check_quantum_waves_complex_test(complex: QuantumWavesComplex):
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


cases = []
for s in smiles:
    for m in modes:
        cases.append((s, m))


@pytest.mark.parametrize("smile,mode", cases)
def test_small_test_polyatomic_geometry(smile, mode):
    pgs = PolyatomicGeometrySMILE(smile=smile, target_dimension=3, mode=mode)
    pgs = pgs.smiles_to_geom_complex()
    if mode == "abstract":
        assert isinstance(pgs, AbstractComplex)
        assert check_abstract_complex_test(pgs)
    elif mode == "force-field":
        assert isinstance(pgs, ForceComplex)
        assert check_force_complex_test(pgs)
    elif mode == "quantum":
        assert isinstance(pgs, QuantumComplex)
        assert check_quantum_complex_test(pgs)
    elif mode == "quantum-waves":
        assert isinstance(pgs, QuantumWavesComplex)
        assert check_quantum_waves_complex_test(pgs)
    else:
        raise Exception("INVALID + UNSUPPORTED")


# test_small_test_polyatomic_geometry(smile=smiles[0], mode=modes[0])
# test_small_test_polyatomic_geometry(smile=smiles[1], mode=modes[0])
