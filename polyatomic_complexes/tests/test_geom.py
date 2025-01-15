import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


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
from scipy.sparse import coo_matrix
from networkx import Graph


def nice_print(argname, arg):
    print("*" * 20)
    print(f"The {argname} is")
    print(f"type is: {type(arg)}")
    if isinstance(arg, dict):
        print(arg.keys())
    if isinstance(arg, defaultdict):
        print(arg.keys())
    if isinstance(arg, list):
        print(f"internal size: {len(arg[0])}")
        print(f"internal tupe: {[type(x) for x in arg]}")
    if isinstance(arg, np.ndarray):
        print(f"internal size: {arg.shape}")
        print(f"internal size: {arg[0].shape}")
    if isinstance(arg, tuple):
        print(f"lengthis: {len(arg)}")
        print(f"internal tupe: {[type(x) for x in arg]}")
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
    rk, feats = get_ato
    assert isinstance(rk, list) and isinstance(feats, dict)
    assert set(
        [isinstance(r, tuple) and len(r) == 2 and isinstance(r[1], list) for r in rk]
    ) == set([True])
    # simple check bonds
    get_bonds = complex.get_bonds()
    assert isinstance(get_bonds, list)
    assert True in check_bonds(get_bonds) and len(check_bonds(get_bonds)) == 1
    # simple check betti
    get_betti = complex.get_betti_numbers()
    assert isinstance(get_betti, defaultdict)
    assert "molecule_betti_numbers" in get_betti
    assert get_betti["molecule_betti_numbers"] is None or isinstance(
        get_betti["molecule_betti_numbers"], list
    )
    # simple check incidence
    get_incidences = complex.get_incidence()
    assert isinstance(get_incidences, defaultdict)
    assert "molecule_incidence" in get_incidences
    for item in get_incidences["molecule_incidence"]:
        assert isinstance(item, dict) and len(item.keys()) > 0
        for key in item.keys():
            value = item[key]
            assert isinstance(value, dict)
            for particles in value:
                assert isinstance(particles, frozenset)
                assert len(particles) > 0
                for p in particles:
                    assert len(p) == 2
                    assert isinstance(p[0], str)
                    assert p[0].split("_")[0] in set(["E", "P", "N"])
                    assert isinstance(p[1], tuple)
                    if len(p[1]) == 2:
                        assert p[1][0] in set(["electron", "proton", "neutron"])
                        try:
                            arr = np.frombuffer(p[1][1], dtype=np.uint8)
                            assert arr.shape != 0
                        except:
                            raise Exception("invalid incidence")
                    elif len(p[1]) == 3:
                        assert p[1][0] in set(["electron", "proton", "neutron"])
                        try:
                            arr = np.frombuffer(p[1][1], dtype=np.uint8)
                            arr_w = np.frombuffer(p[1][2], dtype=np.uint8)
                            assert arr.shape != 0 and arr_w.shape != 0
                        except:
                            raise Exception("invalid incidence")
                    else:
                        assert p[1][0] in set(["electron", "proton", "neutron"])
                        try:
                            for _, items in enumerate(p[1][1:]):
                                arr = np.frombuffer(items, dtype=np.uint8)
                                assert arr.shape != 0
                        except:
                            raise Exception("INVALID")
    # simple abs complex
    get_abs_complex = complex.get_complex("abstract_complex")
    assert isinstance(get_abs_complex, defaultdict)
    for k in get_abs_complex.keys():
        assert len(k) == 2 and isinstance(k, tuple)
        assert isinstance(k[0], AtomComplex) and isinstance(k[1], str)

    # simple check rk complex
    get_rk_complex = complex.get_complex("rank_order")
    assert isinstance(get_rk_complex, defaultdict)
    assert "molecule" in get_rk_complex and len(get_rk_complex["molecule"]) == 2
    assert (
        "nuclear_structure" in get_rk_complex
        and len(get_rk_complex["nuclear_structure"]) == 2
    )
    assert (
        "electronic_structure" in get_rk_complex
        and len(get_rk_complex["electronic_structure"]) == 2
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
            required_features[i] in get_rk_complex[column][1]
            for i in range(len(required_features))
        ]
    )
    assert isinstance(get_rk_complex["molecule"][0], CombinatorialComplex)
    assert isinstance(get_rk_complex["molecule"][1], defaultdict)
    assert True in check_features("molecule") and len(check_features("molecule")) == 1
    assert isinstance(get_rk_complex["nuclear_structure"][0], CombinatorialComplex)
    assert isinstance(get_rk_complex["nuclear_structure"][1], defaultdict)
    assert (
        True in check_features("nuclear_structure")
        and len(check_features("nuclear_structure")) == 1
    )
    assert isinstance(get_rk_complex["electronic_structure"][0], CombinatorialComplex)
    assert isinstance(get_rk_complex["electronic_structure"][1], defaultdict)
    assert (
        True in check_features("electronic_structure")
        and len(check_features("electronic_structure")) == 1
    )
    # simple check dirac
    get_dirac = complex.get_dirac()
    assert isinstance(get_dirac, defaultdict)
    assert "molecule_dirac" in get_dirac
    assert (
        get_dirac["molecule_dirac"] is None
        or isinstance(get_dirac["molecule_dirac"], np.ndarray)
        or isinstance(get_dirac["molecule_dirac"], coo_matrix)
        or isinstance(get_dirac["molecule_dirac"], list)
    )
    if isinstance(get_dirac["molecule_dirac"], list):
        try:
            lst = np.asarray(get_dirac["molecule_dirac"][0])
            assert lst.shape != 0
        except:
            raise Exception("unconverted")
    # simple check orientations
    get_def_or = complex.get_default_orientations()
    assert isinstance(get_def_or, dict) or isinstance(get_def_or, int)
    # simple check laplacians
    get_lap = complex.get_laplacians()
    assert isinstance(get_lap, defaultdict)
    assert "molecule_laplacians" in get_lap
    for sub_lap1 in get_lap["molecule_laplacians"]:
        for term1 in sub_lap1:
            assert isinstance(term1[0], str)
            assert isinstance(term1[1], np.ndarray)
    # simple check stacked
    get_pc = complex.get_pc_matrix("stacked")
    assert isinstance(get_pc, list)
    assert len(get_pc) > 0
    assert isinstance(get_pc[0], Graph)
    # simple check persistence
    get_pers = complex.get_persistence()
    assert isinstance(get_pers, defaultdict)
    assert get_pers["molecule_persistence"] is None or isinstance(
        get_pers["molecule_persistence"], list
    )
    # simple check skeleta
    get_sk = complex.get_skeleta()
    assert isinstance(get_sk, defaultdict)
    assert "molecule_skeleta" in get_sk
    assert (
        isinstance(get_sk["molecule_skeleta"], list)
        and len(get_sk["molecule_skeleta"]) > 0
        and len(get_sk["molecule_skeleta"][0]) > 0
    )
    # simple check coadjacency
    get_coadj = complex.get_coadjacencies()
    assert isinstance(get_coadj, defaultdict)
    assert "molecule_co_adjacencies" in get_coadj
    for sub_adj1 in get_coadj["molecule_co_adjacencies"]:
        for term1 in sub_adj1:
            assert isinstance(term1[0], str)
            assert isinstance(term1[1], np.ndarray)
    return True


def check_force_complex_test(complex: ForceComplex):
    forces = complex.forces()
    assert isinstance(forces, np.ndarray)
    assert forces.shape != 0 and forces.shape[1] == 3
    electrostatics = complex.electrostatics()
    assert isinstance(electrostatics, np.ndarray)
    assert electrostatics.shape != 0 and electrostatics.shape[0] == forces.shape[0]
    return True


def check_quantum_complex_test(complex: QuantumComplex):
    # quick checking forces
    complex.forces()
    # quick checking electrostatics
    complex.electrostatics()
    # quick checking distances
    complex.distances()
    # quick checking positions
    complex.positions()
    # quick checking homo-lumo
    complex.homo_lumo_gap()
    # quick checking dipole
    complex.dipole_moment()
    # quick checking effective potential
    complex.effective_potential()
    # quick checking orbital magnetic moments
    complex.orbital_magnetic_moments()
    # quick checking wavefunctions
    complex.wavefunctions()
    # quick checking forces
    complex.get_forces()
    # quick checking electrostatics
    complex.get_electrostatics()
    # quick checking distances
    complex.get_distances()
    # quick checking positions
    complex.get_positions()
    # quick checking homo-lumo
    complex.get_homo_lumo_gap()
    # quick checking dipole moment
    complex.get_dipole_moment()
    # quick checking effective potential
    complex.get_effective_potential()
    # quick checking omm
    complex.get_orbital_magnetic_moments()
    # quick checking wavefunctions
    complex.get_wavefunctions()
    # quick checking potential energy
    complex.get_potential_energy()
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


test_small_test_polyatomic_geometry(smile=smiles[0], mode=modes[2])
# test_small_test_polyatomic_geometry(smile=smiles[1], mode=modes[0])
