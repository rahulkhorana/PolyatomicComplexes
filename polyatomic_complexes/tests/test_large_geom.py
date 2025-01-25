import os
import sys
import pytest

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


from test_geom import (
    check_abstract_complex_test,
    check_force_complex_test,
    check_quantum_complex_test,
    check_quantum_waves_complex_test,
)

from test_geom_unique import (
    check_adjacency_lists_unique,
    check_betti_numbers_unique,
    check_incidence_unique,
    check_dirac_unique,
    check_laplacians_unique,
    check_persistence_unique,
    check_skeleta_unique,
    check_coadjacency_unique,
)

modes = ["abstract", "force-field", "quantum", "quantum-waves"]


#### LARGE SANITY ####
large_cases = []


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
    elif mode == "quantum":
        assert isinstance(pgs, QuantumComplex)
        assert check_quantum_complex_test(pgs)
    elif mode == "quantum-waves":
        assert isinstance(pgs, QuantumWavesComplex)
        assert check_quantum_waves_complex_test(pgs)
    else:
        raise Exception("INVALID + UNSUPPORTED")


#### LARGE UNIQUE ####
large_cases_pair = []


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
