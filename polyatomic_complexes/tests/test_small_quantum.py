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
from polyatomic_complexes.src.complexes.quantum_complex import QuantumComplex
from polyatomic_complexes.src.complexes.quantum_theor_complex import QuantumWavesComplex


smiles = [
    "CC(=O)OC",
    "CCc1ccccn1",
]


@pytest.mark.parametrize("smile", smiles)
def test_small_test_polyatomic_geometry_cmplx(smile):
    pg = PolyatomicGeometrySMILE(smile, mode="quantum")
    quantum_mol = pg.smiles_to_geom_complex()
    assert isinstance(quantum_mol, QuantumComplex)
    quantum_mol.E0_elec_plus_zpe()


@pytest.mark.parametrize("smile", smiles)
def test_small_test_polyatomic_geometry(smile):
    pg = PolyatomicGeometrySMILE(smile, mode="quantum-waves")
    quantum_mol = pg.smiles_to_geom_complex()
    assert isinstance(quantum_mol, QuantumWavesComplex)
    quantum_mol.compute_long_range_interactions()
