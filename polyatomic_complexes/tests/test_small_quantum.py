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
    "OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O",
    "Cc1occc1C(=O)Nc2ccccc2",
    "c1ccc2c(c1)ccc3c2ccc4c5ccccc5ccc43",
    "CCc1ccccn1",
    "COc1ccc(cc1)N2CCN(CC2)C(=O)[C@@H]3CCCC[C@H]3C(=O)NCC#N",
    "Cn1c(CN2CCN(CC2)c3ccc(Cl)cc3)nc4ccccc14",
]


# @pytest.mark.parametrize("smile", smiles)
def test_small_test_polyatomic_geometry(smile):
    pg = PolyatomicGeometrySMILE(smile, mode="quantum")
    quantum_mol = pg.smiles_to_geom_complex()
    assert isinstance(quantum_mol, QuantumComplex)
    quantum_mol.E0_elec_plus_zpe()


# @pytest.mark.parametrize("smile", smiles)
def test_small_test_polyatomic_geometry(smile):
    pg = PolyatomicGeometrySMILE(smile, mode="quantum-waves")
    quantum_mol = pg.smiles_to_geom_complex()
    assert isinstance(quantum_mol, QuantumWavesComplex)
    quantum_mol.compute_long_range_interactions()
