import os
import sys
import dill
import json
import numpy as np
import networkx as nx
import jax.numpy as jnp
from typing import List, Tuple, Optional
from pathlib import Path

# chemistry
from rdkit import Chem
from rdkit.Chem.rdchem import BondType

project_root = Path(__file__).parent.parent.parent.parent.resolve()
src_dir = project_root
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# all pc variations
from polyatomic_complexes.src.complexes.polyatomic_complex_cls import PolyatomicComplex
from polyatomic_complexes.src.complexes.force_complex import ForceComplex
from polyatomic_complexes.src.complexes.quantum_complex import QuantumComplex
from polyatomic_complexes.src.complexes.abstract_complex import AbstractComplex
from polyatomic_complexes.src.complexes.quantum_theor_complex import QuantumWavesComplex


class PolyatomicGeometrySMILE:
    def __init__(self, smile: str, target_dimension: int = 3, mode: str = "abstract"):
        """
        smile: this is supposed to be a valid smiles string
        target_dimension: this is an integer describing the dimension of the resulting polyatomic complex (can be anything > 0)
        mode: abstract |  force-field |  quantum-abstract | quantum-waves | material
        abstract: vanilla option
            - Constraints satisfied: generalizable/generality, invariances, uniqueness, cont/diff, efficient/time complexity
            - Constraints relaxed: geometric accuracy (electrostatics) -> less accurate than SOAP
        force-field: uses force fields to optimize for bond distances, angles
            - Constraints satisfied: generalizable/generality, invariances, uniqueness, cont/diff, efficient, long range interactions, geometric accuracy
            - Constraints relaxed: time complexity (quadratic)
        quantum-abstract: uses constrained DFT + B3LYP exchange-correlation functional to refine the geometry
            - Constraints satisfied: generalizable/generality, invariances, uniqueness, cont/diff, efficient, long range interactions, geometric accuracy
            - Constraints relaxed: time complexity
        quantum-waves: quantum abstract with wave functions computed
            - Constraints satisfied: generalizable/generality, invariances, uniqueness, cont/diff, efficient, long range interactions, geometric accuracy
            - Constraints relaxed: time complexity
        material: UNSUPPORTED with SMILES input
            - Please use the PolyatomicGeometryPYMATGEN class for this
        diffuse-spherical-harmonics:
            - Under development: adds diffuse functions and spherical harmonics
        """
        assert (
            isinstance(smile, str)
            and isinstance(target_dimension, int)
            and isinstance(mode, str)
        )
        self.smile = smile
        self.dim = target_dimension
        self.mode = mode
        self.bond_types = {
            BondType.SINGLE: ["SINGLE", 1.0],
            BondType.ONEANDAHALF: ["ONEANDAHALF", 1.5],
            BondType.DOUBLE: ["DOUBLE", 2.0],
            BondType.TWOANDAHALF: ["TWOANDAHALF", 2.5],
            BondType.TRIPLE: ["TRIPLE", 3.0],
            BondType.THREEANDAHALF: ["THREEANDAHALF", 3.5],
            BondType.QUADRUPLE: ["QUADRUPLE", 4.0],
            BondType.FOURANDAHALF: ["FOURANDAHALF", 4.5],
            BondType.QUINTUPLE: ["QUINTUPLE", 5.0],
            BondType.FIVEANDAHALF: ["FIVEANDAHALF", 5.5],
            BondType.HEXTUPLE: ["HEXTUPLE", 6.0],
            BondType.AROMATIC: ["AROMATIC", 7.0],
            BondType.IONIC: ["IONIC", 8.0],
            BondType.HYDROGEN: ["HYDROGEN", 9.0],
            BondType.THREECENTER: ["THREECENTER", 10.0],
            BondType.DATIVEONE: ["DATIVEONE", 11.0],
            BondType.DATIVE: ["DATIVE", 12.0],
            BondType.DATIVEL: ["DATIVEL", 13.0],
            BondType.DATIVER: ["DATIVER", 14.0],
            BondType.UNSPECIFIED: ["UNSPECIFIED", -1.0],
            BondType.OTHER: ["OTHER", -2.0],
            BondType.ZERO: ["ZERO", 0.0],
        }

    def extract_smile_info(self) -> Tuple[List, List]:
        molecule = Chem.MolFromSmiles(self.smile)
        if not molecule:
            raise ValueError("Invalid SMILES string")
        atoms = [atom.GetSymbol() for atom in molecule.GetAtoms()]
        bonds = []
        for bond in molecule.GetBonds():
            atom1 = bond.GetBeginAtom().GetSymbol()
            atom2 = bond.GetEndAtom().GetSymbol()
            bond_type = bond.GetBondType()
            if bond_type not in self.bond_types.keys():
                raise Exception("INVALID SMILES BONDING")
            bonds.append((atom1, atom2, self.bond_types[bond_type]))
        assert isinstance(atoms, list) and isinstance(bonds, list)
        return tuple([atoms, bonds])

    def smiles_to_geom_complex(self) -> Exception | PolyatomicComplex:
        try:
            atoms, bonds = self.extract_smile_info()
        except:
            raise Exception("INVALID SMILES STRING")
        if self.mode == "abstract":
            ac = AbstractComplex(self.smile, self.dim, atoms, bonds)
            return ac
        elif self.mode == "force-field":
            fc = ForceComplex(self.smile, self.dim, atoms, bonds)
            return fc
        elif self.mode == "quantum":
            qwc = QuantumComplex(self.smile, self.dim, atoms, bonds)
            return qwc
        elif self.mode == "quantum-waves":
            qu_abs = QuantumWavesComplex(self.smile, self.dim, atoms, bonds)
            return qu_abs
        elif self.mode == "material":
            raise Exception(
                "YOU ARE USING THE WRONG CLASS -> PolyatomicGeometryPYMATGEN is the correct one. This will be released in VERSION 2.0"
            )
        elif self.mode == "diffuse-spherical-harmonics":
            raise NotImplementedError("TO BE RELEASED IN VERSION 2.0")
        else:
            raise Exception("PLEASE SELECT A VALID MODE")


if __name__ == "__main__":
    pg = PolyatomicGeometrySMILE(smile="CC(=O)OC", mode="abstract")
    bonds, elements = pg.extract_smile_info()
    pg.smiles_to_geom_complex()
    print(bonds, elements)
    pg_force = PolyatomicGeometrySMILE(smile="CC(=O)OC", mode="force-field")
    pg_force.smiles_to_geom_complex()
    pg_quantum = PolyatomicGeometrySMILE(smile="CC(=O)OC", mode="quantum")
    pg_quantum.smiles_to_geom_complex()
    pg_waves = PolyatomicGeometrySMILE(smile="CC(=O)OC", mode="quantum-waves")
    pg_waves.smiles_to_geom_complex()
    print("*" * 10)
    print("DONE")
    print("*" * 10)
