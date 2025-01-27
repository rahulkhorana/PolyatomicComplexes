import os
import sys
import json
import numpy as np
from typing import List
from pathlib import Path
from collections import defaultdict
from scipy.spatial import distance_matrix

BASE_PATH = Path(__file__)
project_root = BASE_PATH.parent.parent.parent.parent.resolve()
src_dir = project_root
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# src
from polyatomic_complexes.src.complexes.abstract_complex import AbstractComplex

# rdkit
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

# ase and pyscf
from ase import Atoms
from pyscf import gto, dft, grad

# tnx
from toponetx import CombinatorialComplex


class ForceComplex(AbstractComplex):
    def __init__(self, smile, target_dimension, atoms, bonds):
        super().__init__(smile, target_dimension, atoms, bonds)
        self.smile = smile
        self.dim = target_dimension
        self.atoms = atoms
        self.bnds = bonds
        self.roc = self.rank_order_complex()
        self.lookup_fp = BASE_PATH.parent.parent.parent.__str__() + "/dataset/construct"
        with open(self.lookup_fp + "/basis_sets.json", "rb") as f:
            self.basis_sets = json.load(f)
        self.gto = self._build_gto()

    def unpack_roc(self):
        self._molecule, self._molecule_feat = self.roc["molecule"]
        self._nucleus, self._nucleus_feat = self.roc["nuclear_structure"]
        self._electrons, self._electron_feat = self.roc["electronic_structure"]
        return

    def _build_gto(self) -> gto.Mole:
        """
        Outputs a PySCF-compatible molecule.
        """
        mol = Chem.MolFromSmiles(self.smile)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        atoms = mol.GetAtoms()
        coordinates = mol.GetConformers()[0].GetPositions()
        symbols = [atom.GetSymbol() for atom in atoms]
        mole = gto.M(
            atom=[[symbols[i], *coordinates[i]] for i in range(len(atoms))],
            basis=self.basis_sets,
        )
        self.gto = mole
        return mole

    def electrostatics(self) -> np.ndarray:
        """
        returns the electrostatic potential
        """
        if not hasattr(self, "gto"):
            self._build_gto()
        mf = dft.RKS(self.gto).density_fit()
        mf.xc = "b3lyp"
        mf.kernel(nproc=4)
        veff = mf.get_veff()
        return np.asarray(veff)

    def compute_top_weight(self, atom_i, atom_j, topology_data):
        weight = 1.0
        for persistence in topology_data:
            if atom_i in persistence[1] and atom_j in persistence[1]:
                weight += persistence[0]
        return weight

    def forces(self) -> np.ndarray:
        """
        describes forces/force field for entire molecule
        """
        if not hasattr(self, "_molecule"):
            self.unpack_roc()
        if not hasattr(self, "gto"):
            self._build_gto()
        assert isinstance(self.smile, str) and isinstance(
            self._molecule, CombinatorialComplex
        )
        assert isinstance(self.gto, gto.Mole)
        self._computed_features = defaultdict(list)
        mf = dft.RKS(self.gto).density_fit()
        mf.xc = "b3lyp"
        mf.kernel(nproc=4)
        grad_calc = grad.RKS(mf)
        forces = grad_calc.kernel()
        forces = np.asarray(forces)
        self._computed_features["forces"] = forces
        topology_data = self._molecule_feat["persistence"]
        self._computed_features["molecule_persistence"] = topology_data
        return forces

    def positions(self) -> np.ndarray:
        """
        return positions matrix
        """
        if not hasattr(self, "gto"):
            self._build_gto()
        if not hasattr(self, "_computed_features"):
            self.forces()
        assert isinstance(self.gto, gto.Mole)
        positions = np.array([atom[1:] for atom in self.gto.atom_coords()])
        self._computed_features["positions"] = positions
        return positions

    def dist_matrix(self) -> np.ndarray:
        """
        return distance matrix
        """
        if not hasattr(self, "gto"):
            self._build_gto()
        if not hasattr(self, "_computed_features"):
            self.forces()
        assert isinstance(self.gto, gto.Mole)
        positions = self.positions()
        dist_matrix = distance_matrix(positions, positions)
        self._computed_features["dist_matrix"] = dist_matrix
        return dist_matrix

    def symbols(self) -> List:
        """
        return symbols list
        """
        if not hasattr(self, "gto"):
            self._build_gto()
        if not hasattr(self, "_computed_features"):
            self.forces()
        assert isinstance(self.gto, gto.Mole)
        symbols = [atom[0] for atom in self.gto.atom]
        self._computed_features["symbols"] = symbols
        return symbols

    def get_electrostatics(self) -> np.ndarray:
        """
        getter method for electrostatic potential
        """
        return self.electrostatics()

    def get_forces(self) -> np.ndarray:
        """
        getter method for forces matrix
        """
        return self.forces()

    def get_positions(self) -> np.ndarray:
        """
        getter method for positions matrix
        """
        return self.positions()

    def get_dist_matrix(self) -> np.ndarray:
        """
        getter method for distance matrix
        """
        return self.dist_matrix()

    def get_symbols(self) -> List:
        """
        getter method for symbols list
        """
        return self.symbols()
