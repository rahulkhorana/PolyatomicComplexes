import os
import sys
import numpy as np
from typing import List
from pathlib import Path
from collections import defaultdict
from scipy.spatial import distance_matrix

# src
from polyatomic_complexes.src.complexes.abstract_complex import AbstractComplex

# rdkit
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

# ase and gpaw
from ase import Atoms
from gpaw import GPAW
from gpaw.poisson import PoissonSolver

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
        self.set_gpaw_path()

    def set_gpaw_path(self) -> None:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent
        setup_path = project_root / "gpaw_files"
        if not setup_path.is_dir():
            raise FileNotFoundError(f"The setup path '{setup_path}' does not exist.")
        os.environ["GPAW_SETUP_PATH"] = str(setup_path)
        print(f"GPAW_SETUP_PATH set to: {setup_path}")
        return

    def unpack_roc(self):
        self._molecule, self._molecule_feat = self.roc["molecule"]
        self._nucleus, self._nucleus_feat = self.roc["nuclear_structure"]
        self._electrons, self._electron_feat = self.roc["electronic_structure"]
        return

    def electrostatics(self):
        """
        returns the electrostatic potential
        """
        if not hasattr(self, "_computed_features"):
            self.forces()
        positions, dist_matrix, top_data = (
            self._computed_features["positions"],
            self._computed_features["dist_matrix"],
            self._computed_features["molecule_persistence"],
        )
        assert isinstance(positions, np.ndarray)
        num_atoms = positions.shape[0]
        electrostatic_potential = np.zeros(num_atoms)
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j:
                    distance = dist_matrix[i, j]
                    topo_weight = self.compute_top_weight(i, j, top_data)
                    if topo_weight != 0:
                        electrostatic_potential[i] += topo_weight / distance
                    else:
                        electrostatic_potential[i] += 1 / distance
        return electrostatic_potential

    def compute_top_weight(self, atom_i, atom_j, topology_data):
        weight = 1.0
        for persistence in topology_data:
            if atom_i in persistence[1] and atom_j in persistence[1]:
                weight += persistence[0]
        return weight

    def forces(self):
        """
        describes forces/force field for entire molecule
        """
        if not hasattr(self, "_molecule"):
            self.unpack_roc()
        assert isinstance(self.smile, str) and isinstance(
            self._molecule, CombinatorialComplex
        )
        self._computed_features = defaultdict(list)
        molecule = Chem.MolFromSmiles(self.smile)
        molecule = Chem.AddHs(molecule)
        AllChem.EmbedMolecule(molecule)
        AllChem.UFFOptimizeMolecule(molecule)
        conformer = molecule.GetConformer()
        positions = np.array(
            [list(conformer.GetAtomPosition(i)) for i in range(molecule.GetNumAtoms())]
        )
        symbols = [atom.GetSymbol() for atom in molecule.GetAtoms()]
        atoms = Atoms(symbols=symbols, positions=positions, pbc=False)
        atoms.center(vacuum=5.0)
        calc = GPAW(
            mode="lcao", basis="dzp", xc="PBE", poissonsolver=PoissonSolver(eps=1e-12)
        )
        atoms.calc = calc
        forces = atoms.get_forces()
        dist_matrix = distance_matrix(positions, positions)
        topology_data = self._molecule_feat["persistence"]
        self._computed_features["molecule_persistence"] = topology_data
        self._computed_features["positions"] = positions
        self._computed_features["dist_matrix"] = dist_matrix
        self._computed_features["forces"] = forces
        self._computed_features["symbols"] = symbols
        return forces

    def get_electrostatics(self):
        """
        getter method for electrostatic potential
        """
        return self.electrostatics()

    def get_forces(self):
        """
        getter method for forces matrix
        """
        return self.forces()
