import os
import sys
import dill
import json
import numpy as np
import periodictable
from ase import Atoms
from gpaw import GPAW
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem import AllChem
from gpaw.cdft.cdft import CDFT
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial import distance_matrix
from rdkit.Chem.Descriptors import NumRadicalElectrons


from polyatomic_complexes.src.complexes.abstract_complex import AbstractComplex


class QuantumComplex(AbstractComplex):
    def __init__(self, smile, target_dimension, atoms, bonds):
        super().__init__(smile, target_dimension, atoms, bonds)
        self.smile = smile
        self.dim = target_dimension
        self.atoms = atoms
        self.bnds = bonds
        self.roc = self.rank_order_complex()
        self.figure_path = f"../../results/electron_density_viz_{smile}.png"

    def unpack_roc(self):
        self._molecule, self._molecule_feat = self.roc["molecule"]
        self._nucleus, self._nucleus_feat = self.roc["nuclear_structure"]
        self._electrons, self._electron_feat = self.roc["electronic_structure"]
        return

    def _generate_atoms_from_smile(self) -> Atoms:
        """
        Generate ASE Atoms object from a SMILES string.
        Args:
            smile (str): SMILES string of the molecule.
        Returns:
            Atoms: ASE Atoms object with 3D coordinates.
        """
        molecule = Chem.MolFromSmiles(self.smile)
        if molecule is None:
            raise ValueError("Invalid SMILES string")
        Chem.AddHs(molecule)
        AllChem.EmbedMolecule(molecule)
        AllChem.UFFOptimizeMolecule(molecule)
        conformer = molecule.GetConformer()
        symbols = [atom.GetSymbol() for atom in molecule.GetAtoms()]
        positions = np.array(
            [list(conformer.GetAtomPosition(i)) for i in range(molecule.GetNumAtoms())]
        )
        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.center(vacuum=5.0)
        return atoms

    def _compute_realistic_constraints(self, atoms: Atoms, molecule: Mol):
        """
        Compute realistic constraints (charges and spins) for constrained DFT.
        Args:
            atoms (Atoms): ASE Atoms object.
            molecule (rdkit.Chem.Mol): RDKit molecule object.
        Returns:
            defaultdict: A defaultdict containing charge and spin constraints.
        """
        assert isinstance(atoms, Atoms) and isinstance(molecule, Mol)
        default_charges = {
            atom.symbol: getattr(
                periodictable.elements.__getattr__(atom.symbol),
                "electronegativity_pauling",
                0.0,
            )
            for atom in atoms
        }
        constraints = defaultdict(list)
        for i, atom in enumerate(atoms):
            symbol = atom.symbol
            charge = (
                default_charges.get(symbol, 0.0) - 0.5
                if symbol in ["O", "N"]
                else default_charges.get(symbol, 0.0)
            )
            constraints["charge_regions"].append([i])
            constraints["charges"].append(charge)
        for atom_idx in range(molecule.GetNumAtoms()):
            num_radical_electrons = NumRadicalElectrons(molecule)
            if num_radical_electrons > 0:
                constraints["spin_regions"].append([atom_idx])
                constraints["spins"].append(num_radical_electrons)
        return constraints

    def _get_constrained_dft(
        self, atoms: Atoms, calc: GPAW, constraints: defaultdict
    ) -> CDFT:
        """
        Apply Constrained DFT (CDFT) using GPAW to the given Atoms object.
        Args:
            atoms (Atoms): ASE Atoms object.
            calc (GPAW): GPAW calculator instance.
            constraints (defaultdict): Constraints containing charge and spin regions.
        Returns:
            CDFT: Initialized CDFT object with applied constraints.
        """
        assert (
            isinstance(atoms, Atoms)
            and isinstance(calc, GPAW)
            and isinstance(constraints, defaultdict)
        )
        cdft = CDFT(
            calc=calc,
            atoms=atoms,
            charge_regions=constraints["charge_regions"],
            charges=constraints["charges"],
            spin_regions=constraints["spin_regions"],
            spins=constraints["spins"],
            method="CG",
            minimizer_options={"gtol": 0.01},
        )
        return cdft

    def _compute_quantum_properties(self):
        """
        Compute quantum properties for a molecule using Constrained DFT.
        Args:
            smile (str): SMILES string of the molecule.
        Returns:
            defaultdict: A defaultdict containing computed properties.
        """
        properties = defaultdict(dict)
        atoms = self._generate_atoms_from_smile(self.smile)
        molecule = Chem.MolFromSmiles(self.smile)
        Chem.AddHs(molecule)
        calc = GPAW(
            xc="SCAN",
            mode="lcao",
            basis="dzp",
            convergence={"density": 1e-6},
        )
        constraints = self._compute_realistic_constraints(atoms, molecule)
        cdft = self._get_constrained_dft(atoms, calc, constraints)
        atoms.calc = cdft
        assert (
            isinstance(calc, GPAW)
            and isinstance(cdft, CDFT)
            and isinstance(atoms, Atoms)
        )
        num_bands = calc.get_number_of_bands()
        num_spins = calc.get_number_of_spins()
        wavefunctions = []
        for band in range(num_bands):
            for spin in range(num_spins):
                wf = calc.get_pseudo_wave_function(band=band, spin=spin)
                wavefunctions.append({"band": band, "spin": spin, "wavefunction": wf})

        properties["potential_energy"] = atoms.get_potential_energy()
        properties["forces"] = atoms.get_forces()
        properties["refined_positions"] = atoms.get_positions()
        properties["dist_matrix"] = distance_matrix(
            properties["refined_positions"], properties["refined_positions"]
        )
        properties["fermi_level"] = calc.get_fermi_level()
        properties["eigenvalues"] = calc.get_eigenvalues()
        properties["homo_lumo_gap"] = calc.get_homo_lumo()
        properties["dipole_moment"] = calc.get_dipole_moment()
        properties["effective_potential"] = calc.get_effective_potential()
        properties["electrostatic_potentials"] = calc.get_electrostatic_potential()
        properties["orbital_magnetic_moments"] = calc.get_magnetic_moments()
        properties["wavefunctions"] = wavefunctions
        return properties

    def visualize_property(atoms, property_values, title="Molecular Properties"):
        """
        Visualize molecular properties in 3D space.
        Args:
            atoms (Atoms): ASE Atoms object.
            property_values (list): Values of the property to visualize for each atom.
            title (str): Title of the plot.
        """
        positions = atoms.get_positions()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        scatter = ax.scatter(x, y, z, c=property_values, cmap="viridis")
        plt.colorbar(scatter, ax=ax, label="Property Value")
        ax.set_title(title)
        plt.show()

    def set_features(self):
        self.computed_props = self._compute_quantum_properties()
        return

    def _get_props(self, column_name):
        if not hasattr(self, "computed_props"):
            self.set_features()
        all_columns = set(
            [
                "forces",
                "refined_positions",
                "dist_matrix",
                "fermi_level",
                "eigenvalues",
                "homo_lumo_gap",
                "dipole_moment",
                "effective_potential",
                "electrostatic_potentials",
                "orbital_magnetic_moments",
                "wavefunctions",
                "potential_energy",
            ]
        )
        if column_name in all_columns:
            return self.computed_props[column_name]
        else:
            raise Exception("invalid column name")

    def forces(self):
        return self._get_props("forces")

    def electrostatics(self):
        return self._get_props("electrostatic_potentials")

    def distances(self):
        return self._get_props("dist_matrix")

    def positions(self):
        return self._get_props("refined_positions")

    def homo_lumo_gap(self):
        return self._get_props("homo_lumo_gap")

    def dipole_moment(self):
        return self._get_props("dipole_moment")

    def effective_potential(self):
        return self._get_props("effective_potential")

    def orbital_magnetic_moments(self):
        return self._get_props("orbital_magnetic_moments")

    def wavefunctions(self):
        return self._get_props("wavefunctions")

    def get_forces(self):
        return self._get_props("forces")

    def get_electrostatics(self):
        return self._get_props("electrostatic_potentials")

    def get_distances(self):
        return self._get_props("dist_matrix")

    def get_positions(self):
        return self._get_props("refined_positions")

    def get_homo_lumo_gap(self):
        return self._get_props("homo_lumo_gap")

    def get_dipole_moment(self):
        return self._get_props("dipole_moment")

    def get_effective_potential(self):
        return self._get_props("effective_potential")

    def get_orbital_magnetic_moments(self):
        return self._get_props("orbital_magnetic_moments")

    def get_wavefunctions(self):
        return self._get_props("wavefunctions")

    def get_potential_energy(self):
        return self._get_props("potential_energy")
