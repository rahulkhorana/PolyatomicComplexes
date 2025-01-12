import os
import sys
import dill
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# atoms
from ase import Atoms
import periodictable

# pyscf
from pyscf.gto import Mole
from pyscf import gto, dft

# rdkit
from rdkit import Chem
from rdkit.Chem.Descriptors import NumRadicalElectrons

# scipy
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform

# GPAW
from gpaw import GPAW
from gpaw.cdft.cdft import CDFT
from gpaw.utilities.dos import print_projectors

# topology
from toponetx import CombinatorialComplex
from polyatomic_complexes.src.complexes.quantum_complex import QuantumComplex


class QuantumWavesComplex(QuantumComplex):
    def __init__(self, smile, target_dimension, atoms, bonds):
        super().__init__(smile, target_dimension, atoms, bonds)
        self.smile = smile
        self.dim = target_dimension
        self.atoms = atoms
        self.bnds = bonds
        self.k_b = 1.380649e-23
        self.temperature = 298.15

    def compute_realistic_constraints(self, atoms, molecule):
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
            # rdkit_atom = molecule.GetAtomWithIdx(atom_idx)
            num_radical_electrons = NumRadicalElectrons(molecule)
            if num_radical_electrons > 0:
                constraints["spin_regions"].append([atom_idx])
                constraints["spins"].append(num_radical_electrons)
        return constraints

    def compute_total_dos(self, calc):
        try:
            energies, dos = calc.get_dos(spin=0, npts=2001, width=0.1)
            return energies, dos
        except Exception as e:
            print(f"Error computing total DOS: {e}")
            return None, None

    def compute_atomic_orbital_pdos(self, calc, atom_index):
        try:
            atom_symbol = calc.atoms[atom_index].symbol
            orbital_types = []
            projectors = print_projectors(atom_symbol)
            for line in projectors.splitlines():
                if "l=" in line:
                    orbital_types.append(line.split()[-1])
            pdos = {}
            for orbital in orbital_types:
                energies, density = calc.get_orbital_ldos(
                    a=atom_index, angular=orbital, npts=2001, width=0.1
                )
                pdos[orbital] = {"energies": energies, "pdos": density}
            return pdos
        except Exception as e:
            print(f"Error computing PDOS for atom {atom_index}: {e}")
            return None

    def apply_constrained_dft(self, atoms, calc, constraints):
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

    def compute_long_range_interactions(self):
        if not hasattr(self, "computed_props"):
            self.compute_quantum_properties()
        mol = gto.Mole()
        mol.atom = gto.M(atom=self.smile)
        mol.basis = "aug-cc-pVDZ"
        mol.build()
        assert isinstance(mol, Mole)
        mf = dft.RKS(mol)
        mf.xc = "B3LYP-D3"
        total_energy = mf.kernel()
        dispersion_energy = (
            mf.with_dispersion if hasattr(mf, "with_dispersion") else None
        )
        dipole_moment = mol.dip_moment()
        quadrupole_moment = mol.quad_moment()
        atom_positions = mol.atom_coords()
        interatomic_distances = pdist(atom_positions)
        radius_of_gyration = np.sqrt(np.mean(np.sum(atom_positions**2, axis=1)))
        thermal_energy = self.k_b * self.temperature
        free_energy = total_energy - self.temperature * thermal_energy

        self.computed_props["total_energy"] = total_energy
        self.computed_props["dispersion_energy"] = dispersion_energy
        self.computed_props["dipole_moment"] = dipole_moment
        self.computed_props["quadrupole_moment"] = quadrupole_moment
        self.computed_props["radius_of_gyration"] = radius_of_gyration
        self.computed_props["interatomic_distances"] = squareform(
            interatomic_distances
        ).tolist()
        self.computed_props["thermal_energy"] = thermal_energy
        self.computed_props["free_energy"] = free_energy
        return

    def compute_quantum_properties(self):
        properties = defaultdict(dict)
        atoms = super()._generate_atoms_from_smile(self.smile)
        molecule = Chem.MolFromSmiles(self.smile)
        Chem.AddHs(molecule)
        assert isinstance(atoms, Atoms)
        calc = GPAW(
            xc="SCAN",
            mode="lcao",
            basis="aug-dzp",
            convergence={"density": 1e-6},
        )
        constraints = self.compute_realistic_constraints(atoms, molecule)
        cdft = self.apply_constrained_dft(atoms, calc, constraints)
        atoms.calc = cdft
        atoms.get_potential_energy()
        energies, dos = self.compute_total_dos(calc)
        if energies and dos:
            properties["total_dos"] = {"energies": energies, "dos": dos}
        molecule_indices = list(range(len(atoms)))
        molecular_pdos = self.compute_molecular_orbital_pdos(calc, molecule_indices)
        if molecular_pdos:
            properties["molecular_pdos"] = molecular_pdos
        properties["atomic_pdos"] = {}
        for i, _ in enumerate(atoms):
            pdos = self.compute_atomic_orbital_pdos(calc, atom_index=i)
            if pdos:
                properties["atomic_pdos"][i] = pdos
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
        self.computed_props = properties
        return properties

    def electronic_wave_topology(self):
        """
        Compute the topological features of the wavefunctions.
        Returns:
            defaultdict: Dictionary of topological features of wavefunctions.
                - vertex_features: Average wavefunction amplitudes at vertices.
                - edge_features: Total wavefunction amplitudes along edges.
                - betti_numbers: Topological Betti numbers (connected components, holes, etc.).
                - localization: Localization metrics (max, min amplitudes, regions).
                - wavefunction_overlaps: Overlaps between wavefunctions in different regions.
        """
        wavefunctions = self.computed_props.get("wavefunctions")
        roc = self.roc
        assert isinstance(roc, CombinatorialComplex)
        if wavefunctions is None or roc is None:
            raise ValueError("Wavefunction data or ROC is missing.")
        topology_features = defaultdict(dict)
        positions = self.computed_props.get("refined_positions")
        if positions is None:
            raise ValueError("Refined positions are missing.")
        vertex_features = defaultdict(float)
        edge_features = defaultdict(float)
        for vertex in roc.cells(dim=0):
            indices = list(vertex)
            amplitudes = [wavefunctions[i] for i in indices]
            vertex_features[vertex] = np.mean(amplitudes)
        for edge in roc.cells(dim=1):
            indices = list(edge)
            amplitudes = [wavefunctions[i] for i in indices]
            edge_features[edge] = np.sum(amplitudes)
        topology_features["vertex_features"] = vertex_features
        topology_features["edge_features"] = edge_features
        max_amplitude = np.max(wavefunctions)
        min_amplitude = np.min(wavefunctions)
        localization = np.where(wavefunctions > 0.1 * max_amplitude)[0]
        topology_features["localization"] = defaultdict(dict)
        topology_features["localization"]["max_amplitude"] = max_amplitude
        topology_features["localization"]["min_amplitude"] = min_amplitude
        topology_features["localization"]["localized_regions"] = localization.tolist()
        overlaps = defaultdict(float)
        vertices = list(roc.cells(dim=0))
        for i, vertex1 in enumerate(vertices):
            for _, vertex2 in enumerate(vertices[i + 1 :], start=i + 1):
                indices1 = list(vertex1)
                indices2 = list(vertex2)
                wf1 = np.array([wavefunctions[i] for i in indices1])
                wf2 = np.array([wavefunctions[i] for i in indices2])
                overlap = np.dot(wf1, wf2)
                overlaps[f"({vertex1}, {vertex2})"] = overlap
        topology_features["wavefunction_overlaps"] = overlaps
        return topology_features

    def _get_props(self, column_name):
        if not hasattr(self, "computed_props"):
            self.compute_quantum_properties()
        all_possible_columns = set(
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
                "molecular_pdos",
                "atomic_pdos",
                "total_dos",
                "total_energy",
                "dispersion_energy",
                "quadrupole_moment",
                "radius_of_gyration",
                "interatomic_distances",
                "thermal_energy",
                "free_energy",
            ]
        )
        if column_name in all_possible_columns:
            return self.computed_props[column_name]
        else:
            raise Exception("invalid column name")

    def visualize_property(atoms, property_values, title="Molecular Properties"):
        """
        Visualize molecular properties in 3D space.
        """
        positions = atoms.get_positions()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        scatter = ax.scatter(x, y, z, c=property_values, cmap="viridis")
        plt.colorbar(scatter, ax=ax, label="Property Value")
        ax.set_title(title)
        plt.show()

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

    def get_molecular_orbitals(self):
        return self._get_props("molecular_pdos")

    def get_atomic_pdos(self):
        return self._get_props("atomic_pdos")

    def get_total_dos(self):
        return self._get_props("total_dos")

    def get_total_energy(self):
        return self._get_props("total_energy")

    def get_dispersion_energy(self):
        return self._get_props("dispersion_energy")

    def get_quadrupole_moment(self):
        return self._get_props("quadrupole_moment")

    def get_radius_of_gyration(self):
        return self._get_props("radius_of_gyration")

    def get_interatomic_distances(self):
        return self._get_props("interatomic_distances")

    def get_thermal_energy(self):
        return self._get_props("thermal_energy")

    def get_free_energy(self):
        return self._get_props("free_energy")
