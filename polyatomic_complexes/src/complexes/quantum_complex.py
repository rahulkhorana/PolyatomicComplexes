import os
import sys
import dill
import json
import numpy as np
import periodictable
from ase import Atoms

from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem import AllChem

import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial import distance_matrix
from rdkit.Chem.Descriptors import NumRadicalElectrons
from pathlib import Path

from typing import List, Tuple, Dict, Any, Optional
from scipy.fft import fftn, ifftn, fftfreq

# pyscf
from pyscf import gto, dft, grad
from pyscf.geomopt import geometric_solver
from pyscf.hessian import rks as rks_hessian
from pyscf.dft import numint
from pyscf.hessian import thermo
import inspect


BASE_PATH = Path(__file__)
project_root = BASE_PATH.parent.parent.parent.parent.resolve()
src_dir = project_root
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

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
        self.lookup_fp = BASE_PATH.parent.parent.parent.__str__() + "/dataset/construct"
        with open(self.lookup_fp + "/basis_sets.json", "rb") as f:
            self.basis_sets = json.load(f)
        self.gto = self._build_gto()
        self.cm_to_au = 4.556335252767e-06
        self.T = 298.15
        self.k_B = 3.166811563e-06

    def unpack_roc(self):
        self._molecule, self._molecule_feat = self.roc["molecule"]
        self._nucleus, self._nucleus_feat = self.roc["nuclear_structure"]
        self._electrons, self._electron_feat = self.roc["electronic_structure"]
        return

    def _build_gto(self) -> gto.Mole:
        mol = Chem.MolFromSmiles(self.smile)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {self.smile}")
        mol = Chem.AddHs(mol)
        embed_result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if embed_result != 0:
            raise ValueError(
                "RDKit failed to embed the molecule. Please check the SMILES string."
            )
        atoms = mol.GetAtoms()
        coordinates = mol.GetConformers()[0].GetPositions()
        symbols = [atom.GetSymbol() for atom in atoms]
        mole = gto.M(
            atom=[[symbols[i], *coordinates[i]] for i in range(len(atoms))],
            basis=self.basis_sets,
            unit="Ang",
            verbose=0,
        )
        if not isinstance(mole, gto.Mole):
            raise TypeError(
                f"Expected 'mole' to be a pyscf.gto.Mole instance, got {type(mole)}"
            )
        self.gto = mole
        return mole

    def _vib_thermal_correction(self, freq: float) -> float:
        """
        Computes the vibrational thermal correction for a given frequency.

        Parameters:
            freq (float): Vibrational frequency in atomic units (Hartree).

        Returns:
            float: Thermal correction energy in Hartree.
        """
        if isinstance(freq, (np.complex128, complex)):
            if abs(freq.imag) < 1e-10:
                freq = freq.real
            else:
                return 0.0
        if not (isinstance(freq, float) or isinstance(freq, int)):
            raise TypeError(f"Expected 'freq' to be a number, got {type(freq)}")
        if freq < 1e-12:
            return 0.0
        x = freq / (self.k_B * self.T)
        return freq / (np.exp(x) - 1.0)

    def _nuclear_potential(self, mol_obj: gto.Mole, grid: np.ndarray) -> np.ndarray:
        """
        Computes the nuclear potential at each point in the grid.
        Parameters:
            mol_obj (pyscf.gto.Mole): PySCF Mole object.
            grid (np.ndarray): Grid points as an (N, 3) array.
        Returns:
            np.ndarray: Nuclear potential at each grid point.
        """
        if not isinstance(mol_obj, gto.Mole):
            raise TypeError(
                f"Expected 'mol_obj' to be a pyscf.gto.Mole instance, got {type(mol_obj)}"
            )
        if not isinstance(grid, np.ndarray):
            raise TypeError(f"Expected 'grid' to be a numpy.ndarray, got {type(grid)}")
        if grid.ndim != 2 or grid.shape[1] != 3:
            raise ValueError(f"Expected 'grid' to be of shape (N, 3), got {grid.shape}")
        v_nuc = np.zeros(len(grid))
        for ia in range(mol_obj.natm):
            Z = mol_obj.atom_charge(ia)
            Ra = mol_obj.atom_coord(ia)
            diff = grid - Ra
            r = np.linalg.norm(diff, axis=1)
            r = np.where(r < 1e-6, 1e-6, r)
            v_nuc += Z / r
        return v_nuc

    def _compute_quantum_properties(self) -> defaultdict:
        """
        Computes quantum-level properties at a DFT (B3LYP) level of theory,
        including geometry optimization, thermal corrections via a frequency
        calculation (harmonic approximation), and advanced properties like
        effective potential and electrostatic potentials.

        Returns:
            defaultdict: A dictionary with property names as keys and computed
                         values or None as values.
        """
        self.computed_props = defaultdict(list)
        mol = self.gto
        if not isinstance(mol, gto.Mole):
            raise TypeError(
                f"Expected 'mol' to be a pyscf.gto.Mole instance, got {type(mol)}"
            )
        mf = dft.RKS(mol)
        mf.xc = "b3lyp"
        mf.verbose = 4
        mf_optimized = geometric_solver.optimize(mf)
        mol_opt = mf_optimized
        if not isinstance(mol_opt, gto.Mole):
            raise TypeError(
                f"Expected 'mol_opt' to be a pyscf.gto.Mole instance, got {type(mol_opt)}"
            )
        self.mf_final = dft.RKS(mol_opt)
        self.mf_final.xc = "b3lyp"
        self.mf_final.kernel()
        mf_final = self.mf_final
        mo_energies = mf_final.mo_energy
        mo_coeff = mf_final.mo_coeff
        mo_occ = mf_final.mo_occ
        total_energy = mf_final.e_tot
        if not isinstance(mo_coeff, np.ndarray):
            raise TypeError(
                f"Expected mo_coeff to be a numpy.ndarray, got {type(mo_coeff)}"
            )
        if mo_coeff.ndim != 2:
            raise ValueError(
                f"Expected mo_coeff to be a 2D array, got {mo_coeff.ndim}D"
            )
        if not isinstance(mo_occ, np.ndarray):
            raise TypeError(
                f"Expected mo_occ to be a numpy.ndarray, got {type(mo_occ)}"
            )
        if mo_occ.ndim != 1:
            raise ValueError(f"Expected mo_occ to be a 1D array, got {mo_occ.ndim}D")
        if not (
            isinstance(total_energy, float) or isinstance(total_energy, np.float32)
        ):
            raise TypeError(
                f"Expected total_energy to be a float, got {type(total_energy)}"
            )
        hessian_obj = rks_hessian.Hessian(mf_final)
        hess_mat = hessian_obj.kernel()
        mass = mf_final.mol.atom_mass_list()
        print(
            f"thermo harmonic analysis signature: {inspect.signature(thermo.harmonic_analysis)}"
        )
        freq_analysis = thermo.harmonic_analysis(
            mol=mf_final.mol,
            hess=hess_mat,
            exclude_trans=True,
            exclude_rot=True,
            imaginary_freq=True,
            mass=mass,
        )
        freqs_cm = freq_analysis["freq_wavenumber"]
        freqs_au = np.array(freqs_cm) * self.cm_to_au
        if not isinstance(freqs_cm, np.ndarray):
            freqs_cm = np.array(freqs_cm)
        if freqs_cm.ndim != 1:
            raise ValueError(
                f"Expected 'freqs_cm' to be a 1D array, got shape {freqs_cm.shape}"
            )
        positive_freqs_au = freqs_au[freqs_au > 0.0]
        zpe_hartree = 0.5 * positive_freqs_au.sum()
        vib_thermal = sum(self._vib_thermal_correction(w) for w in positive_freqs_au)
        E_trans_rot = 4.5 * self.k_B * self.T
        thermal_corr_internal_energy = zpe_hartree + vib_thermal + E_trans_rot
        E0 = total_energy + zpe_hartree
        refined_positions = mol_opt.atom_coords()
        if not isinstance(refined_positions, np.ndarray):
            refined_positions = np.array(refined_positions)
        if refined_positions.ndim != 2 or refined_positions.shape[1] != 3:
            raise ValueError(
                f"Expected 'refined_positions' to be of shape (N, 3), got {refined_positions.shape}"
            )
        num_atoms = refined_positions.shape[0]
        dist_matrix = np.zeros((num_atoms, num_atoms))
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                dist = np.linalg.norm(refined_positions[i] - refined_positions[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        grad_calculator = grad.RKS(mf_final)
        forces = -grad_calculator.kernel()
        if not isinstance(forces, np.ndarray):
            forces = np.array(forces)
        if forces.shape != refined_positions.shape:
            raise ValueError(
                f"Expected 'forces' to have shape {refined_positions.shape}, got {forces.shape}"
            )
        occupied_energies = mo_energies[mo_occ > 0]
        virtual_energies = mo_energies[mo_occ == 0]
        homo: Optional[float] = (
            max(occupied_energies) if len(occupied_energies) else None
        )
        lumo: Optional[float] = min(virtual_energies) if len(virtual_energies) else None
        if homo is not None and lumo is not None:
            fermi_level = 0.5 * (homo + lumo)
            homo_lumo_gap = lumo - homo
        else:
            fermi_level = None
            homo_lumo_gap = None
        dip_moment_components = mf_final.dip_moment()
        if not isinstance(dip_moment_components, np.ndarray):
            dip_moment_components = np.array(dip_moment_components)
        print(f"dip moment com: {dip_moment_components}")
        dipole_vector = dip_moment_components[:3]
        dipole_magnitude = dip_moment_components[2]
        dm = mf_final.make_rdm1()
        veff = mf_final.get_veff(mol_opt, dm)
        coords = refined_positions
        pad = 3.0
        min_xyz = coords.min(axis=0) - pad
        max_xyz = coords.max(axis=0) + pad
        spacing = 0.2
        xs = np.arange(min_xyz[0], max_xyz[0] + spacing, spacing)
        ys = np.arange(min_xyz[1], max_xyz[1] + spacing, spacing)
        zs = np.arange(min_xyz[2], max_xyz[2] + spacing, spacing)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        if not isinstance(grid_points, np.ndarray):
            grid_points = np.array(grid_points)
        if grid_points.ndim != 2 or grid_points.shape[1] != 3:
            raise ValueError(
                f"Expected 'grid_points' to be of shape (N, 3), got {grid_points.shape}"
            )
        v_nuc = self._nuclear_potential(mol_opt, grid_points)
        if not isinstance(v_nuc, np.ndarray):
            v_nuc = np.array(v_nuc)
        if v_nuc.shape[0] != grid_points.shape[0]:
            raise ValueError(
                f"Expected 'v_nuc' to have shape ({grid_points.shape[0]},), got {v_nuc.shape}"
            )
        ni = numint.NumInt()
        ao = ni.eval_ao(mol_opt, grid_points)
        electron_density_map = ni.eval_rho(mol_opt, ao, dm)
        if not isinstance(electron_density_map, np.ndarray):
            electron_density_map = np.array(electron_density_map)
        if electron_density_map.shape[0] != grid_points.shape[0]:
            raise ValueError(
                f"Expected 'electron_density_map' to have shape ({grid_points.shape[0]},), got {electron_density_map.shape}"
            )
        grid_shape = X.shape
        grid_spacing = spacing
        grid_volume = grid_spacing**3
        rho_k = fftn(electron_density_map)
        rho_k = rho_k.reshape(grid_shape)
        kx = fftfreq(grid_shape[0], d=grid_spacing) * 2 * np.pi
        ky = fftfreq(grid_shape[1], d=grid_spacing) * 2 * np.pi
        kz = fftfreq(grid_shape[2], d=grid_spacing) * 2 * np.pi
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
        K_sq = KX**2 + KY**2 + KZ**2
        K_sq[0, 0, 0] = 1.0
        v_coul_k = 4 * np.pi * rho_k / K_sq
        v_coul = np.real(ifftn(v_coul_k)) * grid_volume
        v_nuc = v_nuc.reshape(grid_shape)
        v_total = v_nuc + v_coul
        assert isinstance(v_total, np.ndarray)
        v_total = v_total.reshape(grid_shape)
        esp_data: Dict[str, Any] = {
            "grid_coords": grid_points.tolist(),
            "total_electrostatic_potential": v_total.tolist(),
        }
        potential_energy = total_energy
        self.computed_props["forces"] = forces.tolist()
        self.computed_props["refined_positions"] = refined_positions.tolist()
        self.computed_props["dist_matrix"] = dist_matrix.tolist()
        self.computed_props["fermi_level"] = fermi_level
        self.computed_props["eigenvalues"] = mo_energies.tolist()
        self.computed_props["homo_lumo_gap"] = homo_lumo_gap
        self.computed_props["dipole_moment"] = {
            "vector": dipole_vector.tolist(),
            "magnitude": dipole_magnitude,
        }
        self.computed_props["effective_potential"] = veff
        self.computed_props["electrostatic_potentials"] = esp_data
        self.computed_props["wavefunctions"] = mo_coeff.tolist()
        self.computed_props["potential_energy"] = potential_energy
        self.computed_props["zpe_hartree"] = zpe_hartree
        self.computed_props["E0_elec_plus_zpe"] = E0
        self.computed_props["freqs_cm^-1"] = freqs_cm.tolist()
        self.computed_props["thermal_corr_internal_energy"] = (
            thermal_corr_internal_energy
        )
        return self.computed_props

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
                "wavefunctions",
                "potential_energy",
                "zpe_hartree",
                "E0_elec_plus_zpe",
                "freqs_cm^-1",
                "thermal_corr_internal_energy",
            ]
        )
        if column_name in all_columns:
            return self.computed_props[column_name]
        else:
            raise Exception("invalid column name")

    def forces(self):
        return self._get_props("forces")

    def positions(self):
        return self._get_props("refined_positions")

    def distance_matrix(self):
        return self._get_props("dist_matrix")

    def fermi_level(self):
        return self._get_props("fermi_level")

    def eigenvalues(self):
        return self._get_props("eigenvalues")

    def homo_lumo_gap(self):
        return self._get_props("homo_lumo_gap")

    def dipole_moment(self):
        return self._get_props("dipole_moment")

    def effective_potential(self):
        return self._get_props("effective_potential")

    def electrostatic_potentials(self):
        return self._get_props("electrostatic_potentials")

    def wavefunctions(self):
        return self._get_props("wavefunctions")

    def potential_energy(self):
        return self._get_props("potential_energy")

    def zpe_hartree(self):
        return self._get_props("zpe_hartree")

    def E0_elec_plus_zpe(self):
        return self._get_props("E0_elec_plus_zpe")

    def freqs_cm(self):
        return self._get_props("freqs_cm^-1")

    def thermal_corr_internal_energy(self):
        return self._get_props("thermal_corr_internal_energy")
