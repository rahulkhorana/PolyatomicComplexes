import os
import sys
import dill
import json
import numpy as np
import periodictable
import traceback
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

# PySCF
from pyscf.dft import numint
from pyscf import gto, dft, grad
from dftd3.pyscf import DFTD3Dispersion
from pyscf.geomopt import geometric_solver
from pyscf.hessian import rks as rks_hessian
from pyscf.hessian import thermo

BASE_PATH = Path(__file__)
project_root = BASE_PATH.parent.parent.parent.parent.resolve()
src_dir = project_root
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from toponetx import CombinatorialComplex
from polyatomic_complexes.src.complexes.quantum_complex import QuantumComplex


class QuantumWavesComplex(QuantumComplex):
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

    def _vib_thermal_correction(self, freq: float) -> float:
        """
        Computes the vibrational thermal correction for a given frequency.

        Parameters:
            freq (float): Vibrational frequency in atomic units (Hartree).

        Returns:
            float: Thermal correction energy in Hartree.
        """
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
        self.computed_props["effective_potential"] = (
            veff.tolist() if isinstance(veff, np.ndarray) else veff
        )
        self.computed_props["electrostatic_potentials"] = esp_data
        self.computed_props["wavefunctions"] = mo_coeff.tolist()
        self.computed_props["potential_energy"] = total_energy
        self.computed_props["zpe_hartree"] = zpe_hartree
        self.computed_props["E0_elec_plus_zpe"] = E0
        self.computed_props["freqs_cm^-1"] = freqs_cm.tolist()
        self.computed_props["thermal_corr_internal_energy"] = (
            thermal_corr_internal_energy
        )
        dispersion_energy = self._compute_dispersion_energy()
        self.computed_props["dispersion_energy"] = dispersion_energy
        return self.computed_props

    def compute_long_range_interactions(self):
        """
        Computes long-range interaction properties such as dispersion energy,
        quadrupole moment, radius of gyration, interatomic distances,
        thermal energy, and free energy.

        Stores the computed properties in self.computed_props.
        """
        if not hasattr(self, "computed_props"):
            self._compute_quantum_properties()
        total_energy = self.computed_props.get("potential_energy")
        dipole_moment = self.computed_props.get("dipole_moment")
        refined_positions = np.array(self.computed_props.get("refined_positions", []))
        interatomic_distances = np.array(self.computed_props.get("dist_matrix", []))
        thermal_corr_internal_energy = self.computed_props.get(
            "thermal_corr_internal_energy"
        )
        dispersion_energy = self.computed_props.get("dispersion_energy")
        if dispersion_energy is not None:
            dispersion_energy = dispersion_energy[0]
        quadrupole_moment = self._compute_quadrupole_moment()
        radius_of_gyration = self._compute_radius_of_gyration()
        free_energy = None
        if (
            total_energy is not None
            and thermal_corr_internal_energy is not None
            and dispersion_energy is not None
        ):
            free_energy = (
                total_energy + thermal_corr_internal_energy + dispersion_energy
            )
        self.computed_props["dispersion_energy"] = dispersion_energy
        self.computed_props["quadrupole_moment"] = quadrupole_moment
        self.computed_props["radius_of_gyration"] = radius_of_gyration
        if interatomic_distances.size > 0:
            from scipy.spatial.distance import squareform

            if interatomic_distances.shape[0] > interatomic_distances.shape[1]:
                interatomic_distances = squareform(interatomic_distances)
            self.computed_props["interatomic_distances"] = (
                interatomic_distances.tolist()
            )
        else:
            self.computed_props["interatomic_distances"] = []
        self.computed_props["thermal_energy"] = thermal_corr_internal_energy
        self.computed_props["free_energy"] = free_energy

    def _compute_dispersion_energy(self) -> Optional[Tuple]:
        """
        Computes the dispersion energy using DFT-D3 via the DFTD3Dispersion class.
        Returns:
            Optional[float]: Tuple of (dispersion energy in Hartree, d3 energy matrix), or None if computation fails.
        """
        try:
            if not hasattr(self, "mf_final"):
                raise AttributeError("mf_final not found")
            mol = self.mf_final.mol
            if mol is None:
                raise ValueError("Molecule object is None")
            d3_model = DFTD3Dispersion(mol)
            d3_model.scf = self.mf_final
            kernel = d3_model.kernel()
            assert len(kernel) == 2
            d3_energy = kernel[0]
            d3_energy_matrix = kernel[1]
            assert isinstance(d3_energy, np.ndarray)
            d3_energy = d3_energy.tolist()
            assert (
                isinstance(d3_energy, float)
                or isinstance(d3_energy, int)
                or isinstance(d3_energy, np.floating)
            )
            assert isinstance(d3_energy_matrix, np.ndarray)
            if not (isinstance(d3_energy, (int, float, np.floating))):
                raise TypeError(
                    f"Expected 'd3_energy' to be a number, got {type(d3_energy)}"
                )
            return float(d3_energy), d3_energy_matrix
        except Exception as e:
            print(f"Dispersion energy computation failed: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print(traceback.format_exc())
            return None

    def _compute_quadrupole_moment(self) -> Optional[List[float]]:
        """
        Computes the quadrupole moment of the molecule.

        Returns:
            Optional[List[float]]: Quadrupole moment components [Qxx, Qyy, Qzz, Qxy, Qxz, Qyz],
                                   or None if computation fails.
        """
        try:
            # Add verification steps
            if not hasattr(self, "mf_final"):
                raise AttributeError("mf_final not found")
            mol_opt = self.mf_final.mol
            if mol_opt is None:
                raise ValueError("Molecule object is None")
            dm = self.mf_final.make_rdm1()
            if "electrostatic_potentials" not in self.computed_props:
                raise KeyError("electrostatic_potentials not found in computed_props")
            grid_coords = np.array(
                self.computed_props["electrostatic_potentials"]["grid_coords"]
            )
            ni = numint.NumInt()
            ao = ni.eval_ao(mol_opt, grid_coords)
            electron_density = ni.eval_rho(mol_opt, ao, dm)
            Q = np.zeros((3, 3))
            r_sq = np.sum(grid_coords**2, axis=1)
            grid_volume = 0.2**3
            for i in range(3):
                for j in range(3):
                    Q[i, j] = (
                        np.sum(
                            (
                                3 * grid_coords[:, i] * grid_coords[:, j]
                                - r_sq * (i == j)
                            )
                            * electron_density
                        )
                        * grid_volume
                    )
            assert isinstance(mol_opt, gto.Mole)
            for ia in range(mol_opt.natm):
                Z = mol_opt.atom_charge(ia)
                R = mol_opt.atom_coord(ia)
                R_sq = np.dot(R, R)
                for i in range(3):
                    for j in range(3):
                        Q[i, j] -= Z * (3 * R[i] * R[j] - R_sq * (i == j))
            quad_moment_list = [
                Q[0, 0],
                Q[1, 1],
                Q[2, 2],
                Q[0, 1],
                Q[0, 2],
                Q[1, 2],
            ]
            return quad_moment_list
        except Exception as e:
            print(f"Quadrupole moment computation failed: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print(traceback.format_exc())
            return None

    def _compute_radius_of_gyration(self) -> Optional[float]:
        """
        Computes the mass-weighted radius of gyration of the molecule.

        Returns:
            Optional[float]: Radius of gyration in Angstroms, or None if computation fails.
        """
        try:
            if not hasattr(self, "mf_final"):
                raise AttributeError("mf_final not found")
            mol = self.mf_final.mol
            if mol is None:
                raise ValueError("Molecule object is None")
            coords = mol.atom_coords()
            if coords is None:
                raise ValueError("Could not get atomic coordinates")
            assert isinstance(mol, gto.Mole)
            natm = mol.natm
            masses = np.array(
                [
                    periodictable.elements.symbol(mol.atom_symbol(ia)).mass
                    for ia in range(natm)
                ]
            )
            total_mass = masses.sum()
            center_of_mass = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
            rg_sq = (
                np.sum(masses * np.sum((coords - center_of_mass) ** 2, axis=1))
                / total_mass
            )
            radius_of_gyration = np.sqrt(rg_sq)
            print(f"radius of gyration: {radius_of_gyration}")
            return radius_of_gyration
        except Exception as e:
            print(f"Radius of gyration computation failed: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print(traceback.format_exc())
            return None

    def _compute_wavefunction_overlaps(
        self, wavefunctions: np.ndarray
    ) -> Dict[str, float]:
        """
        Computes overlaps between different wavefunctions.

        Parameters:
            wavefunctions (np.ndarray): Array of wavefunction coefficients.

        Returns:
            Dict[str, float]: Dictionary mapping wavefunction pairs to their overlap values.
        """
        overlaps = {}
        num_wf = wavefunctions.shape[1]
        for i in range(num_wf):
            for j in range(i + 1, num_wf):
                overlap = np.dot(wavefunctions[:, i], wavefunctions[:, j])
                overlaps[f"WF_{i}-WF_{j}"] = float(overlap)
        return overlaps

    def visualize_property(
        self, property_name: str, title: str = "Molecular Property Visualization"
    ):
        """
        Visualizes a specified molecular property in 3D space.

        Parameters:
            property_name (str): Name of the property to visualize.
            title (str): Title of the visualization plot.
        """
        try:
            property_data = self._get_props(property_name)
        except KeyError as e:
            print(e)
            return
        if property_name == "refined_positions":
            positions = np.array(property_data)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                positions[:, 0], positions[:, 1], positions[:, 2], c="blue", marker="o"
            )
            ax.set_title(title)
            ax.set_xlabel("X (Angstrom)")
            ax.set_ylabel("Y (Angstrom)")
            ax.set_zlabel("Z (Angstrom)")
            plt.show()
            return
        elif property_name == "dipole_moment":
            dipole = property_data
            dipole_vector = np.array(dipole["vector"])
            dipole_magnitude = dipole["magnitude"]
            positions = np.array(self.computed_props.get("refined_positions", []))
            if positions.size == 0:
                print("Refined positions are missing. Cannot visualize dipole moment.")
                return
            center = positions.mean(axis=0)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.quiver(
                center[0],
                center[1],
                center[2],
                dipole_vector[0],
                dipole_vector[1],
                dipole_vector[2],
                length=1.0,
                normalize=True,
                color="red",
                linewidth=2,
            )
            ax.set_title(
                f"{title} - Dipole Moment Magnitude: {dipole_magnitude:.3f} Debye"
            )
            ax.set_xlabel("X (Angstrom)")
            ax.set_ylabel("Y (Angstrom)")
            ax.set_zlabel("Z (Angstrom)")
            plt.show()
            return
        elif property_name == "wavefunctions":
            wavefunctions = np.array(property_data)
            if wavefunctions.ndim != 2 or wavefunctions.shape[1] == 0:
                print("Wavefunction data is not in expected format.")
                return
            wf1 = wavefunctions[:, 0]
            positions = np.array(self.computed_props.get("refined_positions", []))
            if positions.size == 0:
                print("Refined positions are missing. Cannot visualize wavefunctions.")
                return
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            sc = ax.scatter(
                positions[:, 0], positions[:, 1], positions[:, 2], c=wf1, cmap="viridis"
            )
            plt.colorbar(sc, ax=ax, label="Wavefunction Amplitude")
            ax.set_title(f"{title} - Molecular Orbital 1")
            ax.set_xlabel("X (Angstrom)")
            ax.set_ylabel("Y (Angstrom)")
            ax.set_zlabel("Z (Angstrom)")
            plt.show()
            return
        else:
            print(f"Visualization for property '{property_name}' is not implemented.")
            return

    def set_features(self):
        """
        Computes and sets the quantum properties.
        """
        self.computed_props = self._compute_quantum_properties()
        return

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

    def _get_props(self, column_name: str) -> Any:
        """
        Retrieves a computed property by its column name.

        Parameters:
            column_name (str): The name of the property to retrieve.

        Returns:
            Any: The value of the requested property.

        Raises:
            KeyError: If the property does not exist.
        """
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
                "dispersion_energy",
                "quadrupole_moment",
                "radius_of_gyration",
                "free_energy",
                "interatomic_distances",
            ]
        )
        if column_name in all_columns:
            return self.computed_props[column_name]
        else:
            raise KeyError(
                f"Property '{column_name}' not found in computed properties."
            )

    def get_forces(self) -> Optional[np.ndarray]:
        """
        Retrieves the forces acting on each atom.

        Returns:
            Optional[np.ndarray]: Forces array of shape (N, 3), or None if not available.
        """
        forces = self._get_props("forces")
        return np.array(forces) if forces else None

    def get_positions(self) -> Optional[np.ndarray]:
        """
        Retrieves the optimized atomic positions.

        Returns:
            Optional[np.ndarray]: Positions array of shape (N, 3), or None if not available.
        """
        positions = self._get_props("refined_positions")
        return np.array(positions) if positions else None

    def get_distance_matrix(self) -> Optional[np.ndarray]:
        """
        Retrieves the interatomic distance matrix.

        Returns:
            Optional[np.ndarray]: Distance matrix of shape (N, N), or None if not available.
        """
        distances = self._get_props("dist_matrix")
        return np.array(distances) if distances else None

    def get_fermi_level(self) -> Optional[float]:
        """
        Retrieves the Fermi level.

        Returns:
            Optional[float]: Fermi level in Hartree, or None if not available.
        """
        return self._get_props("fermi_level")

    def get_eigenvalues(self) -> Optional[np.ndarray]:
        """
        Retrieves the molecular orbital eigenvalues.

        Returns:
            Optional[np.ndarray]: Eigenvalues array, or None if not available.
        """
        eigenvalues = self._get_props("eigenvalues")
        return np.array(eigenvalues) if eigenvalues else None

    def get_homo_lumo_gap(self) -> Optional[float]:
        """
        Retrieves the HOMO-LUMO gap.

        Returns:
            Optional[float]: HOMO-LUMO gap in Hartree, or None if not available.
        """
        return self._get_props("homo_lumo_gap")

    def get_dipole_moment(self) -> Optional[Dict[str, Any]]:
        """
        Retrieves the dipole moment.

        Returns:
            Optional[Dict[str, Any]]: Dipole moment data, or None if not available.
        """
        return self._get_props("dipole_moment")

    def get_effective_potential(self) -> Optional[Any]:
        """
        Retrieves the effective potential.

        Returns:
            Optional[Any]: Effective potential data, or None if not available.
        """
        return self._get_props("effective_potential")

    def get_electrostatic_potentials(self) -> Optional[Dict[str, Any]]:
        """
        Retrieves the electrostatic potentials.

        Returns:
            Optional[Dict[str, Any]]: Electrostatic potentials data, or None if not available.
        """
        return self._get_props("electrostatic_potentials")

    def get_wavefunctions(self) -> Optional[np.ndarray]:
        """
        Retrieves the molecular wavefunctions.

        Returns:
            Optional[np.ndarray]: Wavefunctions array, or None if not available.
        """
        wavefunctions = self._get_props("wavefunctions")
        return np.array(wavefunctions) if wavefunctions else None

    def get_potential_energy(self) -> Optional[float]:
        """
        Retrieves the potential energy.

        Returns:
            Optional[float]: Potential energy in Hartree, or None if not available.
        """
        return self._get_props("potential_energy")

    def get_zpe_hartree(self) -> Optional[float]:
        """
        Retrieves the zero-point energy (ZPE).

        Returns:
            Optional[float]: ZPE in Hartree, or None if not available.
        """
        return self._get_props("zpe_hartree")

    def get_E0_elec_plus_zpe(self) -> Optional[float]:
        """
        Retrieves the electronic energy plus ZPE.

        Returns:
            Optional[float]: E0 (electronic energy + ZPE) in Hartree, or None if not available.
        """
        return self._get_props("E0_elec_plus_zpe")

    def get_freqs_cm(self) -> Optional[np.ndarray]:
        """
        Retrieves the vibrational frequencies in cm^-1.

        Returns:
            Optional[np.ndarray]: Frequencies array, or None if not available.
        """
        freqs_cm = self._get_props("freqs_cm^-1")
        return np.array(freqs_cm) if freqs_cm else None

    def get_thermal_corr_internal_energy(self) -> Optional[float]:
        """
        Retrieves the thermal correction to the internal energy.

        Returns:
            Optional[float]: Thermal correction in Hartree, or None if not available.
        """
        return self._get_props("thermal_corr_internal_energy")

    def get_dispersion_energy(self) -> Optional[float]:
        """
        Retrieves the dispersion energy.

        Returns:
            Optional[float]: Dispersion energy in Hartree, or None if not available.
        """
        return self._get_props("dispersion_energy")

    def get_quadrupole_moment(self) -> Optional[List[float]]:
        """
        Retrieves the quadrupole moment.

        Returns:
            Optional[List[float]]: Quadrupole moment components, or None if not available.
        """
        return self._get_props("quadrupole_moment")

    def get_radius_of_gyration(self) -> Optional[float]:
        """
        Retrieves the radius of gyration.

        Returns:
            Optional[float]: Radius of gyration in Angstroms, or None if not available.
        """
        return self._get_props("radius_of_gyration")

    def get_free_energy(self) -> Optional[float]:
        """
        Retrieves the free energy.

        Returns:
            Optional[float]: Free energy in Hartree, or None if not available.
        """
        return self._get_props("free_energy")
