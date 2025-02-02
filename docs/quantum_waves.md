# QuantumWavesComplex

## Overview
`QuantumWavesComplex` inherits from `QuantumComplex` and specializes in **wavefunction-based quantum calculations** with extended capabilities.

=== "**Methods**"
- **`compute_long_range_interactions()`** → Calculates long-range molecular interactions.
- **`get_quadrupole_moment()`** → Returns quadrupole moments.
- **`get_radius_of_gyration()`** → Returns radius of gyration.
- **`get_free_energy()`** → Returns the computed free energy.
- **`get_dispersion_energy()`** → Returns dispersion energy contributions.
- **`get_forces()`** → Returns force matrix.
- **`get_positions()`** → Returns atomic positions.
- **`get_distance_matrix()`** → Returns the distance matrix.
- **`get_fermi_level()`** → Returns the Fermi level energy.
- **`get_eigenvalues()`** → Returns the eigenvalues of the Kohn-Sham matrix.
- **`get_homo_lumo_gap()`** → Returns the HOMO-LUMO gap.
- **`get_dipole_moment()`** → Returns dipole moment calculation.
- **`get_effective_potential()`** → Returns effective potential.
- **`get_electrostatic_potentials()`** → Returns electrostatic potential.
- **`get_wavefunctions()`** → Returns calculated wavefunctions.
- **`get_potential_energy()`** → Returns the potential energy of the molecule.
- **`get_zpe_hartree()`** → Returns zero-point energy in Hartree units.
- **`get_E0_elec_plus_zpe()`** → Returns electronic energy plus ZPE.
- **`get_freqs_cm()`** → Returns vibrational frequency values.
- **`get_thermal_corr_internal_energy()`** → Returns thermal correction to internal energy.


## Usage Example

```python title="Quantum Waves Complex" linenums="1"
from polyatomic_complexes.src.complexes import PolyatomicGeometrySMILE
from polyatomic_complexes.src.complexes.quantum_theor_complex import QuantumWavesComplex

pg = PolyatomicGeometrySMILE(smile="CC(=O)OC", mode="quantum-waves")
quantum_w_mol = pg.smiles_to_geom_complex()
energy = quantum_w_mol.get_potential_energy()
dispersion = quantum_w_mol.get_dispersion_energy()
interactions = quantum_w_mol.compute_long_range_interactions()
radius = quantum_w_mol.get_radius_of_gyration()
free_energy = quantum_w_mol.get_free_energy()
```


## Methods Explained

!!! note "compute_long_range_interactions()"
    This method computes quantum-level properties at a DFT (B3LYP) level of theory including geometry optimization, thermal corrections via a frequency calculation (harmonic approximation).

!!! note "get_quadrupole_moment()"
    This method returns the quadrupole moment tensor. Recall
    $$
    Q_{ij} = \sum_k q_k \left( 3r_{ki} r_{kj} - \delta_{ij} r_k^2 \right)
    $$
    where $q_k$ is the charge at nucleus/electron $k$, $r_{ki}$ is the $i$-th coordinate charge of $k$ and $\delta_{ij}$ is the Kronecker delta.
    
    Essentially we return $[Q_{xx}, Q_{yy}, Q_{zz}, Q_{xy}, Q_{xz}, Q_{yz}]$ as a `List`.

!!! note "get_radius_of_gyration()"
    This method returns the radius of gyration. Recall that $R_{g}$ is a measure of the spatial distribution of atomic positions around the center of mass (compactness). Moreover,
    $$
        R_g = \sqrt{\frac{\sum_{i} m_i r_i^2}{\sum_{i} m_i}}
    $$
    Note that a small $R_g$ usually corresponds to a compact structure like a folded protein or dense polymer. Large $R_g$ usually correspons to extended structures (stretched polymer).
    We return the mass weighted radius of gyration in Angstroms as a `float`.

!!! note "get_free_energy()"
    This method returns the thermal and dispersion corrected total/free energy. Recall
    $$
        G = E_{\text{elec}} + U_{\text{thermal}} + PV - TS
    $$
    where $E_{\text{elec}}$ is the electronic energy from HF/DFT, $U_{\text{thermal}}$ is the thermal correction, $PV$ is pressure-volume work, and $TS$ is temperature times entropy from a harmonic analysis (aka via hessian). From here we can compute 
    $$
        E_{\text{corrected}} =  E_{\text{elec}} + U_{\text{thermal}} + E_{\text{dispersion}}
    $$
    where $E_{\text{dispersion}}$ accounts for van der Waals interactions.
    We return $E_{\text{corrected}}$ as a `float` in Hartree.

!!! note "get_dispersion_energy()"
    This method returns the dispersion correction used above. Recall that dispersion energy accounts for long-range weak interactions that are not captured by standard DFT functionals. This method returns $E_{\text{dispersion}}$ as a `float`.

