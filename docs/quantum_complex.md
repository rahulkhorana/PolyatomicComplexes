# QuantumComplex

## Overview
`QuantumComplex` inherits from `AbstractComplex` and provides **quantum chemistry calculations**.

=== "**Methods**"
- **`E0_elec_plus_zpe()`** → Returns the electronic energy plus zero-point energy.
- **`fermi_level()`** → Returns the Fermi level of the system.
- **`eigenvalues()`** → Returns the eigenvalues of the Kohn-Sham matrix.
- **`homo_lumo_gap()`** → Returns the HOMO-LUMO gap.
- **`effective_potential()`** → Returns the effective potential of the system.
- **`dipole_moment()`** → Returns the dipole moment of the molecule.
- **`electrostatic_potentials()`** → Returns the electrostatic potential.
- **`wavefunctions()`** → Returns the LCAO Coefficient Matrix.
- **`potential_energy()`** → Returns the computed potential energy.
- **`zpe_hartree()`** → Returns zero-point energy in Hartree units.
- **`freqs_cm()`** → Returns vibrational frequencies in cm⁻¹.
- **`thermal_corr_internal_energy()`** → Returns the thermal correction to internal energy aka $U_{\text{thermal}}$.


## Usage Example

```python title="Quantum Complex" linenums="1"
from polyatomic_complexes.src.complexes import PolyatomicGeometrySMILE
from polyatomic_complexes.src.complexes.quantum_complex import QuantumComplex

pg = PolyatomicGeometrySMILE(smile="CC(=O)OC", mode="quantum")
quantum_mol = pg.smiles_to_geom_complex()
energy = quantum_mol.get_potential_energy()
wavefunctions = quantum_mol.get_wavefunctions()
dipole = quantum_mol.get_dipole_moment()
eigenvalues = quantum_mol.get_eigenvalues()
homo_lumo_gap = quantum_mol.get_homo_lumo_gap()
```


## Methods Explained


!!! note "E0_elec_plus_zpe()"
    This method returns total_energy + zpe_hartree as a `numpy.float`.

!!! note "fermi_level()"
    This method will return either $0.5 \cdot (homo + lumo)$ or `None`. The output is a scalar aka `numpy.float`.

!!! note "eigenvalues()"
    This method will return the eigenvalues of the Kohn-Sham matrix. The return type is a `List`.
    Recall from the Kohn-Sham equations:
    $$
    v_{\text{eff}}(r) = v_{ext}(r) + e^{2} \int \frac{p(r')}{\|r - r'\|}  dr' + \frac{\delta E_{xc}[p]}{\delta p(r)}
    $$
    where the last term is the exchange-correlation potential. We are returning the $v_{\text{eff}}$ matrix (eigenvalue equation) as a list.

!!! note "homo_lumo_gap()"
    This will return `lumo - homo` as a scalar. Essentially the return type is a `np.float`.

!!! note "effective_potential()"
    This method returns the Effective core potentials as a `numpy.ndarray`.

!!! note "dipole_moment()"
    Returns the dipole moment `dict` containing the keys: `"vector"` and `"magnitude"`. This follows the standard dipole moment calculation:
    $$
    \mu_x = - \sum_{\mu} \sum_{\mu} P_{\mu,\nu}(\nu | x| \mu) + \sum_{A} Q_{A} X_{A}$$
    $$
    \mu_y = - \sum_{\mu} \sum_{\mu} P_{\mu,\nu}(\nu | y| \mu) + \sum_{A} Q_{A} Y_{A}$$
    $$
    \mu_z =  - \sum_{\mu} \sum_{\mu} P_{\mu,\nu}(\nu | z| \mu) + \sum_{A} Q_{A} Z_{A}
    $$
    Note that `"vector"` will be a list and `"magnitude"` a scalar aka `numpy.float`.

!!! note "electrostatic_potentials()"
    This method returns the electrostatic potential as a `List[List]` and the grid as a `List[List]`. The actual return value is a `dict` with  the keys: `"grid_coords"` and `"total_electrostatic_potential"`. Note that this is the usual molecular electrostatic potential (MEP) at a slightly higher resolution than normal.

!!! note "wavefunctions()"
    The electronic wavefunction $\Psi$ is typically expressed as a LCAO:
    $$
        \Psi_i = \sum_{j} C_{ji} \phi_j
    $$
    In matrix form this becomes $\Psi = C \Phi$. We return the $C$ matrix in this method. One can derive $\Phi$ from the atomic basis functions at grid points. An end user can get the $\Psi$ matrix by specifying a basis and performing matrix multiplication. This may not always be necessary and is extremely expensive to compute as the basis and grid scales. Therefore we only provide the $C$ matrix. The return type is a `List[List[float]]`. We may integrate computation of $\Phi$ in a later release. Conceptually this matrix describes how the atomic orbitals mix into MOs.

!!! note "potential_energy()"
    This returns the potential energy as a scalar aka `numpy.float`.

!!! note "zpe_hartree()"
    This returns the ZPE as a `numpy.float`.

!!! note "freqs_cm()"
    This method uses the Hessian matrix and extracts the vibrational frequencies in wavenumbers $cm^{-1}$. This is returned as a `list`.

!!! note "thermal_corr_internal_energy()"
    This method calculates the thermal correction to internal energy, which includes contributions from zero-point energy (ZPE), vibrational, translational, and rotational energies.
    Mathematically, it returns:
    $U_{\text{thermal}} = \text{ZPE} + U_{\text{vib}} + U_{\text{trans}} + U_{\text{rot}}$.
    The return value is a `numpy.float`.