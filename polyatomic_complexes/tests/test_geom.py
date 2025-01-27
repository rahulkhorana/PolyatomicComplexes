import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


from polyatomic_complexes.src.complexes.polyatomic_geometry import (
    PolyatomicGeometrySMILE,
)

from polyatomic_complexes.src.complexes.atomic_complex import AtomComplex
from polyatomic_complexes.src.complexes.force_complex import ForceComplex
from polyatomic_complexes.src.complexes.quantum_complex import QuantumComplex
from polyatomic_complexes.src.complexes.abstract_complex import AbstractComplex
from polyatomic_complexes.src.complexes.quantum_theor_complex import QuantumWavesComplex


from polyatomic_complexes.src.complexes.space_utils import (
    geometricPolyatomicComplex,
    geometricAtomicComplex,
)

from typing import List
import os
import pytest
import json
import random
import numpy as np
from collections import defaultdict
from toponetx import CombinatorialComplex
from scipy.sparse import coo_matrix
from networkx import Graph


def nice_print(argname, arg):
    print("*" * 20)
    print(f"The {argname} is")
    print(f"type is: {type(arg)}")
    if isinstance(arg, dict):
        print(arg.keys())
    if isinstance(arg, defaultdict):
        print(arg.keys())
    if isinstance(arg, list):
        print(f"internal size: {len(arg[0])}")
        print(f"internal tupe: {[type(x) for x in arg]}")
    if isinstance(arg, np.ndarray):
        print(f"internal size: {arg.shape}")
        print(f"internal size: {arg[0].shape}")
    if isinstance(arg, tuple):
        print(f"lengthis: {len(arg)}")
        print(f"internal tupe: {[type(x) for x in arg]}")
    print("*" * 20)


def check_abstract_complex_test(complex: AbstractComplex):
    complex_ac = complex.abstract_complex()
    assert isinstance(complex_ac, defaultdict)
    for k in complex_ac.keys():
        assert len(k) == 2 and isinstance(k, tuple)
        assert isinstance(k[0], AtomComplex) and isinstance(k[1], str)
    complex_roc = complex.rank_order_complex()
    assert isinstance(complex_roc, defaultdict)
    assert "molecule" in complex_roc and len(complex_roc["molecule"]) == 2
    assert (
        "nuclear_structure" in complex_roc
        and len(complex_roc["nuclear_structure"]) == 2
    )
    assert (
        "electronic_structure" in complex_roc
        and len(complex_roc["electronic_structure"]) == 2
    )
    required_features = [
        "incidence",
        "laplacians",
        "adjacencies",
        "co_adjacencies",
        "skeleta",
        "all_cell_coadj",
        "dirac",
    ]
    check_features = lambda column: set(
        [
            required_features[i] in complex_roc[column][1]
            for i in range(len(required_features))
        ]
    )
    assert isinstance(complex_roc["molecule"][0], CombinatorialComplex)
    assert isinstance(complex_roc["molecule"][1], defaultdict)
    assert True in check_features("molecule") and len(check_features("molecule")) == 1
    assert isinstance(complex_roc["nuclear_structure"][0], CombinatorialComplex)
    assert isinstance(complex_roc["nuclear_structure"][1], defaultdict)
    assert (
        True in check_features("nuclear_structure")
        and len(check_features("nuclear_structure")) == 1
    )
    assert isinstance(complex_roc["electronic_structure"][0], CombinatorialComplex)
    assert isinstance(complex_roc["electronic_structure"][1], defaultdict)
    assert (
        True in check_features("electronic_structure")
        and len(check_features("electronic_structure")) == 1
    )
    check_types = lambda column: set(
        [
            len(x) == 3
            and isinstance(x[0], defaultdict)
            and isinstance(x[2], defaultdict)
            for x in column
        ]
    )
    atm_struct = complex.atomic_structure()
    assert isinstance(atm_struct, list)
    assert True in check_types(atm_struct) and len(check_types(atm_struct)) == 1
    bonds = complex.bonds()
    check_bonds = lambda bnds: set(
        [
            len(b) == 3
            and isinstance(b, tuple)
            and isinstance(b[0], str)
            and isinstance(b[1], str)
            and isinstance(b[2], list)
            for b in bnds
        ]
    )
    assert isinstance(bonds, list)
    assert True in check_bonds(bonds) and len(check_bonds(bonds)) == 1
    expected_to_NI_methods = [
        complex.forces,
        complex.electrostatics,
        complex.get_forces,
        complex.get_electrostatics,
        complex.wavefunctions,
    ]
    for method in expected_to_NI_methods:
        with pytest.raises(NotImplementedError) as excinfo:
            method()
        assert (
            str(excinfo.value)
            == "This is not defined behavior for an Abstract Complex!"
        )
    ato_top = complex.atomic_topology()
    rk, feats = ato_top
    assert isinstance(rk, list) and isinstance(feats, dict)
    assert set(
        [isinstance(r, tuple) and len(r) == 2 and isinstance(r[1], list) for r in rk]
    ) == set([True])
    ato_struct = complex.atomic_structure()
    assert isinstance(ato_struct, list)
    adj = complex.get_adjacencies()
    assert isinstance(adj, defaultdict)
    assert "molecule_adjacencies" in adj
    for sub_adj1 in adj["molecule_adjacencies"]:
        for term1 in sub_adj1:
            assert isinstance(term1[0], str)
            assert isinstance(term1[1], np.ndarray)
    all_cell_coaj = complex.get_all_cell_coadj()
    assert isinstance(all_cell_coaj, defaultdict)
    assert "molecule_all_cell_coadj" in all_cell_coaj
    assert isinstance(all_cell_coaj["molecule_all_cell_coadj"], list)
    assert isinstance(all_cell_coaj["molecule_all_cell_coadj"][0], list)
    try:
        all_cell = all_cell_coaj["molecule_all_cell_coadj"]
        data = [np.asarray(row) for row in all_cell]
        as_arr = np.asarray(data, dtype=object)
        assert isinstance(as_arr, np.ndarray)
        for c in as_arr:
            assert isinstance(c, np.ndarray)
    except Exception as e:
        raise e
    get_structure = complex.get_atomic_structure()
    assert isinstance(get_structure, list) and len(get_structure) > 0
    get_ato = complex.get_atomic_topology()
    rk, feats = get_ato
    assert isinstance(rk, list) and isinstance(feats, dict)
    assert set(
        [isinstance(r, tuple) and len(r) == 2 and isinstance(r[1], list) for r in rk]
    ) == set([True])
    get_bonds = complex.get_bonds()
    assert isinstance(get_bonds, list)
    assert True in check_bonds(get_bonds) and len(check_bonds(get_bonds)) == 1
    get_betti = complex.get_betti_numbers()
    assert isinstance(get_betti, defaultdict)
    assert "molecule_betti_numbers" in get_betti
    assert get_betti["molecule_betti_numbers"] is None or isinstance(
        get_betti["molecule_betti_numbers"], list
    )
    get_incidences = complex.get_incidence()
    assert isinstance(get_incidences, defaultdict)
    assert "molecule_incidence" in get_incidences
    for item in get_incidences["molecule_incidence"]:
        assert isinstance(item, dict) and len(item.keys()) > 0
        for key in item.keys():
            value = item[key]
            assert isinstance(value, dict)
            for particles in value:
                assert isinstance(particles, frozenset)
                assert len(particles) > 0
                for p in particles:
                    assert len(p) == 2
                    assert isinstance(p[0], str)
                    assert p[0].split("_")[0] in set(["E", "P", "N"])
                    assert isinstance(p[1], tuple)
                    if len(p[1]) == 2:
                        assert p[1][0] in set(["electron", "proton", "neutron"])
                        try:
                            arr = np.frombuffer(p[1][1], dtype=np.uint8)
                            assert arr.shape != 0
                        except:
                            raise Exception("invalid incidence")
                    elif len(p[1]) == 3:
                        assert p[1][0] in set(["electron", "proton", "neutron"])
                        try:
                            arr = np.frombuffer(p[1][1], dtype=np.uint8)
                            arr_w = np.frombuffer(p[1][2], dtype=np.uint8)
                            assert arr.shape != 0 and arr_w.shape != 0
                        except:
                            raise Exception("invalid incidence")
                    else:
                        assert p[1][0] in set(["electron", "proton", "neutron"])
                        try:
                            for _, items in enumerate(p[1][1:]):
                                arr = np.frombuffer(items, dtype=np.uint8)
                                assert arr.shape != 0
                        except:
                            raise Exception("INVALID")
    get_abs_complex = complex.get_complex("abstract_complex")
    assert isinstance(get_abs_complex, defaultdict)
    for k in get_abs_complex.keys():
        assert len(k) == 2 and isinstance(k, tuple)
        assert isinstance(k[0], AtomComplex) and isinstance(k[1], str)
    get_rk_complex = complex.get_complex("rank_order")
    assert isinstance(get_rk_complex, defaultdict)
    assert "molecule" in get_rk_complex and len(get_rk_complex["molecule"]) == 2
    assert (
        "nuclear_structure" in get_rk_complex
        and len(get_rk_complex["nuclear_structure"]) == 2
    )
    assert (
        "electronic_structure" in get_rk_complex
        and len(get_rk_complex["electronic_structure"]) == 2
    )
    required_features = [
        "incidence",
        "laplacians",
        "adjacencies",
        "co_adjacencies",
        "skeleta",
        "all_cell_coadj",
        "dirac",
    ]
    check_features = lambda column: set(
        [
            required_features[i] in get_rk_complex[column][1]
            for i in range(len(required_features))
        ]
    )
    assert isinstance(get_rk_complex["molecule"][0], CombinatorialComplex)
    assert isinstance(get_rk_complex["molecule"][1], defaultdict)
    assert True in check_features("molecule") and len(check_features("molecule")) == 1
    assert isinstance(get_rk_complex["nuclear_structure"][0], CombinatorialComplex)
    assert isinstance(get_rk_complex["nuclear_structure"][1], defaultdict)
    assert (
        True in check_features("nuclear_structure")
        and len(check_features("nuclear_structure")) == 1
    )
    assert isinstance(get_rk_complex["electronic_structure"][0], CombinatorialComplex)
    assert isinstance(get_rk_complex["electronic_structure"][1], defaultdict)
    assert (
        True in check_features("electronic_structure")
        and len(check_features("electronic_structure")) == 1
    )
    get_dirac = complex.get_dirac()
    assert isinstance(get_dirac, defaultdict)
    assert "molecule_dirac" in get_dirac
    assert (
        get_dirac["molecule_dirac"] is None
        or isinstance(get_dirac["molecule_dirac"], np.ndarray)
        or isinstance(get_dirac["molecule_dirac"], coo_matrix)
        or isinstance(get_dirac["molecule_dirac"], list)
    )
    if isinstance(get_dirac["molecule_dirac"], list):
        try:
            lst = np.asarray(get_dirac["molecule_dirac"][0])
            assert lst.shape != 0
        except:
            raise Exception("unconverted")
    get_def_or = complex.get_default_orientations()
    assert isinstance(get_def_or, dict) or isinstance(get_def_or, int)
    get_lap = complex.get_laplacians()
    assert isinstance(get_lap, defaultdict)
    assert "molecule_laplacians" in get_lap
    for sub_lap1 in get_lap["molecule_laplacians"]:
        for term1 in sub_lap1:
            assert isinstance(term1[0], str)
            assert isinstance(term1[1], np.ndarray)
    get_pc = complex.get_pc_matrix("stacked")
    assert isinstance(get_pc, list)
    assert len(get_pc) > 0
    assert isinstance(get_pc[0], Graph)
    get_pers = complex.get_persistence()
    assert isinstance(get_pers, defaultdict)
    assert get_pers["molecule_persistence"] is None or isinstance(
        get_pers["molecule_persistence"], list
    )
    get_sk = complex.get_skeleta()
    assert isinstance(get_sk, defaultdict)
    assert "molecule_skeleta" in get_sk
    assert (
        isinstance(get_sk["molecule_skeleta"], list)
        and len(get_sk["molecule_skeleta"]) > 0
        and len(get_sk["molecule_skeleta"][0]) > 0
    )
    get_coadj = complex.get_coadjacencies()
    assert isinstance(get_coadj, defaultdict)
    assert "molecule_co_adjacencies" in get_coadj
    for sub_adj1 in get_coadj["molecule_co_adjacencies"]:
        for term1 in sub_adj1:
            assert isinstance(term1[0], str)
            assert isinstance(term1[1], np.ndarray)
    return True


def check_force_complex_test(complex: ForceComplex):
    forces = complex.forces()
    assert isinstance(forces, np.ndarray)
    assert forces.shape != 0 and forces.shape[1] == 3
    electrostatics = complex.electrostatics()
    assert isinstance(electrostatics, np.ndarray)
    assert electrostatics.shape != 0
    positions = complex.positions()
    assert isinstance(positions, np.ndarray)
    dist = complex.dist_matrix()
    assert isinstance(dist, np.ndarray)
    symb = complex.symbols()
    assert isinstance(symb, list)
    return True


def check_quantum_complex_test(complex: QuantumComplex):
    try:
        forces = complex.forces()
        assert isinstance(forces, np.ndarray), "Forces should be a numpy.ndarray"
        assert forces.size != 0, "Forces array should not be empty"
        assert forces.ndim == 2, "Forces array should be 2D"
        assert forces.shape[1] == 3, "Forces array should have shape (N, 3)"
        assert np.any(forces != 0), "Forces should have at least one non-zero component"
        electrostatics = complex.electrostatic_potentials()
        assert isinstance(
            electrostatics, dict
        ), "Electrostatic potentials should be a dict"
        grid_coords = electrostatics.get("grid_coords")
        total_esp = electrostatics.get("total_electrostatic_potential")
        assert isinstance(
            grid_coords, list
        ), "Electrostatic 'grid_coords' should be a list"
        assert isinstance(
            total_esp, list
        ), "Electrostatic 'total_electrostatic_potential' should be a list"
        assert len(grid_coords) > 0, "Electrostatic 'grid_coords' should not be empty"
        assert (
            len(total_esp) > 0
        ), "Electrostatic 'total_electrostatic_potential' should not be empty"
        positions = complex.positions()
        assert isinstance(positions, np.ndarray), "Positions should be a numpy.ndarray"
        assert positions.size != 0, "Positions array should not be empty"
        assert positions.ndim == 2, "Positions array should be 2D"
        assert positions.shape[1] == 3, "Positions array should have shape (N, 3)"
        dist = complex.distance_matrix()
        assert isinstance(dist, np.ndarray), "Distance matrix should be a numpy.ndarray"
        assert dist.size != 0, "Distance matrix should not be empty"
        assert dist.ndim == 2, "Distance matrix should be 2D"
        assert dist.shape[0] == dist.shape[1], "Distance matrix should be square"
        assert np.all(dist >= 0), "All distances should be non-negative"
        assert not np.all(dist == 0), "Not all distances should be zero"
        assert len(np.unique(dist)) > 1, "Distance matrix should have distinct values"
        fermi_level = complex.fermi_level()
        assert (
            isinstance(fermi_level, float) or fermi_level is None
        ), "Fermi level should be a float or None"
        if fermi_level is not None:
            assert fermi_level != 0.0, "Fermi level should not be zero"
        eigenvalues = complex.eigenvalues()
        assert isinstance(
            eigenvalues, np.ndarray
        ), "Eigenvalues should be a numpy.ndarray"
        assert eigenvalues.size != 0, "Eigenvalues array should not be empty"
        assert eigenvalues.ndim == 1, "Eigenvalues array should be 1D"
        assert np.all(
            np.diff(eigenvalues) >= 0
        ), "Eigenvalues should be sorted in ascending order"
        homo_lumo_gap = complex.homo_lumo_gap()
        assert (
            isinstance(homo_lumo_gap, float) or homo_lumo_gap is None
        ), "HOMO-LUMO gap should be a float or None"
        if homo_lumo_gap is not None:
            assert homo_lumo_gap > 0.0, "HOMO-LUMO gap should be positive"
        dipole_moment = complex.dipole_moment()
        assert isinstance(dipole_moment, dict), "Dipole moment should be a dict"
        assert (
            "vector" in dipole_moment and "magnitude" in dipole_moment
        ), "Dipole moment should contain 'vector' and 'magnitude'"
        vector = dipole_moment["vector"]
        magnitude = dipole_moment["magnitude"]
        assert isinstance(vector, list), "Dipole vector should be a list"
        assert len(vector) == 3, "Dipole vector should have 3 components"
        assert all(
            isinstance(v, float) for v in vector
        ), "Dipole vector components should be floats"
        assert isinstance(magnitude, float), "Dipole magnitude should be a float"
        assert magnitude > 0.0, "Dipole magnitude should be positive"
        effective_potential = complex.effective_potential()
        assert effective_potential is not None, "Effective potential should not be None"
        if isinstance(effective_potential, np.ndarray):
            assert (
                effective_potential.size != 0
            ), "Effective potential array should not be empty"
        wavefunctions = complex.wavefunctions()
        assert isinstance(
            wavefunctions, np.ndarray
        ), "Wavefunctions should be a numpy.ndarray"
        assert wavefunctions.size != 0, "Wavefunctions array should not be empty"
        assert wavefunctions.ndim == 2, "Wavefunctions array should be 2D"
        unique_wavefunctions = np.unique(wavefunctions, axis=0)
        assert (
            unique_wavefunctions.shape[0] > 1
        ), "Wavefunctions should have distinct values"
        potential_energy = complex.potential_energy()
        assert isinstance(potential_energy, float), "Potential energy should be a float"
        assert potential_energy != 0.0, "Potential energy should not be zero"
        zpe_hartree = complex.zpe_hartree()
        assert isinstance(zpe_hartree, float), "ZPE Hartree should be a float"
        assert zpe_hartree > 0.0, "ZPE Hartree should be positive"
        E0_elec_plus_zpe = complex.E0_elec_plus_zpe()
        assert isinstance(E0_elec_plus_zpe, float), "E0_elec_plus_zpe should be a float"
        assert E0_elec_plus_zpe != 0.0, "E0_elec_plus_zpe should not be zero"
        freqs_cm = complex.freqs_cm()
        assert isinstance(
            freqs_cm, np.ndarray
        ), "Frequencies (cm^-1) should be a numpy.ndarray"
        assert freqs_cm.size != 0, "Frequencies array should not be empty"
        assert freqs_cm.ndim == 1, "Frequencies array should be 1D"
        assert np.all(freqs_cm > 0.0), "Frequencies should be positive"
        thermal_corr_internal_energy = complex.thermal_corr_internal_energy()
        assert isinstance(
            thermal_corr_internal_energy, float
        ), "Thermal correction internal energy should be a float"
        assert (
            thermal_corr_internal_energy != 0.0
        ), "Thermal correction internal energy should not be zero"
    except AssertionError as e:
        print(f"Test failed: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")
        return False
    return True


def check_quantum_waves_complex_test(complex: QuantumWavesComplex):
    properties = complex._compute_quantum_properties()
    assert isinstance(
        properties, dict
    ), "compute_quantum_properties should return a dict."
    essential_keys = [
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
    ]
    for key in essential_keys:
        assert (
            key in properties
        ), f"Missing key '{key}' in compute_quantum_properties output."
    forces = properties["forces"]
    assert isinstance(forces, list), "Forces should be a list."
    for force in forces:
        assert isinstance(force, list), "Each force should be a list."
        assert len(force) == 3, "Each force list should have exactly 3 elements."
        assert all(
            isinstance(component, (float, int)) for component in force
        ), "Force components should be floats or ints."
    refined_positions = properties["refined_positions"]
    assert isinstance(refined_positions, list), "refined_positions should be a list."
    for pos in refined_positions:
        assert isinstance(pos, list), "Each position should be a list."
        assert len(pos) == 3, "Each position list should have exactly 3 elements."
        assert all(
            isinstance(coord, (float, int)) for coord in pos
        ), "Position coordinates should be floats or ints."
    dist_matrix = properties["dist_matrix"]
    assert isinstance(dist_matrix, list), "dist_matrix should be a list."
    N = len(refined_positions)
    assert len(dist_matrix) == N, f"dist_matrix should have {N} rows."
    for row in dist_matrix:
        assert isinstance(row, list), "Each row in dist_matrix should be a list."
        assert len(row) == N, f"Each row in dist_matrix should have {N} elements."
        assert all(
            isinstance(distance, (float, int)) for distance in row
        ), "Distances should be floats or ints."
    fermi_level = properties["fermi_level"]
    assert (
        isinstance(fermi_level, float) or fermi_level is None
    ), "fermi_level should be a float or None."
    eigenvalues = properties["eigenvalues"]
    assert isinstance(eigenvalues, list), "eigenvalues should be a list."
    assert all(
        isinstance(ev, (float, int)) for ev in eigenvalues
    ), "Each eigenvalue should be a float or int."
    homo_lumo_gap = properties["homo_lumo_gap"]
    assert (
        isinstance(homo_lumo_gap, float) or homo_lumo_gap is None
    ), "homo_lumo_gap should be a float or None."
    dipole_moment = properties["dipole_moment"]
    assert isinstance(dipole_moment, dict), "dipole_moment should be a dict."
    assert (
        "vector" in dipole_moment and "magnitude" in dipole_moment
    ), "dipole_moment should have 'vector' and 'magnitude'."
    dipole_vector = dipole_moment["vector"]
    assert isinstance(dipole_vector, list), "dipole_moment['vector'] should be a list."
    assert (
        len(dipole_vector) == 3
    ), "dipole_moment['vector'] should have exactly 3 elements."
    assert all(
        isinstance(component, (float, int)) for component in dipole_vector
    ), "Dipole vector components should be floats or ints."
    dipole_magnitude = dipole_moment["magnitude"]
    assert isinstance(
        dipole_magnitude, (float, int)
    ), "dipole_moment['magnitude'] should be a float or int."
    effective_potential = properties["effective_potential"]
    assert isinstance(
        effective_potential, (list, type(None))
    ), "effective_potential should be a list or None."
    if isinstance(effective_potential, list):
        assert all(
            isinstance(v, (float, int)) for v in effective_potential
        ), "Effective potential components should be floats or ints."
    electrostatic_potentials = properties["electrostatic_potentials"]
    assert isinstance(
        electrostatic_potentials, dict
    ), "electrostatic_potentials should be a dict."
    assert (
        "grid_coords" in electrostatic_potentials
        and "total_electrostatic_potential" in electrostatic_potentials
    ), "electrostatic_potentials should contain 'grid_coords' and 'total_electrostatic_potential'."
    grid_coords = electrostatic_potentials["grid_coords"]
    assert isinstance(
        grid_coords, list
    ), "electrostatic_potentials['grid_coords'] should be a list."
    for coord in grid_coords:
        assert isinstance(coord, list), "Each grid coordinate should be a list."
        assert (
            len(coord) == 3
        ), "Each grid coordinate list should have exactly 3 elements."
        assert all(
            isinstance(c, (float, int)) for c in coord
        ), "Grid coordinate elements should be floats or ints."
    total_esp = electrostatic_potentials["total_electrostatic_potential"]
    assert isinstance(
        total_esp, list
    ), "electrostatic_potentials['total_electrostatic_potential'] should be a list."
    assert len(total_esp) == len(
        grid_coords
    ), "Length of 'total_electrostatic_potential' should match 'grid_coords'."
    assert all(
        isinstance(v, (float, int)) for v in total_esp
    ), "Electrostatic potential values should be floats or ints."
    wavefunctions = properties["wavefunctions"]
    assert isinstance(wavefunctions, list), "wavefunctions should be a list."
    for wf in wavefunctions:
        assert isinstance(wf, list), "Each wavefunction should be a list."
        assert all(
            isinstance(coeff, (float, int)) for coeff in wf
        ), "Wavefunction coefficients should be floats or ints."
    potential_energy = properties["potential_energy"]
    assert isinstance(potential_energy, float), "potential_energy should be a float."
    zpe_hartree = properties["zpe_hartree"]
    assert isinstance(zpe_hartree, float), "zpe_hartree should be a float."
    E0_elec_plus_zpe = properties["E0_elec_plus_zpe"]
    assert isinstance(E0_elec_plus_zpe, float), "E0_elec_plus_zpe should be a float."
    freqs_cm = properties["freqs_cm^-1"]
    assert isinstance(freqs_cm, list), "freqs_cm^-1 should be a list."
    assert all(
        isinstance(freq, (float, int)) for freq in freqs_cm
    ), "Frequencies should be floats or ints."
    thermal_corr_internal_energy = properties["thermal_corr_internal_energy"]
    assert isinstance(
        thermal_corr_internal_energy, float
    ), "thermal_corr_internal_energy should be a float."
    dispersion_energy = properties["dispersion_energy"]
    assert (
        isinstance(dispersion_energy, float) or dispersion_energy is None
    ), "dispersion_energy should be a float or None."
    complex.compute_long_range_interactions()
    long_range_props = complex.computed_props
    assert (
        "thermal_energy" in long_range_props
        and "dispersion_energy" in long_range_props
        and "quadrupole_moment" in long_range_props
        and "radius_of_gyration" in long_range_props
        and "interatomic_distances" in long_range_props
        and "free_energy" in long_range_props
    )
    dispersion_energy_lr = long_range_props.get("dispersion_energy")
    assert (
        isinstance(dispersion_energy_lr, float) or dispersion_energy is None
    ), "dispersion_energy should be a float or None."
    quadrupole_moment = long_range_props.get("quadrupole_moment")
    assert (
        isinstance(quadrupole_moment, list) or quadrupole_moment is None
    ), "quadrupole_moment should be a list or None."
    if isinstance(quadrupole_moment, list):
        assert (
            len(quadrupole_moment) == 6
        ), "quadrupole_moment should have exactly 6 elements."
        assert all(
            isinstance(q, (float, int)) for q in quadrupole_moment
        ), "Each quadrupole component should be a float or int."
    radius_of_gyration = long_range_props.get("radius_of_gyration")
    assert (
        isinstance(radius_of_gyration, float) or radius_of_gyration is None
    ), "radius_of_gyration should be a float or None."
    interatomic_distances = long_range_props.get("interatomic_distances")
    assert isinstance(
        interatomic_distances, list
    ), "interatomic_distances should be a list."
    for row in interatomic_distances:
        assert isinstance(
            row, list
        ), "Each row in interatomic_distances should be a list."
        assert (
            len(row) == N
        ), f"Each row in interatomic_distances should have exactly {N} elements."
        assert all(
            isinstance(distance, (float, int)) for distance in row
        ), "Distances should be floats or ints."
    free_energy = long_range_props.get("free_energy")
    assert (
        isinstance(free_energy, float) or free_energy is None
    ), "free_energy should be a float or None."
    forces_accessor = complex.get_forces()
    assert isinstance(
        forces_accessor, np.ndarray
    ), "get_forces should return a NumPy array."
    assert (
        forces_accessor.ndim == 2 and forces_accessor.shape[1] == 3
    ), "Forces array should have shape (N, 3)."
    assert forces_accessor.shape[0] == N, "Forces array should have N rows."
    assert np.issubdtype(
        forces_accessor.dtype, np.number
    ), "Forces array should contain numerical values."
    positions_accessor = complex.get_positions()
    assert isinstance(
        positions_accessor, np.ndarray
    ), "get_positions should return a NumPy array."
    assert (
        positions_accessor.ndim == 2 and positions_accessor.shape[1] == 3
    ), "Positions array should have shape (N, 3)."
    assert positions_accessor.shape[0] == N, "Positions array should have N rows."
    assert np.issubdtype(
        positions_accessor.dtype, np.number
    ), "Positions array should contain numerical values."
    distance_matrix_accessor = complex.get_distance_matrix()
    assert isinstance(
        distance_matrix_accessor, np.ndarray
    ), "get_distance_matrix should return a NumPy array."
    assert distance_matrix_accessor.shape == (
        N,
        N,
    ), f"Distance matrix should have shape ({N}, {N})."
    assert np.issubdtype(
        distance_matrix_accessor.dtype, np.number
    ), "Distance matrix should contain numerical values."
    fermi_level_accessor = complex.get_fermi_level()
    assert (
        isinstance(fermi_level_accessor, float) or fermi_level_accessor is None
    ), "get_fermi_level should return a float or None."
    eigenvalues_accessor = complex.get_eigenvalues()
    assert isinstance(
        eigenvalues_accessor, np.ndarray
    ), "get_eigenvalues should return a NumPy array."
    assert eigenvalues_accessor.ndim == 1, "Eigenvalues array should be 1-dimensional."
    assert eigenvalues_accessor.shape[0] == len(
        eigenvalues
    ), "Eigenvalues array length should match computed eigenvalues."
    assert np.issubdtype(
        eigenvalues_accessor.dtype, np.number
    ), "Eigenvalues array should contain numerical values."
    homo_lumo_gap_accessor = complex.get_homo_lumo_gap()
    assert (
        isinstance(homo_lumo_gap_accessor, float) or homo_lumo_gap_accessor is None
    ), "get_homo_lumo_gap should return a float or None."
    dipole_moment_accessor = complex.get_dipole_moment()
    assert isinstance(
        dipole_moment_accessor, dict
    ), "get_dipole_moment should return a dict."
    assert (
        "vector" in dipole_moment_accessor and "magnitude" in dipole_moment_accessor
    ), "Dipole moment should have 'vector' and 'magnitude'."
    dipole_vector_accessor = dipole_moment_accessor["vector"]
    dipole_magnitude_accessor = dipole_moment_accessor["magnitude"]
    assert isinstance(dipole_vector_accessor, list), "Dipole vector should be a list."
    assert (
        len(dipole_vector_accessor) == 3
    ), "Dipole vector should have exactly 3 elements."
    assert all(
        isinstance(component, (float, int)) for component in dipole_vector_accessor
    ), "Dipole vector components should be floats or ints."
    assert isinstance(
        dipole_magnitude_accessor, float
    ), "Dipole magnitude should be a float."
    effective_potential_accessor = complex.get_effective_potential()
    assert isinstance(
        effective_potential_accessor, (list, type(None))
    ), "get_effective_potential should return a list or None."
    if isinstance(effective_potential_accessor, list):
        assert all(
            isinstance(v, (float, int)) for v in effective_potential_accessor
        ), "Effective potential components should be floats or ints."
    electrostatic_potentials_accessor = complex.get_electrostatic_potentials()
    assert isinstance(
        electrostatic_potentials_accessor, dict
    ), "get_electrostatic_potentials should return a dict."
    assert (
        "grid_coords" in electrostatic_potentials_accessor
        and "total_electrostatic_potential" in electrostatic_potentials_accessor
    ), "Electrostatic potentials should contain 'grid_coords' and 'total_electrostatic_potential'."
    grid_coords_accessor = electrostatic_potentials_accessor["grid_coords"]
    assert isinstance(grid_coords_accessor, list), "grid_coords should be a list."
    for coord in grid_coords_accessor:
        assert isinstance(coord, list), "Each grid coordinate should be a list."
        assert (
            len(coord) == 3
        ), "Each grid coordinate list should have exactly 3 elements."
        assert all(
            isinstance(c, (float, int)) for c in coord
        ), "Grid coordinate elements should be floats or ints."
    total_esp_accessor = electrostatic_potentials_accessor[
        "total_electrostatic_potential"
    ]
    assert isinstance(
        total_esp_accessor, list
    ), "total_electrostatic_potential should be a list."
    assert len(total_esp_accessor) == len(
        grid_coords_accessor
    ), "Length of total_electrostatic_potential should match grid_coords."
    assert all(
        isinstance(v, (float, int)) for v in total_esp_accessor
    ), "Electrostatic potential values should be floats or ints."
    wavefunctions_accessor = complex.get_wavefunctions()
    assert isinstance(
        wavefunctions_accessor, np.ndarray
    ), "get_wavefunctions should return a NumPy array."
    assert (
        wavefunctions_accessor.ndim == 2
    ), "Wavefunctions array should be 2-dimensional."
    assert wavefunctions_accessor.shape == (
        len(wavefunctions),
        len(wavefunctions[0]),
    ), "Wavefunctions array shape mismatch."
    assert np.issubdtype(
        wavefunctions_accessor.dtype, np.number
    ), "Wavefunctions array should contain numerical values."
    potential_energy_accessor = complex.get_potential_energy()
    assert isinstance(
        potential_energy_accessor, float
    ), "get_potential_energy should return a float."
    zpe_hartree_accessor = complex.get_zpe_hartree()
    assert isinstance(
        zpe_hartree_accessor, float
    ), "get_zpe_hartree should return a float."
    E0_elec_plus_zpe_accessor = complex.get_E0_elec_plus_zpe()
    assert isinstance(
        E0_elec_plus_zpe_accessor, float
    ), "get_E0_elec_plus_zpe should return a float."
    freqs_cm_accessor = complex.get_freqs_cm()
    assert isinstance(
        freqs_cm_accessor, np.ndarray
    ), "get_freqs_cm should return a NumPy array."
    assert freqs_cm_accessor.ndim == 1, "freqs_cm should be a 1-dimensional array."
    assert freqs_cm_accessor.shape[0] == len(
        freqs_cm
    ), "freqs_cm array length mismatch."
    assert np.issubdtype(
        freqs_cm_accessor.dtype, np.number
    ), "freqs_cm array should contain numerical values."
    thermal_corr_internal_energy_accessor = complex.get_thermal_corr_internal_energy()
    assert isinstance(
        thermal_corr_internal_energy_accessor, float
    ), "get_thermal_corr_internal_energy should return a float."
    dispersion_energy_accessor = complex.get_dispersion_energy()
    assert (
        isinstance(dispersion_energy_accessor, float) or dispersion_energy is None
    ), "get_dispersion_energy should return a float or None."
    try:
        complex.visualize_property("refined_positions", title="Test Refined Positions")
    except Exception as e:
        pytest.fail(f"visualize_property('refined_positions') raised an exception: {e}")
    try:
        complex.visualize_property("dipole_moment", title="Test Dipole Moment")
    except Exception as e:
        pytest.fail(f"visualize_property('dipole_moment') raised an exception: {e}")
    try:
        complex.visualize_property("wavefunctions", title="Test Wavefunctions")
    except Exception as e:
        pytest.fail(f"visualize_property('wavefunctions') raised an exception: {e}")

    return True


smiles = [
    "OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O",
    "Cc1occc1C(=O)Nc2ccccc2",
    "c1ccc2c(c1)ccc3c2ccc4c5ccccc5ccc43",
    "CCc1ccccn1",
    "COc1ccc(cc1)N2CCN(CC2)C(=O)[C@@H]3CCCC[C@H]3C(=O)NCC#N",
    "Cn1c(CN2CCN(CC2)c3ccc(Cl)cc3)nc4ccccc14",
]
modes = ["abstract", "force-field"]
# quantum not tested

cases = []
for s in smiles:
    for m in modes:
        cases.append((s, m))


@pytest.mark.parametrize("smile,mode", cases)
def test_small_test_polyatomic_geometry(smile, mode):
    pgs = PolyatomicGeometrySMILE(smile=smile, target_dimension=3, mode=mode)
    pgs = pgs.smiles_to_geom_complex()
    if mode == "abstract":
        assert isinstance(pgs, AbstractComplex)
        assert check_abstract_complex_test(pgs)
    elif mode == "force-field":
        assert isinstance(pgs, ForceComplex)
        assert check_force_complex_test(pgs)
    elif mode == "quantum":
        assert isinstance(pgs, QuantumComplex)
        assert check_quantum_complex_test(pgs)
    elif mode == "quantum-waves":
        assert isinstance(pgs, QuantumWavesComplex)
        assert check_quantum_waves_complex_test(pgs)
    else:
        raise Exception("INVALID + UNSUPPORTED")
