import os
import sys
import dill
import json
import numpy as np
import networkx as nx
import jax.numpy as jnp
from pathlib import Path
from typing import List, Tuple, Optional
from collections import defaultdict

BASE_PATH = Path(__file__)
project_root = BASE_PATH.parent.parent.parent.parent.resolve()
src_dir = project_root
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# pc utils
from polyatomic_complexes.src.complexes.atomic_complex import AtomComplex
from polyatomic_complexes.src.complexes.polyatomic_complex_cls import PolyatomicComplex
from polyatomic_complexes.src.complexes.polyatomic_complex import PolyAtomComplex
from polyatomic_complexes.src.complexes.space_utils import (
    geometricAtomicComplex,
    geometricPolyatomicComplex,
)

from scipy.sparse import coo_matrix


class AbstractComplex(PolyatomicComplex):
    def __init__(self, smile, target_dimension, atoms, bonds, orientations=None):
        self.smile = smile
        self.dim = target_dimension
        self.atoms = atoms
        self.bnds = bonds
        self.lookup_fp = BASE_PATH.parent.parent.parent.__str__() + "/dataset/construct"
        assert "atom_lookup.pkl" in os.listdir(self.lookup_fp)
        assert "lookup_map.json" in os.listdir(self.lookup_fp)
        with open(self.lookup_fp + "/lookup_map.json", "rb") as f:
            self.lookup = json.load(f)
        assert orientations is None or isinstance(orientations, dict)
        self.orientations = orientations
        self.has_rank_order = False

    def get_default_ac(self, p, n, e):
        default = AtomComplex(
            protons=p,
            neutrons=n,
            electrons=e,
            cutoff=5,
            proton_dims=3,
            neutron_dims=3,
            electron_dims=3,
        )
        return default

    def get_default_orientations(self, atom=None):
        """
        NOTE: orientations can be a dictionary mapping indiviual atoms to particular orientation values.
        By default we just specify "No Orientation" encoded by a 0
        """
        if self.orientations is None:
            return 0
        else:
            return self.orientations[atom]

    def abstract_complex(self):
        """
        most abstract method to describe the complex (more for human intuition than fitting a model 2)
        aka {} with AtomComplex -> [AtomComplex, bond_type, orientation]
        Each AtomComplex is of dimension 3 by default
        """
        abstract_rep = defaultdict(list)
        for a1, a2, bnd in self.bnds:
            p1, n1, e1 = self.lookup[a1]
            ac1 = self.get_default_ac(p1, n1, e1)
            p2, n2, e2 = self.lookup[a2]
            ac2 = self.get_default_ac(p2, n2, e2)
            key_1 = tuple([ac1, a1])
            if self.orientations is None:
                value_1 = [[ac2, a2], bnd, 0]
            else:
                value_1 = [[ac2, a2], bnd, self.get_default_orientations(a1)]
            abstract_rep[key_1] = value_1
        self._abstract_complex = abstract_rep
        return abstract_rep

    def rank_order_complex(self):
        """
        electrons: lowest rank 1
        protons/neutrons: rank 2
        atoms: rank 3
        molecule: higher rank > 3
        # this uses assignment and ranks to specify more complex features
        """
        relations = []
        for a1, a2, bnd in self.bnds:
            p1, n1, e1 = self.lookup[a1]
            gatom_complex1 = geometricAtomicComplex(
                a1, p1, n1, e1
            ).generate_comb_complexes()
            p2, n2, e2 = self.lookup[a2]
            gatom_complex2 = geometricAtomicComplex(
                a2, p2, n2, e2
            ).generate_comb_complexes()
            rel = (gatom_complex1, bnd, gatom_complex2)
            relations.append(rel)
        self._relations = relations
        rank_order_complex = geometricPolyatomicComplex(
            relations
        ).generate_geom_poly_complex()
        self._rank_order_complex = rank_order_complex
        self.has_rank_order = True
        return rank_order_complex

    def atomic_topology(self):
        """
        provides ranks and features of the entire molecule
        ranks: skeleta for every dimension aka -> [(dim, skeleton)...]
        features: topological features computed for complex
        """
        if not self.has_rank_order:
            self.rank_order_complex()
        molecule, features = self._rank_order_complex["molecule"]
        ranks = [(rk, molecule.skeleton(rk)) for rk in range(molecule.dim + 1)]
        self._ranks = ranks
        return ranks, features

    def atomic_structure(self):
        """
        provides the lower dimensional inputs (geometricAtomicComplexes) fed into the polyatomic complex.
        """
        if not self.has_rank_order:
            self.rank_order_complex()
        return self._relations

    def bonds(self):
        """
        describes all bonds for standard abstract_complex
        """
        return self.bnds

    def get_atomic_structure(self):
        """
        provides the lower dimensional inputs (geometricAtomicComplexes) fed into the polyatomic complex.
        """
        return self.atomic_structure()

    def get_atomic_topology(self):
        """
        provides ranks and features of the entire molecule
        ranks: skeleta for every dimension aka -> [(dim, skeleton)...]
        features: topological features computed for complex
        """
        return self.atomic_topology()

    def get_bonds(self):
        """
        getter method for bonds
        """
        return self.bonds()

    def get_complex(self, kind: str):
        """
        getter method for complex
        args: 'abstract_complex' || 'rank_order'
        abstract_complex: for high level human intuition is a dict: AtomComplex -> [AtomComplex, bond_type, orientation]
        rank_order: for machine learning models may require further processing (to tensors) and ideally should use other features;
        example features to use:
        - get_laplacians_and_3D
        - get_electrostatics
        - get_forces
        - get_pc_matrix
        """
        if kind == "abstract_complex":
            return self.abstract_complex()
        if kind == "rank_order":
            return self.rank_order_complex()
        else:
            raise Exception("Unsupported Option")

    def get_pc_matrix(self, mode: str):
        """
        input: mode: fast, general, stacked
        getter method for standard PolyAtomComplex matrix
        returns: PolyAtomComplex -> fast complex
        """
        pac = PolyAtomComplex(self.atoms)
        if mode == "fast":
            pac = pac.fast_build_complex()
        elif mode == "general":
            pac = pac.general_build_complex()
        elif mode == "stacked":
            pac = pac.fast_stacked_complex()
        else:
            raise Exception(
                "Invalid kind must be from: 'fast' || 'general' || 'stacked' "
            )
        return pac[0]

    def get_laplacians(self):
        """
        Uses the Geometric Complex to provide all laplacians at various resolutions:
        returns:
        - defaultdict(list)
        - "molecule_laplacians" -> laplacians at molecular resolution
        - "electronic_laplacians" -> laplacians at electron resolution
        - "nucleus_laplacians" -> laplacians at nucleus resolution
        """
        out_laplacians = self.general_get("laplacians")
        return out_laplacians

    def get_incidence(self):
        """
        Uses the Geometric Complex to provide all incidence matricies at various resolutions:
        returns:
        - defaultdict(list)
        - "molecule_incidence" -> incidence at molecular resolution
        - "electronic_incidence" -> incidence at electron resolution
        - "nucleus_incidence" -> incidence at nucleus resolution
        """
        out_incidence = self.general_get("incidence")
        return out_incidence

    def get_adjacencies(self):
        """
        Uses the Geometric Complex to provide all adjacency matricies at various resolutions:
        returns:
        - defaultdict(list)
        - "molecule_adjacencies" -> adjacencies at molecular resolution
        - "electronic_adjacencies" -> adjacencies at electron resolution
        - "nucleus_adjacencies" -> adjacencies at nucleus resolution
        """
        out_adj = self.general_get("adjacencies")
        return out_adj

    def get_coadjacencies(self):
        """
        Uses the Geometric Complex to provide all coadjacency matricies at various resolutions:
        returns:
        - defaultdict(list)
        - "molecule_co_adjacencies" -> co_adjacencies at molecular resolution
        - "electronic_co_adjacencies" -> co_adjacencies at electron resolution
        - "nucleus_co_adjacencies" -> co_adjacencies at nucleus resolution
        """
        out_coadj = self.general_get("co_adjacencies")
        return out_coadj

    def get_skeleta(self):
        """
        Uses the Geometric Complex to provide the skeleton:
        returns:
        - defaultdict(list)
        - "molecule_skeleta" -> skeleta at molecular resolution
        - "electronic_skeleta" -> skeleta at electron resolution
        - "nucleus_skeleta" -> skeleta at nucleus resolution
        """
        out_skeleta = self.general_get("skeleta")
        return out_skeleta

    def get_all_cell_coadj(self):
        """
        Uses the Geometric Complex to provide the all cell coadjacency matrix:
        returns:
        - defaultdict(list)
        - "molecule_all_cell_coadj" -> all_cell_coadj at molecular resolution
        - "electronic_all_cell_coadj" -> all_cell_coadj at electron resolution
        - "nucleus_all_cell_coadj" -> all_cell_coadj at nucleus resolution
        """
        out = self.general_get("all_cell_coadj")
        return out

    def get_dirac(self):
        """
        Uses the Geometric Complex to provide the dirac matrix:
        returns:
        - defaultdict(list)
        - "molecule_dirac -> dirac at molecular resolution
        - "electronic_dirac" -> dirac at electron resolution
        - "nucleus_dirac" -> dirac at nucleus resolution
        """
        out = self.general_get("dirac")
        return out

    def get_persistence(self):
        """
        Uses the Geometric Complex to provide the persistence:
        returns:
        - defaultdict(list)
        - "molecule_persistence -> persistence at molecular resolution
        - "electronic_persistence" -> persistence at electron resolution
        - "nucleus_persistence" -> persistence at nucleus resolution
        """
        out = self.general_get("persistence")
        return out

    def get_betti_numbers(self):
        """
        Uses the Geometric Complex to provide the betti numbers:
        returns:
        - defaultdict(list)
        - "molecule_betti_numbers -> betti_numbers at molecular resolution
        - "electronic_betti_numbers" -> betti_numbers at electron resolution
        - "nucleus_betti_numbers" -> betti_numbers at nucleus resolution
        """
        out = self.general_get("betti_numbers")
        return out

    def general_get(self, column_name):
        out_features = defaultdict(list)
        roc = self.rank_order_complex()
        _molecule = roc["molecule"]
        _electronic = roc["electronic_structure"]
        _nucleus = roc["nuclear_structure"]
        molecule_features = _molecule[1]
        electronic_features = _electronic[1]
        nucleus_features = _nucleus[1]
        assert isinstance(molecule_features, defaultdict)
        assert isinstance(electronic_features, defaultdict)
        assert isinstance(nucleus_features, defaultdict)
        mf = molecule_features[f"{column_name}"]
        if isinstance(mf, np.ndarray) or isinstance(mf, coo_matrix):
            if isinstance(mf, np.ndarray):
                mf = mf.tolist()
            if isinstance(mf, coo_matrix):
                mf = mf.toarray()
        out_features[f"molecule_{column_name}"].append(mf)
        ef = electronic_features[f"{column_name}"]
        if isinstance(ef, np.ndarray) or isinstance(ef, coo_matrix):
            if isinstance(ef, np.ndarray):
                ef = ef.tolist()
            if isinstance(ef, coo_matrix):
                ef = ef.toarray()
        out_features[f"molecule_{column_name}"].append(ef)
        nf = nucleus_features[f"{column_name}"]
        if isinstance(nf, np.ndarray) or isinstance(nf, coo_matrix):
            if isinstance(nf, np.ndarray):
                nf = nf.tolist()
            if isinstance(nf, coo_matrix):
                nf = nf.toarray()
        out_features[f"molecule_{column_name}"].append(nf)
        return out_features

    def polyatomcomplex(self):
        return self.get_complex("abstract_complex")

    def electrostatics(self):
        raise NotImplementedError(
            "This is not defined behavior for an Abstract Complex!"
        )

    def forces(self):
        raise NotImplementedError(
            "This is not defined behavior for an Abstract Complex!"
        )

    def get_electrostatics(self):
        raise NotImplementedError(
            "This is not defined behavior for an Abstract Complex!"
        )

    def get_forces(self):
        raise NotImplementedError(
            "This is not defined behavior for an Abstract Complex!"
        )

    def wavefunctions(self):
        raise NotImplementedError(
            "This is not defined behavior for an Abstract Complex!"
        )
