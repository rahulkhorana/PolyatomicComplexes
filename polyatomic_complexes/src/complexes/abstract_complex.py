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

# topological features
import gudhi as gd
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from toponetx import CombinatorialComplex
from sklearn.preprocessing import StandardScaler


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

    def get_spectral_k_chains(self) -> dict:
        all_chains = {}
        cc = self.get_complex("rank_order")["molecule"][0]
        df_k_chains = self.__parse_k_chains_from_complex(cc)
        for i in range(1, 5):
            try:
                chain_i = self.__compute_combined_k_chain_features_fixed(
                    cc, df_k_chains, i
                )
                all_chains[f"chain_{i}"] = chain_i
            except:
                all_chains[f"chain_{i}"] = np.zeros(20)
        return all_chains

    def get_raw_k_chains(self) -> dict:
        all_chains = {}
        cc = self.get_complex("rank_order")["molecule"][0]
        df_k_chains = self.__parse_k_chains_from_complex(cc)
        for i in range(0, 5):
            raw_i_chain, _ = self.__compute_optimized_k_chain(cc, df_k_chains, i)
            all_chains[f"chain_{i}"] = raw_i_chain
        return all_chains

    def k_chains_formal_sum(self) -> dict:
        k_chain_symb = {}
        cc = self.get_complex("rank_order")["molecule"][0]
        for i in range(0, 5):
            str_repn_chain_i = self.__get_k_chain_formal_sum_clean(cc, i)
            k_chain_symb[f"symbolic_chain_{i}"] = str_repn_chain_i
        return k_chain_symb

    def get_df_k_chains(self) -> pd.DataFrame:
        cc = self.get_complex("rank_order")["molecule"][0]
        df_k_chains = self.__parse_k_chains_from_complex(cc)
        return df_k_chains

    def __compute_persistent_homology_safe(self, cc, k, max_dim=2, max_filtration=10):
        assert isinstance(cc, CombinatorialComplex)
        assert isinstance(k, int)
        st = gd.SimplexTree()
        for simplex in cc.skeleton(k):
            simplex_set = frozenset([s[0] for s in simplex])
            st.insert(simplex_set, filtration=1.0)
        st.expansion(max_dim)
        st.prune_above_filtration(max_filtration)
        persistence = st.persistence()
        lifetimes = []
        for interval in persistence:
            birth, death = interval[1]
            if death != float("inf"):
                lifetimes.append(death - birth)
        persistence_vector = np.zeros(5)
        lifetimes = sorted(lifetimes, reverse=True)[:5]
        for i, value in enumerate(lifetimes):
            persistence_vector[i] = value
        return persistence_vector

    def __parse_k_chains_from_complex(
        self, complex: CombinatorialComplex
    ) -> pd.DataFrame:
        assert isinstance(complex, CombinatorialComplex)
        structured_data = []
        for chain_idx, chain in enumerate(complex):
            for element in chain:
                if isinstance(element, tuple) and len(element) == 2:
                    identifier, data_tuple = element
                    if isinstance(data_tuple, tuple) and len(data_tuple) == 2:
                        particle_type, binary_data = data_tuple
                        to_array = np.frombuffer(binary_data, dtype=np.uint8)
                        structured_data.append(
                            {
                                "Chain Index": chain_idx,
                                "Identifier": identifier,
                                "Particle Type": particle_type,
                                "array": to_array,
                            }
                        )
        return pd.DataFrame(structured_data)

    def __compute_optimized_k_chain(
        self, cc: CombinatorialComplex, df_k_chains: pd.DataFrame, k: int
    ):
        assert isinstance(cc, CombinatorialComplex)
        assert isinstance(df_k_chains, pd.DataFrame)
        assert isinstance(k, int)
        k_cells = list(cc.skeleton(k))
        if not k_cells:
            return np.array([]), {}
        basis_mapping = {frozenset(k_cell): idx for idx, k_cell in enumerate(k_cells)}
        k_chain_vector = np.zeros(len(k_cells))
        for _, row in df_k_chains.iterrows():
            identifier = row["Identifier"]
            array_data = row["array"]
            sum_val = np.sum(array_data)
            mean_val = np.mean(array_data)
            std_val = np.std(array_data)
            std_val = std_val if std_val > 1e-6 else 1e-6
            coefficient = np.log1p(sum_val) * ((sum_val - mean_val) / std_val)
            for k_cell in k_cells:
                if identifier in [item[0] for item in k_cell]:
                    index = basis_mapping[frozenset(k_cell)]
                    k_chain_vector[index] += coefficient
        scaler = StandardScaler()
        k_chain_vector = scaler.fit_transform(k_chain_vector.reshape(-1, 1)).flatten()
        return [k_chain_vector, basis_mapping]

    def __get_k_chain_formal_sum_clean(self, cc, k, coefficient=1) -> str:
        assert isinstance(cc, CombinatorialComplex) and isinstance(k, int)
        k_cells = list(cc.skeleton(k))
        if not k_cells:
            return "0"
        formal_sum_terms = []
        for k_cell in k_cells:
            cell_identifiers = sorted([item[0] for item in k_cell])
            term = f"{coefficient} * {{{', '.join(cell_identifiers)}}}"
            formal_sum_terms.append(term)
        return " + ".join(formal_sum_terms)

    def __compute_combined_k_chain_features_fixed(
        self, cc: CombinatorialComplex, df_k_chains: pd.DataFrame, k: int
    ) -> np.ndarray:
        assert isinstance(cc, CombinatorialComplex)
        assert isinstance(df_k_chains, pd.DataFrame)
        k_chain_vector, _ = self.__compute_optimized_k_chain(cc, df_k_chains, k)
        laplacian_matrix = cc.laplacian_matrix(k)
        if sp.issparse(laplacian_matrix):
            laplacian_matrix = laplacian_matrix
        else:
            laplacian_matrix = sp.csr_matrix(laplacian_matrix)
        try:
            num_eigenvalues = min(5, laplacian_matrix.shape[0] - 1)
            eigenvalues = eigsh(
                laplacian_matrix, k=num_eigenvalues, return_eigenvectors=False
            )
        except:
            eigenvalues = np.zeros(5)
        eigenvalues = (
            StandardScaler().fit_transform(eigenvalues.reshape(-1, 1)).flatten()
        )
        try:
            persistence_intervals = self.__compute_persistent_homology_safe(cc, k)[0]
        except:
            persistence_intervals = np.zeros(5)
        combined_feature_vector = np.concatenate(
            [k_chain_vector, eigenvalues, persistence_intervals]
        )
        return combined_feature_vector

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
