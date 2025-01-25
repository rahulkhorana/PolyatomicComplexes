import os
import json
import torch
import gudhi as gd
import numpy as np
from typing import List
from collections import defaultdict
from toponetx import CombinatorialComplex
from pathlib import Path
import sys
import os


BASE_PATH = Path(__file__)
project_root = BASE_PATH.parent.parent.parent.parent.resolve()
src_dir = project_root
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from polyatomic_complexes.src.complexes.building_blocks import Neutron, Proton, Electron

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from scipy.sparse import coo_matrix
import psutil
import math
from functools import partial


def nice_print(arg):
    print("*" * 10)
    print(f"Done: {arg}")
    print("*" * 10)


BASE_PATH = Path(__file__)


class geometricPolyatomicComplex:
    def __init__(self, higher_relations):
        self.relations = higher_relations
        self.fp = (
            BASE_PATH.parent.parent.parent.__str__()
            + "/dataset/construct/atomic_mass.json"
        )
        with open(self.fp, "r") as file:
            self.amu = json.load(file)
        self._memory_limit_gb = 1.0

    def _calculate_num_workers(self, memory_limit_gb=1.0):
        num_cores = psutil.cpu_count(logical=True)
        available_memory = psutil.virtual_memory().available
        memory_limit = min(available_memory, memory_limit_gb * (1024**3))
        memory_per_worker = memory_limit_gb * (1024**3)
        max_workers_by_memory = int(memory_limit // memory_per_worker)
        max_workers = max(1, min(num_cores, max_workers_by_memory))
        return max_workers

    def _calculate_chunk_size(self, vertex_array, filtrations, memory_limit_gb=1.0):
        available_memory = psutil.virtual_memory().available
        memory_limit = min(available_memory, memory_limit_gb * (1024**3))
        num_vertices = vertex_array.shape[0]
        memory_per_simplex = (
            num_vertices * vertex_array.itemsize
        ) + filtrations.itemsize
        max_chunk_size = int(memory_limit // memory_per_simplex)
        return max(100, min(max_chunk_size, vertex_array.shape[1] // 10))

    def get_nuclei_and_efield(self, cc1):
        nuc = cc1["nucleus"][0]
        elec = cc1["electrons"][0]
        return nuc, elec

    def higher_order_glue(self, tgt, cmplxs, adj):
        assert (
            isinstance(tgt, CombinatorialComplex)
            and isinstance(cmplxs, list)
            and isinstance(adj, defaultdict)
        )
        for cc in cmplxs:
            for rank in range(cc.dim + 1):
                for cell in cc.skeleton(rank=rank):
                    tgt.add_cell(cell, rank=rank)

        for (cc_a, cc_b), weight in adj.items():
            if not isinstance(cc_a, CombinatorialComplex) or not isinstance(
                cc_b, CombinatorialComplex
            ):
                raise TypeError(
                    "Keys in adjacency map must be CombinatorialComplex objects"
                )
            if not isinstance(weight, (int, float)):
                raise TypeError("Weights in adjacency map must be numeric")
            for cell_a in cc_a.skeleton(rank=cc_a.dim):
                for cell_b in cc_b.skeleton(rank=cc_b.dim):
                    higher_dim_cell = list(cell_a) + list(cell_b)
                    if not tgt.__contains__(higher_dim_cell):
                        tgt.add_cell(higher_dim_cell, rank=cc_a.dim + 1, weight=weight)
        return tgt

    @staticmethod
    def insert_chunk(start_idx, end_idx, vertex_array, filtrations):
        tree = gd.SimplexTree()
        tree.insert_batch(
            vertex_array[:, start_idx:end_idx], filtrations[start_idx:end_idx]
        )
        return tree

    def process_with_partial(self, chunk_index, chunk_size, vertex_array, filtrations):
        start = chunk_index * chunk_size
        end = min((chunk_index + 1) * chunk_size, vertex_array.shape[1])
        return self.insert_chunk(start, end, vertex_array, filtrations)

    def get_persistence_features(self, CC: CombinatorialComplex, feat: defaultdict):
        def process_rank(rank):
            return [(tuple(cell), rank) for cell in CC.skeleton(rank)]

        with ThreadPoolExecutor() as executor:
            simplex_list = [
                item
                for result in executor.map(process_rank, range(CC.dim + 1))
                for item in result
            ]
        max_vertices = max(len(simplex) for simplex, _ in simplex_list)
        num_simplices = len(simplex_list)
        row, col, data = [], [], []
        filtrations = np.zeros(num_simplices, dtype=float)
        for idx, (simplex, rank) in enumerate(simplex_list):
            for j, vertex in enumerate(simplex):
                row.append(j)
                col.append(idx)
                data.append(hash(vertex) % 1000)
            filtrations[idx] = rank
        vertex_array = coo_matrix(
            (data, (row, col)), shape=(max_vertices, num_simplices)
        ).toarray()
        chunk_size = 100
        calc_chunk = self._calculate_chunk_size(
            vertex_array, filtrations, self._memory_limit_gb
        )
        chunk_size = min(chunk_size, math.floor(calc_chunk / 50) * 50)
        num_chunks = (vertex_array.shape[1] + chunk_size - 1) // chunk_size
        num_workers = self._calculate_num_workers(self._memory_limit_gb)
        process_partial = partial(
            self.process_with_partial,
            chunk_size=chunk_size,
            vertex_array=vertex_array,
            filtrations=filtrations,
        )
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            trees = list(executor.map(process_partial, range(num_chunks - 1)))
        simplex_tree = gd.SimplexTree()
        for tree in trees:
            for simplex, filtration in tree.get_simplices():
                simplex_tree.insert(simplex, filtration)
        feat["persistence"] = simplex_tree.persistence()
        feat["betti_numbers"] = simplex_tree.betti_numbers()
        return feat

    def get_comb_features(
        self, CC: CombinatorialComplex, mode: str = ""
    ) -> defaultdict:
        feat = defaultdict(list)
        incidence_dict = CC.incidence_dict
        feat["incidence"] = incidence_dict
        laplacians = []
        for i in range(1, CC.dim + 1):
            lap = CC.laplacian_matrix(i)
            laplacians.append((f"{i}", lap))
        feat["laplacians"] = laplacians
        adjacencies = []
        for i in range(0, CC.dim + 1):
            adj = CC.adjacency_matrix(rank=i, via_rank=i + 1).data
            adjacencies.append((f"({i},{i+1})", adj))
        feat["adjacencies"] = adjacencies
        co_adjacencies = []
        for i in range(1, CC.dim + 1):
            coadj = CC.coadjacency_matrix(rank=i, via_rank=i - 1).data
            co_adjacencies.append((f"({i},{i-1})", coadj))
        feat["co_adjacencies"] = co_adjacencies
        skeleta = []
        for i in range(0, CC.dim):
            sk = CC.skeleton(i)
            skeleta.append((f"{i}", sk))
        feat["skeleta"] = skeleta
        all_cell_coadj = CC.all_cell_to_node_coadjacency_matrix().data
        feat["all_cell_coadj"] = all_cell_coadj
        try:
            dirac = CC.dirac_operator_matrix()
            feat["dirac"] = dirac
        except:
            feat["dirac"] = None
        if mode == "get_persistence":
            feat = self.get_persistence_features(CC, feat)
        return feat

    def generate_geom_poly_complex(self) -> defaultdict:
        polyatomic_structure = defaultdict()
        molecule = CombinatorialComplex()
        nuc_adj, elec_adj = defaultdict(), defaultdict()
        nuclear_struct, electronic_struct = [], []
        for c1, bnd, c2 in self.relations:
            assert isinstance(c1, defaultdict) and isinstance(c2, defaultdict)
            assert (
                "electrons" in c1
                and "protons" in c1
                and "neutrons" in c1
                and "nucleus" in c1
                and "atom" in c1
            )
            assert (
                "electrons" in c2
                and "protons" in c2
                and "neutrons" in c2
                and "nucleus" in c2
                and "atom" in c2
            )
            a1, a2 = c1["atom"], c2["atom"]
            nc1, ef1 = self.get_nuclei_and_efield(c1)
            nc2, ef2 = self.get_nuclei_and_efield(c2)
            nuc_adj[(nc1, nc2)] = self.amu[a1] + self.amu[a2]
            elec_adj[(ef1, ef2)] = bnd[1]
            nuclear_struct.extend([nc1, nc2])
            electronic_struct.extend([ef1, ef2])
        molecule = self.higher_order_glue(molecule, nuclear_struct, nuc_adj)
        molecule = self.higher_order_glue(molecule, electronic_struct, elec_adj)
        polyatomic_structure["molecule"] = [
            molecule,
            self.get_comb_features(molecule, "molecule"),
        ]
        nuclear_structure = CombinatorialComplex()
        nuclear_structure = self.higher_order_glue(
            nuclear_structure, nuclear_struct, nuc_adj
        )
        polyatomic_structure["nuclear_structure"] = [
            nuclear_structure,
            self.get_comb_features(nuclear_structure, "nucleus"),
        ]
        electronic = CombinatorialComplex()
        electronic = self.higher_order_glue(electronic, electronic_struct, elec_adj)
        polyatomic_structure["electronic_structure"] = [
            electronic,
            self.get_comb_features(electronic, "electrons"),
        ]
        return polyatomic_structure


class geometricAtomicComplex:
    def __init__(self, atom, protons, neutrons, electrons):
        assert (
            isinstance(atom, str)
            and isinstance(protons, int)
            and isinstance(neutrons, int)
            and isinstance(electrons, int)
        )
        self.atom = atom
        self.protons = protons
        self.neutrons = neutrons
        self.electrons = electrons
        self.fp = (
            BASE_PATH.parent.parent.parent.__str__()
            + "/dataset/construct/atomic_mass.json"
        )
        with open(self.fp, "r") as file:
            self.amu = json.load(file)

        self._memory_limit_gb = 1.0

    def hashable(self, kind, obj):
        if kind == "proton" or kind == "neutron":
            assert isinstance(obj, tuple)
            assert isinstance(obj[0], np.ndarray)
            _bytes = obj[0].tobytes()
            return tuple([kind, _bytes])
        elif kind == "electron":
            assert isinstance(obj, tuple)
            assert isinstance(obj[0], np.ndarray)
            assert isinstance(obj[1], np.ndarray)
            _bytes_ee = obj[0].tobytes()
            _bytes_w = obj[1].tobytes()
            return tuple([kind, _bytes_ee, _bytes_w])

    def get_comb_features(self, CC: CombinatorialComplex):
        feat = defaultdict()
        incidence_dict = CC.incidence_dict
        feat["incidence"] = incidence_dict
        laplacians = []
        for i in range(1, CC.dim + 1):
            lap = CC.laplacian_matrix(i)
            laplacians.append((f"{i}", lap))
        feat["laplacians"] = laplacians
        adjacencies = []
        for i in range(0, CC.dim + 1):
            adj = CC.adjacency_matrix(rank=i, via_rank=i + 1).data
            adjacencies.append((f"({i},{i+1})", adj))
        feat["adjacencies"] = adjacencies
        co_adjacencies = []
        for i in range(1, CC.dim + 1):
            coadj = CC.coadjacency_matrix(rank=i, via_rank=i - 1).data
            co_adjacencies.append((f"({i},{i-1})", coadj))
        feat["co_adjacencies"] = co_adjacencies
        skeleta = []
        for i in range(0, CC.dim):
            sk = CC.skeleton(i)
            skeleta.append((f"{i}", sk))
        feat["skeleta"] = skeleta
        all_cell_coadj = CC.all_cell_to_node_coadjacency_matrix().data
        feat["all_cell_coadj"] = all_cell_coadj
        try:
            dirac = CC.dirac_operator_matrix()
            feat["dirac"] = dirac
        except:
            feat["dirac"] = None
        feat = self.get_persistence_features(CC, feat)
        return feat

    def make_nucleus(self, c1: CombinatorialComplex, c2: CombinatorialComplex):
        adj_map = {(c1, c2): self.amu[self.atom]}
        nucleus = CombinatorialComplex()
        for cc in [c1, c2]:
            for rank in range(cc.dim + 1):
                for cell in cc.skeleton(rank=rank):
                    nucleus.add_cell(cell, rank=rank)
        for (cc_a, cc_b), weight in adj_map.items():
            for cell_a in cc_a.skeleton(rank=cc_a.dim):
                for cell_b in cc_b.skeleton(rank=cc_b.dim):
                    higher_dim_cell = list(cell_a) + list(cell_b)
                    nucleus.add_cell(higher_dim_cell, rank=cc_a.dim + 1, weight=weight)
        return nucleus

    def _calculate_num_workers(self, memory_limit_gb=1.0):
        num_cores = psutil.cpu_count(logical=True)
        available_memory = psutil.virtual_memory().available
        memory_limit = min(available_memory, memory_limit_gb * (1024**3))
        memory_per_worker = memory_limit_gb * (1024**3)
        max_workers_by_memory = int(memory_limit // memory_per_worker)
        max_workers = max(1, min(num_cores, max_workers_by_memory))
        return max_workers

    def _calculate_chunk_size(self, vertex_array, filtrations, memory_limit_gb=1.0):
        available_memory = psutil.virtual_memory().available
        memory_limit = min(available_memory, memory_limit_gb * (1024**3))
        num_vertices = vertex_array.shape[0]
        memory_per_simplex = (
            num_vertices * vertex_array.itemsize
        ) + filtrations.itemsize
        max_chunk_size = int(memory_limit // memory_per_simplex)
        return max(100, min(max_chunk_size, vertex_array.shape[1] // 10))

    @staticmethod
    def insert_chunk(start_idx, end_idx, vertex_array, filtrations):
        tree = gd.SimplexTree()
        tree.insert_batch(
            vertex_array[:, start_idx:end_idx], filtrations[start_idx:end_idx]
        )
        return tree

    def process_with_partial(self, chunk_index, chunk_size, vertex_array, filtrations):
        start = chunk_index * chunk_size
        end = min((chunk_index + 1) * chunk_size, vertex_array.shape[1])
        return self.insert_chunk(start, end, vertex_array, filtrations)

    def get_persistence_features(self, CC: CombinatorialComplex, feat: defaultdict):
        def process_rank(rank):
            return [(tuple(cell), rank) for cell in CC.skeleton(rank)]

        with ThreadPoolExecutor() as executor:
            simplex_list = [
                item
                for result in executor.map(process_rank, range(CC.dim + 1))
                for item in result
            ]
        max_vertices = max(len(simplex) for simplex, _ in simplex_list)
        num_simplices = len(simplex_list)
        row, col, data = [], [], []
        filtrations = np.zeros(num_simplices, dtype=float)
        for idx, (simplex, rank) in enumerate(simplex_list):
            for j, vertex in enumerate(simplex):
                row.append(j)
                col.append(idx)
                data.append(hash(vertex) % 1000)
            filtrations[idx] = rank
        vertex_array = coo_matrix(
            (data, (row, col)), shape=(max_vertices, num_simplices)
        ).toarray()
        chunk_size = 100
        calc_chunk = self._calculate_chunk_size(
            vertex_array, filtrations, self._memory_limit_gb
        )
        chunk_size = min(chunk_size, math.floor(calc_chunk / 50) * 50)
        num_chunks = (vertex_array.shape[1] + chunk_size - 1) // chunk_size
        num_workers = self._calculate_num_workers(self._memory_limit_gb)
        process_partial = partial(
            self.process_with_partial,
            chunk_size=chunk_size,
            vertex_array=vertex_array,
            filtrations=filtrations,
        )
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            trees = list(executor.map(process_partial, range(num_chunks - 1)))
        simplex_tree = gd.SimplexTree()
        for tree in trees:
            for simplex, filtration in tree.get_simplices():
                simplex_tree.insert(simplex, filtration)
        feat["persistence"] = simplex_tree.persistence()
        feat["betti_numbers"] = simplex_tree.betti_numbers()
        return feat

    def generate_comb_complexes(self) -> defaultdict:
        atomic_structure = defaultdict(list)
        electron_complex = CombinatorialComplex()
        electrons_id = [
            (
                f"E_{i}",
                self.hashable("electron", Electron(dim=3, num_pts=3).build_electron()),
            )
            for i in range(self.electrons)
        ]
        electron_complex.add_cell(electrons_id, rank=1)
        neutron_complex = CombinatorialComplex()
        neutrons_id = [
            (
                f"N_{i}",
                self.hashable("neutron", Neutron(dim=3, num_pts=3).build_neutron()),
            )
            for i in range(self.neutrons)
        ]
        neutron_complex.add_cell(neutrons_id, rank=2)
        proton_complex = CombinatorialComplex()
        protons_id = [
            (f"P_{i}", self.hashable("proton", Proton(dim=3, num_pts=3).build_proton()))
            for i in range(self.protons)
        ]
        proton_complex.add_cell(protons_id, rank=2)
        atomic_structure["electrons"] = [
            electron_complex,
            self.get_comb_features(electron_complex),
        ]
        atomic_structure["neutrons"] = [
            neutron_complex,
            self.get_comb_features(neutron_complex),
        ]
        atomic_structure["protons"] = [
            proton_complex,
            self.get_comb_features(proton_complex),
        ]
        nucleus = self.make_nucleus(proton_complex, neutron_complex)
        atomic_structure["nucleus"] = [nucleus, self.get_comb_features(nucleus)]
        atomic_structure["atom"] = self.atom
        return atomic_structure
