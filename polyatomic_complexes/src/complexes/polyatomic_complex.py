import os
import sys
import dill
import json
import numpy as np
import networkx as nx
import jax.numpy as jnp
from typing import List, Tuple, Optional
from pathlib import Path


BASE_PATH = Path(__file__)
project_root = BASE_PATH.parent.parent.parent.parent.resolve()
src_dir = project_root
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from polyatomic_complexes.src.complexes.core_utils import GluingMap, ElectronField
from polyatomic_complexes.src.complexes.building_blocks import Electron


class PolyAtomComplex:

    def __init__(
        self,
        atom_list,
        using_radial=False,
        using_force=False,
        update_forces=None,
        update_radial=None,
    ):
        assert isinstance(atom_list, list) and len(atom_list) > 0
        assert isinstance(using_radial, bool)
        assert isinstance(using_force, bool)
        self.atoms = atom_list
        self.using_radial = using_radial
        self.using_force = using_force
        self.cwd = BASE_PATH
        self.datapath = BASE_PATH.parent.parent.parent.__str__() + "/dataset/construct"
        assert "atom_lookup.pkl" in os.listdir(self.datapath)
        assert "lookup_map.json" in os.listdir(self.datapath)
        self.reference = self.datapath + "/lookup_map.json"
        with open(self.reference, "rb") as f:
            self.lookup_map = json.load(f)
        assert isinstance(self.lookup_map, dict)
        assert isinstance(using_radial, bool)
        assert isinstance(using_force, bool)
        if using_radial:
            assert hasattr(update_radial, "__call__")
            self.update_radial = update_radial
        if using_force:
            assert hasattr(update_forces, "__call__")
            self.update_forces = update_forces

    def gen_elec_info(self, k) -> List:
        self.AE = []
        self.d_e = 0
        if isinstance(self.d_e, int):
            electrons = Electron(self.d_e, k).build_electron()
            self.AE = list(electrons)
        return self.AE

    def fast_build_complex(self) -> Tuple:
        lookup = self.datapath + "/atom_lookup.pkl"
        with open(lookup, "rb") as f:
            lookup = dill.load(f)
        C = []
        for a in self.atoms:
            acomplex = lookup[a]
            C += list(acomplex[0].ravel())
        C = np.asarray(C)
        return tuple([C, [], [], []])

    def general_build_complex(self):
        lookup = self.datapath + "/atom_lookup.pkl"
        with open(lookup, "rb") as f:
            lookup = dill.load(f)
        eleminfo = self.lookup_map
        A = []
        for i, a in enumerate(self.atoms):
            acomplex = lookup[a]
            inf = eleminfo[a]
            info = [inf[2], acomplex[0], acomplex[1], acomplex[2], acomplex[3]]
            A.append(info)
        C = np.asarray([])
        self.E = ElectronField()
        self.F = np.array([])
        self.DE = np.array([])

        def disjont_union(
            gluing_map: np.ndarray, skeleton: np.ndarray, particle: np.ndarray
        ) -> np.ndarray:
            skeleton = np.concatenate([skeleton.ravel(), gluing_map.ravel()])
            K = np.concatenate([skeleton, particle.ravel()])
            K = np.unique(K)
            return K

        def electron_union(EF_1: tuple, EF_2: tuple) -> None:
            ee0_1 = EF_1[0]
            wv_1 = EF_1[1]
            assert isinstance(ee0_1, np.ndarray)
            assert isinstance(wv_1, np.ndarray)
            e_field = np.union1d(ee0_1, [])
            wves = np.concatenate([wv_1.ravel()])
            e_field, waves = e_field.ravel(), wves.ravel()
            self.E.add_electrons(e_field, waves)
            return None

        def dir_sum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            dsum = np.zeros(np.add(a.shape, b.shape))
            dsum[: a.shape[0], : a.shape[1]] = a
            dsum[a.shape[0] :, a.shape[1] :] = b
            return dsum

        while A:
            data = A.pop(0)
            num_elec, k, _, df, de = data
            if len(C) == 0:
                C = k
            phi = GluingMap(C, k, target=C).construct_map()
            map = np.asarray(list(phi.keys()), dtype=jnp.float32)
            map = np.unique(map)
            C = disjont_union(map, C, k)
            ae = self.gen_elec_info(num_elec)
            electron_union(ae, [])
            if self.using_force:
                self.F = dir_sum(self.F, df)
                self.update_forces(self.F)
            if self.using_radial:
                self.DE = dir_sum(self.DE, de)
                self.update_radial(self.DE)

        return tuple([C, self.E.get_efield(), self.F, self.DE])

    def custom_reshape(self, ac):
        assert isinstance(ac, np.ndarray) and len(ac.shape) == 1
        length = ac.shape[0]
        nearest = pow(round(np.sqrt(length)) + 1, 2)
        diff = abs(nearest - length)
        out = np.asarray([i for i in ac] + [0 for _ in range(diff)])
        r = round(np.sqrt(len(out)))
        out = out.reshape(r, r)
        return out

    def take_nbrs(self, stack):
        return stack.nodes, stack.edges

    def fast_stacked_complex(self) -> Tuple:
        lookup = self.datapath + "/atom_lookup.pkl"
        with open(lookup, "rb") as f:
            lookup = dill.load(f)
        C = []
        for a in self.atoms:
            acomplex = lookup[a]
            data = acomplex[0].ravel()
            assert isinstance(data, np.ndarray)
            if np.sqrt(len(data)) % 1 == 0:
                r = round(np.sqrt(len(data)))
                data = data.reshape(r, r)
                C.append(nx.Graph(data))
            else:
                res = self.custom_reshape(data)
                C.append(nx.Graph(res))
        assert len(C) > 0
        combine = C.pop(0)
        while C:
            next = C.pop(0)
            combine = nx.disjoint_union(combine, next)
        nde, edge = self.take_nbrs(combine)
        assert (
            isinstance(combine, nx.Graph)
            and len(combine) > 0
            and len(nde) > 0
            and len(edge) > 0
        )
        C = [combine]
        return tuple([C, [], [], []])


def sanity_test(atom_list, kind):
    pac = PolyAtomComplex(atom_list)
    if kind == "general":
        try:
            pac.general_build_complex()
            print("success ✅")
        except Exception:
            print("failed ❌")
    elif kind == "fast_stacked":
        try:
            pac.fast_stacked_complex()
            print("success ✅")
        except Exception:
            print("failed ❌")
    else:
        try:
            pac.fast_build_complex()
            print("success ✅")
        except Exception:
            print("failed ❌")


if __name__ == "__main__":
    sanity_test(["H", "H", "O"], "")
    sanity_test(["H", "H", "O"], "fast_stacked")
    sanity_test(["C", "H", "H", "H"], "")
    sanity_test(
        [
            "Np",
            "U",
            "P",
            "P",
            "P",
            "P",
            "H",
            "H",
            "H",
            "H",
            "C",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "C",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
        ],
        "",
    )
    sanity_test(
        [
            "Np",
            "U",
            "P",
            "P",
            "P",
            "P",
            "H",
            "H",
            "H",
            "H",
            "C",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "C",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
        ],
        "",
    )
    sanity_test(["H", "H", "O"], "general")
    sanity_test(["C", "H", "H", "H"], "general")
    sanity_test(
        [
            "Np",
            "U",
            "P",
            "P",
            "P",
            "P",
            "H",
            "H",
            "H",
            "H",
            "C",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "C",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
        ],
        "fast_stacked",
    )
    sanity_test(["H", "H", "O"], "general")
    sanity_test(["C", "H", "H", "H"], "general")
    sanity_test(
        [
            "Np",
            "U",
            "P",
            "P",
            "P",
            "P",
            "H",
            "H",
            "H",
            "H",
            "C",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "C",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
        ],
        "general",
    )
    sanity_test(
        [
            "Np",
            "U",
            "P",
            "P",
            "P",
            "P",
            "H",
            "H",
            "H",
            "H",
            "C",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "C",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
        ],
        "fast_stacked",
    )
