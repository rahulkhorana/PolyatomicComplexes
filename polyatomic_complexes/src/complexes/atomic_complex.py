import sys
import jax
import numpy as np
from typing import List, Tuple
from pathlib import Path

BASE_PATH = Path(__file__)
project_root = BASE_PATH.parent.parent.parent.parent.resolve()
src_dir = project_root
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from polyatomic_complexes.src.complexes.building_blocks import Electron, Proton, Neutron
from polyatomic_complexes.src.complexes.general_utils import GeneralComplexUtils


seed = np.random.randint(0, 10 * 3)
fm = np.float64(1e-15)
key = jax.random.PRNGKey(seed)


class AtomComplex:
    def __init__(
        self,
        protons,
        neutrons,
        electrons,
        cutoff,
        proton_dims,
        neutron_dims,
        electron_dims,
        force_field=None,
        radial_contrib=None,
    ):
        self.P = protons
        self.N = neutrons
        self.E = electrons
        assert isinstance(protons, int) and protons >= 0
        assert isinstance(neutrons, int) and neutrons >= 0
        assert isinstance(electrons, int) and electrons >= 0
        assert max(self.P, self.N) >= 1
        assert min([self.P, self.N, self.E]) >= 0
        assert isinstance(proton_dims, list) or isinstance(proton_dims, int)
        assert isinstance(neutron_dims, list) or isinstance(neutron_dims, int)
        assert isinstance(electron_dims, list) or isinstance(electron_dims, int)
        if isinstance(proton_dims, int):
            assert proton_dims > 0
        else:
            assert len(proton_dims) == 2 and min(proton_dims) >= 0
        if isinstance(neutron_dims, int):
            assert neutron_dims > 0
        else:
            assert len(neutron_dims) == 2 and min(neutron_dims) >= 0
        assert isinstance(electron_dims, int) and electron_dims >= 0
        assert isinstance(cutoff, int)
        self.d_p = proton_dims
        self.d_n = neutron_dims
        self.d_e = electron_dims
        self.cutoff = cutoff
        self.DF = force_field
        self.DE = radial_contrib

    def custom_pad(self, arr, amt):
        new = []
        for a in arr[0]:
            for c in a:
                new.append(c)
        new += [0 for _ in range(amt)]
        res = np.array(new)
        return res

    def build_protons(self) -> List:
        self.AP = []
        if isinstance(self.d_p, int):
            protons = Proton(self.d_p, self.P).build_proton()
            self.AP = list(protons)
        else:
            l, r = self.d_p
            for i in range(l, r + 1):
                protons = Proton(i, self.P).build_proton()
                protons = self.custom_pad(protons, r - i)
                self.AP += list(protons)
        return self.AP

    def build_neutrons(self) -> List:
        self.AN = []
        if isinstance(self.d_n, int):
            neutrons = Neutron(self.d_n, self.N).build_neutron()
            self.AN = list(neutrons)
        else:
            l, r = self.d_n
            for i in range(l, r + 1):
                neutrons = Neutron(i, self.N).build_neutron()
                neutrons = self.custom_pad(neutrons, r - i)
                self.AN += list(neutrons)
        return self.AN

    def build_electrons(self) -> List:
        self.AE = []
        if isinstance(self.d_e, int):
            electrons = Electron(self.d_e, self.E).build_electron()
            self.AE = list(electrons)
        else:
            l, r = self.d_e
            for i in range(l, r + 1):
                electrons = Electron(i, self.E).build_electron()
                electrons = self.custom_pad(electrons, r - i)
                self.AE += list(electrons)
        self.AE = [self.AE]
        return self.AE

    def fast_build_complex(self, using_distances=False, update_distances=None) -> Tuple:
        self.build_protons()
        self.build_neutrons()
        self.build_electrons()
        assert isinstance(using_distances, bool)
        P = []
        while self.AP:
            p = self.AP.pop()
            p = np.asarray(p, dtype=np.float32)
            P.append(p)

        N = []
        while self.AN:
            n = self.AN.pop()
            n = np.asarray(n, dtype=np.float32)
            N.append(n)

        E = []
        while self.AE:
            e, we = self.AE.pop()
            if using_distances and hasattr(update_distances, "__call__"):
                update_distances(self.DE, we, e)
            E.append(e)

        P = np.asarray(P)
        N = np.asarray(N)
        E = np.asarray(E)
        K = np.union1d(P, N)
        K = np.union1d(K, E)
        return tuple([K, self.AE, self.DF, self.DE])

    def general_build_complex(
        self, using_distances=False, update_distances=None
    ) -> Tuple:
        if isinstance(self.d_p, list) or isinstance(self.d_n, list):
            print("this will take too long - stopping early")
            raise AssertionError("invalid input")
        gcu = GeneralComplexUtils(self.cutoff)
        self.build_protons()
        self.build_neutrons()
        self.build_electrons()
        assert isinstance(using_distances, bool)
        K = np.asarray([])
        while self.AP and self.P > 0:
            p = self.AP.pop()
            p = np.asarray(p, dtype=np.float32)
            res = gcu.get_nsphere_and_sampled_boundary(
                radius=1 * fm, center=p[0], sample=10000, eps=1e-19
            )
            nsphere, boundary = res
            n_sphere = np.asarray(nsphere)
            gluing_map = gcu.nuclear_force_map(boundary, boundary, K)
            K = gcu.topological_disjoint_union(gluing_map, K, n_sphere)

        while self.AN and self.N > 0:
            n = self.AN.pop()
            n = np.asarray(n, dtype=np.float32)
            res = gcu.get_nsphere_and_sampled_boundary(
                radius=0.8 * fm, center=n[0], sample=10000, eps=1e-19
            )
            nsphere, boundary = res
            n_sphere = np.asarray(nsphere)
            gluing_map = gcu.nuclear_force_map(boundary, boundary, K)
            K = gcu.topological_disjoint_union(gluing_map, K, n_sphere)

        while self.AE and self.E > 0:
            e, we = self.AE.pop()
            if using_distances and hasattr(update_distances, "__call__"):
                update_distances(self.DE, we, e)
            try:
                ec = np.array([e[0] for _ in range(len(n[0]))])
            except Exception:
                ec = np.array([e[0] for _ in range(self.P + self.E)])
            res = gcu.get_nsphere_and_sampled_boundary(
                radius=2.8 * fm, center=ec, sample=10000, eps=1e-19
            )
            nsphere, boundary = res
            unpack = [j[0] for i in nsphere for j in i if isinstance(j, np.ndarray)] + [
                nsphere[-1][-1]
            ]
            n_sphere = np.asarray(unpack)
            gluing_map = gcu.nuclear_force_map(boundary, boundary, K)
            K = gcu.topological_disjoint_union(gluing_map, K, n_sphere)

        return tuple([K, self.AE, self.DF, self.DE])


def sanity_test(kind, *args):
    p, n, e, c, pd, nd, ed = args
    ac = AtomComplex(p, n, e, c, pd, nd, ed)
    if kind == "general":
        try:
            ac.general_build_complex()
            print("success ✅")
        except Exception:
            print("failed ❌")
    else:
        try:
            ac.fast_build_complex()
            print("success ✅")
        except Exception:
            print("failed ❌")


if __name__ == "__main__":
    sanity_test("", 1, 1, 1, 5, 3, 3, 0)
    sanity_test("", 2, 1, 2, 5, 3, 3, 0)
    sanity_test("", 1, 0, 1, 1, [1, 3], [1, 6], 9)
    sanity_test("", 0, 12, 12, 5, [1, 3], [1, 3], 0)
    sanity_test("", 1, 12, 0, 5, [1, 3], [1, 3], 0)
    print("*" * 10 + "fast done" + "*" * 10)
    sanity_test("general", 1, 1, 1, 5, 3, 3, 0)
    sanity_test("general", 2, 1, 2, 5, 3, 7, 0)
    sanity_test("general", 12, 1, 2, 17, 9, 9, 0)
    print("*" * 10 + "general done" + "*" * 10)
