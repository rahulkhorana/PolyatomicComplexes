import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple

seed = np.random.randint(0, 10 * 3)
fm = np.float64(1e-15)
key = jax.random.PRNGKey(seed)


class Electron:
    def __init__(self, dim, num_pts=1):
        self.num_pts = num_pts
        self.r = np.random.randint(0, 5)
        self.points = jax.random.normal(key * self.r, shape=(dim + 1, num_pts))
        self.d = 1 / (dim + 1)
        self.cutoff = 2.8 * fm
        self.ee = []
        self.w = []

    def build_electron(self) -> Tuple[np.array, np.array]:
        for i in range(self.num_pts):
            v = self.points[:, i]
            length = jnp.linalg.norm(v)
            if length >= self.cutoff:
                tmpr = np.random.rand() ** self.d
                r = tmpr if tmpr != 0 else np.random.randint(0.3, 1) ** self.d
                v = jnp.multiply(v, r * self.cutoff)
            self.ee.append(v)
            wv = self.wave_fn()
            self.w.append(wv)
        self.ee = np.unique(self.ee, axis=0)
        self.w = np.array(self.w)
        assert len(self.ee) == len(self.w)
        return tuple([self.ee, self.w])

    def wave_fn(self, pos=0, mom=0, sigma=0.2):
        randc = np.random.rand()
        return (
            lambda x: jnp.linalg.norm(
                np.exp(-1j * mom * x)
                * np.exp(-np.square(x - pos) / sigma / sigma, dtype=complex)
            )
            + randc
        )


class Proton:
    def __init__(self, dim, num_pts=1):
        self.dim = dim
        assert self.dim > 0
        self.num_pts = num_pts
        self.r = np.random.randint(0, 100)
        self.points = jax.random.normal(
            key * self.r, shape=(dim, num_pts)
        )  # output shape determined by shape param ; key param is random seed state
        self.d = 1 / (dim)
        self.cutoff = 1 * fm  # hardcode cutoff
        self.pd = []

    def build_proton(self) -> Tuple[np.array]:
        for i in range(self.num_pts):
            v = self.points[:, i]
            length = jnp.linalg.norm(v)
            if length >= self.cutoff:
                r = np.random.rand() ** self.d
                v = jnp.multiply(v, r * self.cutoff)
            self.pd.append(v)
        self.pd = np.unique(self.pd, axis=0)
        assert len(self.pd) == self.num_pts
        return tuple([self.pd])


class Neutron:
    def __init__(self, dim, num_pts=1):
        self.dim = dim
        assert self.dim > 0
        self.num_pts = num_pts
        self.r = np.random.randint(0, 100)
        self.points = jax.random.normal(key * self.r, shape=(dim, num_pts))
        self.d = 1 / (dim)
        self.cutoff = 0.8 * fm
        self.nd = []

    def build_neutron(self) -> Tuple[np.array]:
        for i in range(self.num_pts):
            v = self.points[:, i]
            length = jnp.linalg.norm(v)
            if length >= self.cutoff:
                r = np.random.rand() ** self.d
                v = jnp.multiply(v, r * self.cutoff)
            self.nd.append(v)
        self.nd = np.unique(self.nd, axis=0)
        assert len(self.nd) == self.num_pts
        return tuple([self.nd])


if __name__ == "__main__":
    print(fm)
    e = Electron(0).build_electron()
    print(e)
    n = Neutron(3, 5).build_neutron()
    print(n)
    p = Proton(3, 10).build_proton()
    print(p)
