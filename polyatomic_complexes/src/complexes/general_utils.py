import jax
import sys
import random
import numpy as np
import jax.numpy as jnp
from typing import List, Tuple
from collections import defaultdict
from pathlib import Path

BASE_PATH = Path(__file__)
project_root = BASE_PATH.parent.parent.parent.parent.resolve()
src_dir = project_root
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from polyatomic_complexes.src.complexes.core_utils import GluingMap, NSphere


class GeneralComplexUtils:
    def __init__(self, cutoff):
        self.cutoff = cutoff

    def nuclear_force_map(self, boundary_pq, boundary_y, target) -> list:
        G = GluingMap(boundary_pq, boundary_y, target)
        return G.map

    def topological_disjoint_union(
        self, gluing_map: defaultdict, skeleton: np.ndarray, particle: np.ndarray
    ):
        for k in gluing_map.keys():
            k = np.asarray(k)
            skeleton = np.concatenate([skeleton, k])
        p = particle.ravel()
        K = np.concatenate([skeleton, p])
        return K

    def union(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return np.concatenate((X, Y), axis=0)

    def random_element_pop(self, L: List[List[int]]) -> list:
        i = random.randrange(len(L))  # get random index
        L[i], L[-1] = L[-1], L[i]  # swap with the last element
        x = L.pop()
        return x

    def get_nsphere_and_limits(
        self, radius: np.float32, center: np.ndarray
    ) -> Tuple[list, list]:
        representation = NSphere(r=radius, center=center)
        nsphere = representation.approximate()
        boundary = representation.limits
        return Tuple[nsphere, boundary]

    def centered_2_norm(self, X: np.array, center: np.array) -> np.float32:
        return np.linalg.norm(X - center)

    def get_nsphere_and_sampled_boundary(
        self, radius: np.float32, center: np.ndarray, sample: int, eps: np.float32
    ) -> Tuple[list, np.ndarray]:
        representation = NSphere(r=radius, center=center)
        nsphere = representation.approximate()
        dim = center.shape[0]
        boundary = set()

        def generate_normal(sample: int, dim: int) -> jnp.array:
            state = np.random.randint(0, 10 * 5)
            key = jax.random.PRNGKey(state)
            _, subkey = jax.random.split(key)
            X = jax.random.normal(subkey, shape=(sample, dim))
            return X

        @jax.jit
        def normalize_vector(v: jnp.array):
            return v / jnp.linalg.norm(v)

        @jax.jit
        def compute_norm(mat: np.ndarray):
            return jnp.linalg.norm(mat)

        normalize = jax.jit(normalize_vector)
        two_norm = jax.jit(compute_norm)
        numit = 0
        while len(boundary) < dim:
            if numit >= self.cutoff:
                boundary = list(boundary)
                for i, it in enumerate(boundary):
                    boundary[i] = np.asarray(it)
                remain = dim - len(boundary)
                if remain == dim:
                    limits = representation.limits
                    for lim in limits[0]:
                        boundary.append(lim)
                    rem = dim - len(boundary)
                    for _ in range(rem):
                        eps = np.random.randint(0, 10 * 2) * 1e-19
                        if (
                            type(boundary[-1]) is float
                            or type(boundary[-1]) is np.float64
                            or type(boundary[-1]) is np.float32
                        ):
                            boundary[-1] = np.array([boundary[-1]])
                        arr = np.asarray([i + eps for i in boundary[-1]])
                        boundary.append(arr)
                else:
                    for _ in range(remain):
                        epsilon = np.random.randint(0, 10 * 2) * 1e-19
                        if (
                            type(boundary[-1]) is float
                            or type(boundary[-1]) is np.float64
                            or type(boundary[-1]) is np.float32
                        ):
                            boundary[-1] = np.array([boundary[-1]])
                        arr = np.asarray([i + epsilon for i in boundary[-1]])
                        boundary.append(arr)
                break
            X = generate_normal(sample, dim)
            for i in range(sample):
                rand_vector = normalize(X[i, :])
                rand_vector = np.asarray(rand_vector)
                rand_vector = rand_vector.tolist()
                for i in range(len(rand_vector)):
                    rand_vector[i] = rand_vector[i] * radius
                norm = two_norm(rand_vector - center[0])
                if (
                    np.abs(norm - radius) < eps
                    and len(boundary) < dim
                    and type(boundary) is set
                ):
                    rand_vector = tuple(rand_vector)
                    boundary.add(rand_vector)
                elif len(boundary) >= dim:
                    break
                elif (
                    type(boundary) is list
                    and np.abs(norm - radius) < eps
                    and len(boundary) < dim
                ):
                    boundary.append(rand_vector)
            numit += 1
        boundary = list(boundary)
        for i, it in enumerate(boundary):
            boundary[i] = np.asarray(it)
        dim_check = list([boundary[0].size])
        if (
            dim_check[0] < dim
            or type(boundary[0]) is not np.ndarray
            or (
                (type(boundary) is np.ndarray or type(boundary) is list)
                and type(boundary[0]) is np.float64
            )
        ):
            for i, b in enumerate(boundary):
                if type(b) is not list and type(b) is not np.ndarray:
                    boundary[i] = np.asarray([b for _ in range(dim)]).ravel()
                else:
                    boundary[i] = np.asarray([b.ravel() for _ in range(dim)]).ravel()
        boundary = np.asarray(boundary)
        assert (
            len(boundary) == dim
            and len(boundary[0]) == dim
            and type(boundary) is np.ndarray
            and boundary.shape == (dim, dim)
        )
        return nsphere, boundary
