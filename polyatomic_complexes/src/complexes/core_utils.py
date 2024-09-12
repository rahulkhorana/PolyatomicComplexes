import re
import numpy as np
from typing import Tuple
from collections import defaultdict


class GluingMap:
    def __init__(self, boundary1, boundary2, target):
        self.b1 = boundary1
        self.b2 = boundary2
        self.target = target
        self.checker()
        self.construct_map()
        self.map = self.construct_map()

    def checker(self) -> None:
        assert type(self.b1) is np.ndarray
        assert type(self.b2) is np.ndarray
        assert len(self.b1) > 0 and len(self.b2) > 0
        assert len(self.b1.shape) >= 1 and len(self.b2.shape) >= 1
        assert self.b1.shape[0] > 0 and self.b2.shape[0] > 0
        assert type(self.target) is np.ndarray
        assert len(self.target.shape) > 0 and len(self.target) >= 0
        assert self.target.shape[0] >= 0

    def cartesian_product_recursive(self, *arrays, out=None):
        arrays = [np.asarray(x) for x in arrays]
        dtype = arrays[0].dtype

        n = np.prod([x.size for x in arrays])
        if out is None:
            out = np.zeros([n, len(arrays)], dtype=dtype)

        m = n // arrays[0].size
        out[:, 0] = np.repeat(arrays[0], m)
        if arrays[1:]:
            self.cartesian_product_recursive(arrays[1:], out=out[0:m, 1:])
            for j in range(1, arrays[0].size):
                out[j * m : (j + 1) * m, 1:] = out[0:m, 1:]
        return out

    def construct_map(self):
        d = defaultdict(np.ndarray)
        product = self.cartesian_product_recursive(self.b1, self.b2)
        value = self.target
        for row in product:
            key = tuple(row)
            d[key] = value
        return d


class MiscUtilFunctions:
    def __init__(self, valid=True):
        self.valid = valid
        self.num_fns = 1
        self.fns = ["extract_defaultdict"]
        self.version = "1.0"

    def extract_defaultdict(self, str_data: str, dtype: str = "str") -> defaultdict:
        p = re.compile(r"^defaultdict\(<class '(\w+)'>")
        c = p.findall(str_data)[0]
        defdict = eval(str_data.replace("<class '%s'>" % c, dtype))
        assert type(defdict) is defaultdict
        assert len(defdict) > 0
        key_test = list(defdict.keys())[0]
        assert len(key_test) > 0
        return defdict


class ElectronField:
    def __init__(self):
        self.efield = np.array([])
        self.waves = np.array([])

    def get_efield(self) -> np.ndarray:
        return self.efield

    def get_waves(self) -> np.ndarray:
        return self.waves

    def add_electrons(self, electrons: np.ndarray, waves: np.ndarray):
        if len(waves) > len(electrons):
            waves = waves[: len(electrons)]
        elif len(waves) < len(electrons):
            waves = waves + waves[: len(electrons) - len(waves)]
        assert len(waves) == len(electrons)
        assert type(electrons) is np.ndarray and type(waves) is np.ndarray
        self.efield = np.append(self.efield, electrons)
        self.waves = np.append(self.waves, waves)
        assert len(self.efield) == len(self.waves)

    def get_index(self, i: int, j: int) -> int:
        assert type(i) is int and type(j) is int and i > 0 and j > 0
        index = (i + j) % len(self.efield)
        return index

    def get_ij_wave_electron(self, i: int, j: int) -> Tuple[np.ndarray, np.ndarray]:
        index = self.get_index(i, j)
        return tuple([self.efield[index], self.waves[index]])


class NSphere(object):
    # author credited: Kip Hart
    # docs: https://docs.microstructpy.org/en/latest/_modules/microstructpy/geometry/sphere.html
    # we make use of this code per the MIT license
    def __init__(self, **kwargs):
        if "r" in kwargs:
            self.r = kwargs["r"]
        elif "radius" in kwargs:
            self.r = kwargs["radius"]
        elif "d" in kwargs:
            self.r = 0.5 * kwargs["d"]
        elif "diameter" in kwargs:
            self.r = 0.5 * kwargs["diameter"]
        elif "size" in kwargs:
            self.r = 0.5 * kwargs["size"]
        else:
            self.r = 1

        if "center" in kwargs:
            self.center = kwargs["center"]
        elif "position" in kwargs:
            self.center = kwargs["position"]
        else:
            self.center = []

    @classmethod
    def best_fit(cls, points):
        pts = np.array(points)
        n_pts, n_dim = pts.shape
        if n_pts <= n_dim:
            mid = pts.mean(axis=0)
            rel_pos = pts - mid
            dist = np.linalg.norm(rel_pos, axis=1).mean()
            return cls(center=mid, radius=dist)

        # translate points to average position
        bcenter = pts.mean(axis=0)
        pts -= bcenter

        # Assemble matrix and vector of sums
        mat = np.zeros((n_dim, n_dim))
        vec = np.zeros(n_dim)

        for i in range(n_dim):
            for j in range(n_dim):
                mat[i, j] = np.sum(pts[:, i] * pts[:, j])
                vec[i] += np.sum(pts[:, i] * pts[:, j] * pts[:, j])
        vec *= 0.5

        # Solve linear system for the center
        try:
            cen_b = np.linalg.solve(mat, vec)
        except np.linalg.linalg.LinAlgError:
            cen_b = pts.mean(axis=0)
        cen = cen_b + bcenter

        # Calculate the radius
        alpha = np.sum(cen_b * cen_b) + np.trace(mat) / n_pts
        R = np.sqrt(alpha)

        # Create the instance
        return cls(center=cen, radius=R)

    def __str__(self):
        str_str = "Radius: " + str(self.r) + "\n"
        str_str += "Center: " + str(tuple(self.center))
        return str_str

    def __repr__(self):
        repr_str = "NSphere("
        repr_str += "r=" + repr(self.r) + ", "
        repr_str += "center=" + repr(tuple(self.center)) + ")"
        return repr_str

    def __eq__(self, nsphere):
        if not hasattr(nsphere, "r"):
            return False

        if not np.isclose(self.r, nsphere.r):
            return False

        if not hasattr(nsphere, "center"):
            return False

        c1 = np.array(self.center)
        c2 = np.array(nsphere.center)

        if c1.shape != c2.shape:
            return False

        dx = np.array(self.center) - np.array(nsphere.center)
        if not np.all(np.isclose(dx, 0)):
            return False
        return True

    def __neq__(self, nsphere):
        return not self.__eq__(nsphere)

    @property
    def radius(self):
        """float: radius of n-sphere."""
        return self.r

    @property
    def d(self):
        """float: diameter of n-sphere."""
        return 2 * self.r

    @property
    def diameter(self):
        """float: diameter of n-sphere."""
        return 2 * self.r

    @property
    def size(self):
        """float: size (diameter) of n-sphere."""
        return 2 * self.r

    @property
    def position(self):
        """list: position of n-sphere."""
        return self.center

    @property
    def bound_max(self):
        """tuple: maximum bounding n-sphere"""
        return tuple(list(self.center) + [self.r])

    @property
    def bound_min(self):
        """tuple: minimum interior n-sphere"""
        return self.bound_max

    @property
    def limits(self):
        """list: list of (lower, upper) bounds for the bounding box"""
        return [(x - self.r, x + self.r) for x in self.center]

    @property
    def sample_limits(self):
        """list: list of (lower, upper) bounds for the sampling region"""
        return self.limits

    def approximate(self):
        return [tuple(list(self.center) + [self.r])]

    def within(self, points):
        pts = np.array(points)
        single_pt = pts.ndim == 1
        if single_pt:
            pts = pts.reshape(1, -1)

        rel_pos = pts - np.array(self.center)
        sq_dist = np.sum(rel_pos * rel_pos, axis=-1)

        mask = sq_dist <= self.r * self.r
        if single_pt:
            return mask[0]
        else:
            return mask

    def reflect(self, points):
        pts = np.array(points)
        single_pt = pts.ndim == 1
        if single_pt:
            pts = pts.reshape(1, -1)

        rel_pos = pts - np.array(self.center)
        cen_dist = np.sqrt(np.sum(rel_pos * rel_pos, axis=-1))
        mask = cen_dist > 0
        new_dist = 2 * self.r - cen_dist[mask]
        scl = new_dist / cen_dist[mask]

        new_rel_pos = np.zeros(pts.shape)
        new_rel_pos[mask] = scl.reshape(-1, 1) * rel_pos[mask]
        new_rel_pos[~mask] = 0
        new_rel_pos[~mask, 0] = 2 * self.r

        new_pts = new_rel_pos + np.array(self.center)
        if single_pt:
            return new_pts[0]
        else:
            return new_pts
