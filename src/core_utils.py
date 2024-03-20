import re
import jax
import numpy as np
import jax.numpy as jnp
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

    def cartesian_product_recursive(self,*arrays, out=None):
        arrays = [np.asarray(x) for x in arrays]
        dtype = arrays[0].dtype

        n = np.prod([x.size for x in arrays])
        if out is None:
            out = np.zeros([n, len(arrays)], dtype=dtype)

        m = n // arrays[0].size
        out[:,0] = np.repeat(arrays[0], m)
        if arrays[1:]:
            self.cartesian_product_recursive(arrays[1:], out=out[0:m,1:])
            for j in range(1, arrays[0].size):
                out[j*m:(j+1)*m,1:] = out[0:m,1:]
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
        self.fns = ['extract_defaultdict']
        self.version = '1.0'
    
    def extract_defaultdict(self, str_data:str, dtype:str='str') -> defaultdict:
        p = re.compile(r"^defaultdict\(<class '(\w+)'>")
        c = p.findall(str_data)[0]
        defdict = eval(str_data.replace("<class '%s'>"% c, dtype))
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
    
    def add_electrons(self, electrons:np.ndarray, waves:np.ndarray):
        if len(waves) > len(electrons):
            waves =  waves[:len(electrons)]
        elif len(waves) < len(electrons):
            waves = waves + waves[:len(electrons)-len(waves)]
        assert len(waves) == len(electrons)
        assert type(electrons) is np.ndarray and type(waves) is np.ndarray
        self.efield = np.append(self.efield, electrons)
        self.waves = np.append(self.waves, waves)
        assert len(self.efield) == len(self.waves)
    
    def get_index(self, i:int,j:int) -> int:
        assert type(i) is int and type(j) is int and i > 0 and j > 0
        index = (i+j)%len(self.efield)
        return index

    def get_ij_wave_electron(self, i:int, j:int) -> Tuple[np.ndarray, np.ndarray]:
        index = self.get_index(i,j)
        return tuple([self.efield[index], self.waves[index]])