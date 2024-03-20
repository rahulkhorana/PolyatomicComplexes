import os
import dill
import json
import numpy as np
import jax.numpy as jnp
from typing import Tuple
from core_utils import GluingMap, ElectronField

class PolyAtomComplex():

    def __init__(self, atom_list, using_radial=False, using_force=False, update_forces=None, update_radial=None):
        assert isinstance(atom_list, list) and len(atom_list) > 0
        assert isinstance(using_radial, bool)
        assert isinstance(using_force, bool)
        self.atoms = atom_list
        self.using_radial = using_radial
        self.using_force = using_force
        self.cwd = os.getcwd()
        self.datapath = self.cwd + '/dataset/construct'
        assert 'atom_lookup.pkl' in os.listdir(self.datapath)
        assert 'lookup_map.json' in os.listdir(self.datapath)
        self.reference = self.datapath + '/lookup_map.json'
        with open(self.reference, 'rb') as f:
            self.lookup_map = json.load(f)
        assert isinstance(self.lookup_map, dict)
        assert isinstance(using_radial, bool)
        assert isinstance(using_force, bool)
        if using_radial:
            assert hasattr(update_radial, '__call__')
            self.update_radial = update_radial
        if using_force:
            assert hasattr(update_forces, '__call__')
            self.update_forces = update_forces
    
    def fast_build_complex(self) -> Tuple:
        lookup = self.datapath+'/atom_lookup.pkl'
        with open(lookup, 'rb') as f:
            lookup = dill.load(f)
        C = []
        for a in self.atoms:
            acomplex = lookup[a]
            C.append(acomplex[0])
        C = np.asarray(C)
        return tuple([C, [], [], []])
    
    def general_build_complex(self):
        lookup = self.datapath+'/atom_lookup.pkl'
        with open(lookup, 'rb') as f:
            lookup = dill.load(f)
        A = []
        for a in enumerate(self.atoms):
            acomplex = lookup[a]
            A.append(acomplex)
        C = np.asarray([])
        self.E = ElectronField()
        self.F = np.matrix()
        self.DE = np.matrix()

        def disjont_union(gluing_map:np.ndarray, skeleton:np.ndarray, particle:np.ndarray) -> np.ndarray:
            skeleton = np.concatenate([skeleton.ravel(), gluing_map.ravel()])
            K = np.concatenate([skeleton, particle.ravel()])
            K = np.unique(K)
            return K

        def electron_union(EF_1:tuple, EF_2:tuple) -> None:
            ee0_1, ee0_2 = EF_1[0], EF_2[0]
            wv_1, wv_2 = EF_1[1], EF_2[1]
            assert isinstance(ee0_1,np.ndarray) and isinstance(ee0_2,np.ndarray)
            assert isinstance(wv_1,np.ndarray) and isinstance(wv_2,np.ndarray)
            e_field = np.union1d(ee0_1, ee0_2)
            wves = np.concatenate([wv_1.ravel(), wv_2.ravel()])
            e_field, waves = e_field.ravel(), wves.ravel()
            self.E.add_electrons(e_field, waves)
            return None
        
        def dir_sum(a:np.ndarray,b:np.ndarray) -> np.ndarray:
            dsum = np.zeros(np.add(a.shape,b.shape))
            dsum[:a.shape[0],:a.shape[1]]=a
            dsum[a.shape[0]:,a.shape[1]:]=b
            return dsum
        
        while A:
            I = A.pop(0)
            k, ae, df, de = I[0]
            phi = GluingMap(C, k, target=C).construct_map()
            map = np.asarray(list(phi.keys()), dtype=jnp.float32)
            map = np.unique(map)
            if len(C) == 0:
                C = k
            C = disjont_union(map, C, k)
            electron_union(self.E, ae)
            if self.using_force:
                self.F = dir_sum(self.F, df)
                self.update_forces(self.F)
            if self.using_radial:
                self.DE = dir_sum(self.DE, de)
                self.update_radial(self.DE)
        
        return tuple([C, self.E, self.F, self.DE])

def sanity_test(atom_list, kind):
    pac = PolyAtomComplex(atom_list)
    if kind == 'general':
        try:
            pac.general_build_complex()
            print(f'success ✅')
        except:
            print('failed ❌')
    else:
        try:
            pac.fast_build_complex()
            print(f'success ✅')
        except:
            print('failed ❌')



if __name__ == '__main__':
    sanity_test(['H', 'H', 'O'], '')
    sanity_test(['H', 'H', 'O'], 'general')