import os
import json
import dill
from collections import defaultdict
from atomic_complex import AtomComplex

class BuildAtoms():
   
   def __init__(self):
       self.cwd = os.getcwd()
       self.datapath = self.cwd + '/dataset/construct'
   
   def build_lookup_table(self) -> None:
        assert 'lookup_map.json' in os.listdir(self.datapath)
        with open(self.datapath+'/lookup_map.json') as data:
            d = json.load(data)
        
        lookup_table = defaultdict(list)

        for i, element in enumerate(d):
            n_protons, n_neutrons, n_electrons = d[element]
            ac = AtomComplex(n_protons, n_neutrons, n_electrons, 5, 3, 3, 0)
            complex = ac.fast_build_complex()
            print(f'finished {i}')
            lookup_table[element] = complex
        
        with open(self.datapath+'/atom_lookup.pkl', 'wb') as f:
            dill.dump(lookup_table, f)
        
        return None


if __name__ == '__main__':
    build = BuildAtoms()
    build.build_lookup_table()