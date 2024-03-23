import os
import re
import dill
import pandas as pd
from rdkit import Chem
from collections import defaultdict
from polyatomic_complex import PolyAtomComplex

class ProcessLipophilicity:
    def __init__(self):
        self.src = os.getcwd()+'/dataset/lipophilicity/'
        assert 'Lipophilicity.csv' in os.listdir(self.src)
        self.datapath = self.src + 'Lipophilicity.csv'
        self.data = pd.read_csv(self.datapath)
        assert isinstance(self.data, pd.DataFrame)
    
    def process(self) -> None:
        representations = defaultdict(tuple)
        for i, row in enumerate(self.data['smiles']):
            print(f'row {row}')
            print(f'tpe {type(row)}')
            atoms = self.smiles_to_atoms(row)
            representations[i] = PolyAtomComplex(atom_list=atoms).fast_build_complex()
        
        with open(self.src+'fast_complex_lookup_repn.pkl', 'wb') as f:
            dill.dump(representations, f)
        return None
    
    def process_deep_complexes(self) -> None:
        representations = defaultdict(tuple)
        for i, row in enumerate(self.data['smiles']):
            print(f'row {row}')
            print(f'tpe {type(row)}')
            atoms = self.smiles_to_atoms(row)
            representations[i] = PolyAtomComplex(atom_list=atoms).general_build_complex()
        
        with open(self.src+'deep_complex_lookup_repn.pkl', 'wb') as f:
            dill.dump(representations, f)
        return None

    def smiles_to_atoms(self, smile:str) -> list:
        assert isinstance(smile, str)
        mol = Chem.MolFromSmiles(smile)
        Chem.Kekulize(mol)
        atom_counts = {}
        for atom in mol.GetAtoms():
            neighbors = [(neighbor.GetSymbol(), bond.GetBondType())
                        for neighbor, bond in zip(atom.GetNeighbors(), atom.GetBonds())]
            neighbors.sort()
            key = "{}-{}".format(atom.GetSymbol(), "".join(f"{symbol}{'-' if bond_order == 1 else '=' if bond_order == 2 else '#'}" for symbol, bond_order in neighbors))
            atom_counts[key] = atom_counts.get(key, 0) + 1
        regex = re.compile('[^a-zA-Z]')
        atoms_list = []
        for k in atom_counts:
            cleaned = regex.sub(' ', k).split(' ')
            res = []
            for l in cleaned:
                r = l.strip()
                if r != '' and ' ' not in r:
                    res.append(r)
            atoms_list += res
        return atoms_list


if __name__ == '__main__':
    prc = ProcessLipophilicity()
    #prc.process()
    prc.process_deep_complexes()