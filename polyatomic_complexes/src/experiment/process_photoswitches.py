import os
import re
import sys
import dill
import pandas as pd
from rdkit import Chem
from collections import defaultdict

sys.path.append("..")
from polyatomic_complexes.src.complexes.polyatomic_complex import PolyAtomComplex


class ProcessPhotoswitches:
    def __init__(
        self,
        source_path=os.getcwd() + "/polyatomic_complexes/dataset/photoswitches/",
        target_path=os.getcwd() + "/polyatomic_complexes/dataset/photoswitches/",
    ):
        self.src = source_path
        self.tgt = target_path
        assert "photoswitches.csv" in os.listdir(self.src)
        self.datapath = self.src + "photoswitches.csv"
        self.data = pd.read_csv(self.datapath)
        assert isinstance(self.data, pd.DataFrame)

    def process(self) -> None:
        representations = defaultdict(tuple)
        for i, row in enumerate(self.data["SMILES"]):
            print(f"row {row}")
            print(f"tpe {type(row)}")
            atoms = self.smiles_to_atoms(row)
            representations[i] = PolyAtomComplex(atom_list=atoms).fast_build_complex()

        with open(self.tgt + "fast_complex_lookup_repn.pkl", "wb") as f:
            dill.dump(representations, f)
        return None

    def process_deep_complexes(self) -> None:
        representations = defaultdict(tuple)
        for i, row in enumerate(self.data["SMILES"]):
            print(f"row {row}")
            print(f"tpe {type(row)}")
            atoms = self.smiles_to_atoms(row)
            representations[i] = PolyAtomComplex(
                atom_list=atoms
            ).general_build_complex()

        with open(self.tgt + "deep_complex_lookup_repn.pkl", "wb") as f:
            dill.dump(representations, f)
        return None

    def process_stacked(self) -> None:
        representations = defaultdict(tuple)
        for i, row in enumerate(self.data["SMILES"]):
            print(f"row {row}")
            print(f"tpe {type(row)}")
            atoms = self.smiles_to_atoms(row)
            representations[i] = PolyAtomComplex(atom_list=atoms).fast_stacked_complex()

        with open(self.tgt + "stacked_complex_lookup_repn.pkl", "wb") as f:
            dill.dump(representations, f)
        return None

    def smiles_to_atoms(self, smile: str) -> list:
        assert isinstance(smile, str)
        mol = Chem.MolFromSmiles(smile)
        Chem.Kekulize(mol)
        atom_counts = {}
        for atom in mol.GetAtoms():
            neighbors = [
                (neighbor.GetSymbol(), bond.GetBondType())
                for neighbor, bond in zip(atom.GetNeighbors(), atom.GetBonds())
            ]
            neighbors.sort()
            key = "{}-{}".format(
                atom.GetSymbol(),
                "".join(
                    f"{symbol}{'-' if bond_order == 1 else '=' if bond_order == 2 else '#'}"
                    for symbol, bond_order in neighbors
                ),
            )
            atom_counts[key] = atom_counts.get(key, 0) + 1
        regex = re.compile("[^a-zA-Z]")
        atoms_list = []
        for k in atom_counts:
            cleaned = regex.sub(" ", k).split(" ")
            res = []
            for ch in cleaned:
                r = ch.strip()
                if r != "" and " " not in r:
                    res.append(r)
            atoms_list += res
        return atoms_list


if __name__ == "__main__":
    prc = ProcessPhotoswitches()
    # prc.process()
    # prc.process_deep_complexes()
    prc.process_stacked()
