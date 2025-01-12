import os
import sys
import dill
import json
import pandas as pd
from typing import List, Tuple
from multiprocessing.pool import ThreadPool as Pool
from pymatgen.core import Structure
from collections import defaultdict

sys.path.append("..")
from polyatomic_complexes.src.complexes.polyatomic_complex import PolyAtomComplex


class ProcessJDFT:
    def __init__(
        self,
        source_path=os.getcwd() + "/polyatomic_complexes/dataset/mp_matbench_jdft2d/",
        target_path=os.getcwd() + "/polyatomic_complexes/dataset/mp_matbench_jdft2d/",
    ):
        self.src = source_path
        self.tgt = target_path
        assert "matbench_jdft2d.json" in os.listdir(self.src)
        self.datapath = self.src + "matbench_jdft2d.json"
        with open(self.datapath) as f:
            self.data = json.load(f)
        cols = list(self.data["columns"])
        self.data = pd.DataFrame.from_dict(self.data["data"])
        self.data.columns = cols
        self.data.to_csv(self.src + "jdft2d.csv")
        assert isinstance(self.data, pd.DataFrame)

    def process(self) -> None:
        representations = defaultdict(tuple)
        for i, struct in enumerate(self.data["structure"]):
            structure = Structure.from_dict(struct)
            elem, comp = self.extract_structure(structure)
            atoms = self.extract_atoms(elem, comp)
            representations[i] = PolyAtomComplex(atom_list=atoms).fast_build_complex()
        assert len(representations) == len(self.data)
        with open(self.tgt + "fast_complex_lookup_repn.pkl", "wb") as f:
            dill.dump(representations, f)
        return None

    def process_stacked(self) -> None:
        representations = defaultdict(tuple)
        for i, struct in enumerate(self.data["structure"]):
            structure = Structure.from_dict(struct)
            elem, comp = self.extract_structure(structure)
            atoms = self.extract_atoms(elem, comp)
            representations[i] = PolyAtomComplex(atom_list=atoms).fast_stacked_complex()
        assert len(representations) == len(self.data)
        with open(self.tgt + "stacked_complex_lookup_repn.pkl", "wb") as f:
            dill.dump(representations, f)
        return None

    def process_deep_complexes(self) -> None:
        representations = defaultdict(tuple)

        def helper(data):
            i, struct = data
            structure = Structure.from_dict(struct)
            elem, comp = self.extract_structure(structure)
            atoms = self.extract_atoms(elem, comp)
            pc = PolyAtomComplex(atom_list=atoms)
            repn = pc.general_build_complex()
            representations[i] = repn
            print("done")
            return repn

        with Pool() as p:
            p.map(
                func=helper,
                iterable=list(enumerate(self.data["structure"])),
            )

        assert len(representations) == len(self.data)
        with open(self.tgt + "deep_complex_lookup_repn.pkl", "wb") as f:
            dill.dump(representations, f)
        return None

    def extract_structure(self, struct: Structure) -> Tuple[List, dict]:
        structure_info = defaultdict(list)
        cmp = struct.composition
        for k in cmp:
            if k != "@module" and k != "@class" and k != "@version":
                strk = f"{k}"
                key = strk.replace("Element ", "")
                structure_info[key] = cmp[k]
        return tuple([list(structure_info.keys()), structure_info])

    def extract_atoms(self, element, composition) -> List:
        atom_list = []
        assert isinstance(composition, dict) and isinstance(element, list)
        for k in element:
            atom_list += [k] * int(composition[k])
        return atom_list


if __name__ == "__main__":
    prc = ProcessJDFT()
    # prc.process()
    # prc.process_deep_complexes()
    prc.process_stacked()
