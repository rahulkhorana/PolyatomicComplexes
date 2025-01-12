import os
import sys
import dill
import json
import pandas as pd
from typing import List
from multiprocessing.pool import ThreadPool as Pool
from collections import defaultdict

sys.path.append("..")
from polyatomic_complexes.src.complexes.polyatomic_complex import PolyAtomComplex


class ProcessMP:
    def __init__(
        self,
        source_path=os.getcwd() + "/polyatomic_complexes/dataset/materials_project/",
        target_path=os.getcwd() + "/polyatomic_complexes/dataset/materials_project/",
    ):
        self.src = source_path
        self.tgt = target_path
        assert "materials_data.csv" in os.listdir(self.src)
        self.datapath = self.src + "materials_data.csv"
        self.data = pd.read_csv(self.datapath, low_memory=False)
        assert isinstance(self.data, pd.DataFrame)

    def process(self) -> None:
        representations = defaultdict(tuple)
        for i, data in enumerate(zip(self.data["elements"], self.data["composition"])):
            elem, comp = data
            elem = eval(elem)
            print(f"comp {comp}")
            try:
                comp = json.loads(comp)
                atoms = self.extract_atoms(elem, comp)
            except Exception:
                # single edge case ['He']
                comp = eval(comp)
                atoms = comp
            representations[i] = PolyAtomComplex(atom_list=atoms).fast_build_complex()
        assert len(representations) == len(self.data)
        with open(self.tgt + "fast_complex_lookup_repn.pkl", "wb") as f:
            dill.dump(representations, f)
        return None

    def process_stacked(self) -> None:
        representations = defaultdict(tuple)
        for i, data in enumerate(zip(self.data["elements"], self.data["composition"])):
            elem, comp = data
            elem = eval(elem)
            print(f"comp {comp}")
            try:
                comp = json.loads(comp)
                atoms = self.extract_atoms(elem, comp)
            except Exception:
                # single edge case ['He']
                comp = eval(comp)
                atoms = comp
            representations[i] = PolyAtomComplex(atom_list=atoms).fast_stacked_complex()
        assert len(representations) == len(self.data)
        with open(self.tgt + "stacked_complex_lookup_repn.pkl", "wb") as f:
            dill.dump(representations, f)
        return None

    def process_deep_complexes(self) -> None:
        representations = defaultdict(tuple)

        def helper(data):
            i, row = data
            elem, comp = row
            elem = eval(elem)
            try:
                comp = json.loads(comp)
                atoms = self.extract_atoms(elem, comp)
            except Exception:
                # single edge case ['He']
                comp = eval(comp)
                atoms = comp
            pc = PolyAtomComplex(atom_list=atoms)
            repn = pc.general_build_complex()
            representations[i] = repn
            print("done")
            return repn

        with Pool() as p:
            p.map(
                func=helper,
                iterable=list(
                    enumerate(zip(self.data["elements"], self.data["composition"]))
                ),
            )

        assert len(representations) == len(self.data)
        with open(self.tgt + "deep_complex_lookup_repn.pkl", "wb") as f:
            dill.dump(representations, f)
        return None

    def extract_atoms(self, element, composition) -> List:
        atom_list = []
        assert isinstance(composition, dict) and isinstance(element, list)
        for k in element:
            atom_list += [k] * int(composition[k])
        return atom_list


if __name__ == "__main__":
    prc = ProcessMP()
    # prc.process()
    # prc.process_deep_complexes()
    prc.process_stacked()
