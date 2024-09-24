import os
import sys
import json
import dill
import numpy as np
from collections import defaultdict

sys.path.append(".")
from .atomic_complex import AtomComplex

import importlib_resources

import_path = importlib_resources.files("polyatomic_complexes").__str__()


class BuildAtoms:
    def __init__(self):
        self.cwd = import_path
        self.datapath = self.cwd + "/dataset/construct"

    def build_lookup_table(self) -> None:
        assert "lookup_map.json" in os.listdir(self.datapath)
        with open(self.datapath + "/lookup_map.json") as data:
            d = json.load(data)

        lookup_table = defaultdict(list)

        for i, element in enumerate(d):
            n_protons, n_neutrons, n_electrons = d[element]
            ac = AtomComplex(n_protons, n_neutrons, n_electrons, 5, 3, 3, 0)
            complex = ac.fast_build_complex()
            print(f"finished {i}")
            lookup_table[element] = complex

        with open(self.datapath + "/atom_lookup.pkl", "wb") as f:
            dill.dump(lookup_table, f)

        return None

    def sanity(self):
        lookup = self.datapath + "/atom_lookup.pkl"
        with open(lookup, "rb") as f:
            table = dill.load(f)
        try:
            assert len(table.keys()) == 118
            print(table["He"])
            print(table["Os"])
            print(table["Bk"])
            assert isinstance(table["He"], tuple)
            assert isinstance(table["He"][0], np.ndarray)
            assert isinstance(table["Os"], tuple)
            assert isinstance(table["Os"][0], np.ndarray)
            assert isinstance(table["Bk"], tuple)
            assert isinstance(table["Bk"][0], np.ndarray)
            print("Success ✅")
        except Exception:
            print("Failed ❌")


if __name__ == "__main__":
    build = BuildAtoms()
    # build.build_lookup_table()
    build.sanity()
