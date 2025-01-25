import os
import sys
import json
import dill
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE_PATH = Path(__file__)
project_root = BASE_PATH.parent.parent.parent.parent.resolve()
src_dir = project_root
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from polyatomic_complexes.src.complexes.atomic_complex import AtomComplex


class BuildAtoms:
    def __init__(self):
        self.cwd = BASE_PATH
        self.datapath = BASE_PATH.parent.parent.parent.__str__() + "/dataset/construct"

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
