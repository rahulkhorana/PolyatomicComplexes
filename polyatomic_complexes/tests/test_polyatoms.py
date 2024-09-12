from src.complexes.polyatomic_complex import PolyAtomComplex
from typing import List
import os
import pytest
import json
import random


atom_lists = [
    ["H", "H", "O"],
    ["C", "H", "H", "H"],
    [
        "Np",
        "U",
        "P",
        "P",
        "P",
        "P",
        "H",
        "H",
        "H",
        "H",
        "C",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "C",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
    ],
]


cases = [
    (PolyAtomComplex(atom_list=atom_lists[0]), "general"),
    (PolyAtomComplex(atom_list=atom_lists[1]), "general"),
    (PolyAtomComplex(atom_list=atom_lists[2]), "general"),
    (PolyAtomComplex(atom_list=atom_lists[0]), "fast"),
    (PolyAtomComplex(atom_list=atom_lists[1]), "fast"),
    (PolyAtomComplex(atom_list=atom_lists[2]), "fast"),
    (PolyAtomComplex(atom_list=atom_lists[0]), "fast_stacked"),
    (PolyAtomComplex(atom_list=atom_lists[1]), "fast_stacked"),
    (PolyAtomComplex(atom_list=atom_lists[2]), "fast_stacked"),
]

os.getcwd()
root_data = os.getcwd() + "/polyatomic_complexes/"


def fuzz_test(n=20, k=15):
    with open(root_data + "dataset/construct/lookup_map.json") as data:
        lookup = json.load(data)
    assert isinstance(lookup, dict)
    for _ in range(n):
        atom_list = random.sample(list(lookup.keys()), k)
        p = PolyAtomComplex(atom_list)
        case = (p, "general")
        cases.append(case)
        case = (p, "fast")
        cases.append(case)
        case = (p, "fast_stacked")
        cases.append(case)


fuzz_test(50, 1)
fuzz_test(40, 3)
fuzz_test(30, 10)
fuzz_test(20, 15)
fuzz_test(10, 19)
fuzz_test(8, 22)
fuzz_test(5, 27)


@pytest.mark.parametrize(
    "polyatom,build_type",
    cases,
)
def test_build(polyatom: PolyAtomComplex, build_type: str):
    if build_type == "general":
        assert polyatom.general_build_complex()
    elif build_type == "fast_stacked":
        assert polyatom.fast_stacked_complex()
    else:
        assert polyatom.fast_build_complex()
