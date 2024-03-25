from complexes.polyatomic_complex import PolyAtomComplex
from typing import List
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
]


def fuzz_test(n=20):
    with open("dataset/construct/lookup_map.json") as data:
        lookup = json.load(data)
    assert isinstance(lookup, dict)
    for _ in range(n):
        atom_list = random.sample(list(lookup.keys()), 15)
        p = PolyAtomComplex(atom_list)
        case = (p, "general")
        cases.append(case)
        case = (p, "fast")
        cases.append(case)


fuzz_test()


@pytest.mark.parametrize(
    "polyatom,build_type",
    cases,
)
def test_build(polyatom: PolyAtomComplex, build_type: str):
    if build_type == "general":
        assert polyatom.general_build_complex()
    else:
        assert polyatom.fast_build_complex()
