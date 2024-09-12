import os
import torch
import json
import random
import pytest
import polyatomic_complexes
import polyatomic_complexes.src.complexes as complexes
import polyatomic_complexes.experiments as experiments


cwd = os.getcwd()
root_data = cwd + "/polyatomic_complexes/"


cases = []


def fuzz_test(n=20, k=15):
    with open(root_data + "dataset/construct/lookup_map.json") as data:
        lookup = json.load(data)
    assert isinstance(lookup, dict)
    for _ in range(n):
        atom_list = random.sample(list(lookup.keys()), k)
        p = complexes.polyatomic_complex.PolyAtomComplex(atom_list)
        case = (p, "general")
        cases.append(case)
        case = (p, "fast")
        cases.append(case)
        case = (p, "fast_stacked")
        cases.append(case)


fuzz_test(20, 1)
fuzz_test(10, 22)
fuzz_test(5, 27)


@pytest.mark.parametrize(
    "polyatom,build_type",
    cases,
)
def test_build(polyatom: complexes.polyatomic_complex.PolyAtomComplex, build_type: str):
    if build_type == "general":
        assert polyatom.general_build_complex()
    elif build_type == "fast_stacked":
        assert polyatom.fast_stacked_complex()
    else:
        assert polyatom.fast_build_complex()


cases2 = []


def fuzz_test(n=10):
    with open(root_data + "dataset/construct/lookup_map.json") as data:
        lookup = json.load(data)
    assert isinstance(lookup, dict)
    items = random.sample(list(lookup.items()), n)
    for k, it in items:
        p, n, e = it
        a = complexes.atomic_complex.AtomComplex(p, n, e, 5, 3, 3, 0)
        case = (a, p, n, e)
        cases2.append(case)


fuzz_test()


@pytest.mark.parametrize(
    "atom,ans_p,ans_n,ans_e",
    cases,
)
def test_build(
    atom: complexes.atomic_complex.AtomComplex,
    ans_p: int,
    ans_n: int,
    ans_e: int,
):
    assert isinstance(atom.P, int) and atom.P >= 0
    assert isinstance(atom.N, int) and atom.N >= 0
    assert isinstance(atom.E, int) and atom.E >= 0
    assert max(atom.P, atom.N) >= 1
    assert min([atom.P, atom.N, atom.E]) >= 0
    assert isinstance(atom.d_p, list) or isinstance(atom.d_p, int)
    assert isinstance(atom.d_n, list) or isinstance(atom.d_n, int)
    assert isinstance(atom.d_e, list) or isinstance(atom.d_e, int)
    if isinstance(atom.d_p, int):
        assert atom.d_p > 0
    else:
        assert len(atom.d_p) == 2 and min(atom.d_p) >= 0
    if isinstance(atom.d_n, int):
        assert atom.d_n > 0
    else:
        assert len(atom.d_n) == 2 and min(atom.d_n) >= 0
    assert isinstance(atom.d_e, int) and atom.d_e >= 0
    assert isinstance(atom.cutoff, int)
    assert atom.P == ans_p and atom.N == ans_n and atom.E == ans_e
    assert atom.fast_build_complex()
    if (
        isinstance(atom.d_p, int)
        and isinstance(atom.d_n, int)
        and isinstance(atom.d_e, int)
    ):
        assert atom.general_build_complex()


def test_data_processing():
    if "data/esol" not in os.listdir(cwd + "sanity_testing_build"):
        os.makedirs(cwd + "/sanity_testing_build/data/esol")
    src_path = root_data + "dataset/esol/"
    dummy_path = cwd + "/sanity_testing_build/data/esol"
    e = polyatomic_complexes.complexes.process_esol.ProcessESOL(src_path, dummy_path)
    e.process()
    e.process_deep_complexes()
    e.process_stacked()
    assert len(os.listdir(dummy_path)) == 3


test_data_processing()
