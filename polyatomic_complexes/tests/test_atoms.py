import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from polyatomic_complexes.src.complexes.atomic_complex import AtomComplex
from typing import List
import pytest
import json
import random
from pathlib import Path

cases = [
    (AtomComplex(1, 1, 1, 5, 3, 3, 0), 1, 1, 1),
    (AtomComplex(2, 1, 2, 5, 3, 3, 0), 2, 1, 2),
    (AtomComplex(1, 0, 1, 1, [1, 3], [1, 6], 9), 1, 0, 1),
    (AtomComplex(0, 12, 12, 5, [1, 3], [1, 3], 0), 0, 12, 12),
    (AtomComplex(1, 12, 0, 5, [1, 3], [1, 3], 0), 1, 12, 0),
    (AtomComplex(12, 1, 2, 17, 9, 9, 0), 12, 1, 2),
]

root_data = Path(__file__).parent.parent.__str__()


def fuzz_test(n=50):
    with open(root_data + "/dataset/construct/lookup_map.json") as data:
        lookup = json.load(data)
    assert isinstance(lookup, dict)
    items = random.sample(list(lookup.items()), n)
    for k, it in items:
        p, n, e = it
        a = AtomComplex(p, n, e, 5, 3, 3, 0)
        case = (a, p, n, e)
        cases.append(case)


fuzz_test()


@pytest.mark.parametrize(
    "atom,ans_p,ans_n,ans_e",
    cases,
)
def test_build(
    atom: AtomComplex,
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
