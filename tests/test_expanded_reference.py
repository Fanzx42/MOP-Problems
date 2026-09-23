"""Frozen original-code oracles keep core CI independent of MATLAB and network access."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mop_problems import catalog, get_problem

DATA = json.loads((Path(__file__).parent / "data/expanded_reference.json").read_text())
CASES = DATA["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_original_reference_outputs(case):
    p = get_problem(case["name"], **case["parameters"])
    value = p.evaluate(case["X"])
    for key in ("F", "G", "H"):
        if key not in case:
            continue
        actual = getattr(value, key)
        if key == "G":
            if case["constraint_comparison"] == "violation":
                actual = np.maximum(actual, 0)
            elif case["constraint_comparison"] == "equality-band":
                actual = np.abs(value.H) - 1e-4
        assert_allclose(actual, case[key], rtol=5e-10, atol=2e-8)


def test_all_expanded_problems_have_an_independent_oracle():
    expanded = {p["name"] for p in catalog()["problems"] if "formula_anchor" in p}
    assert expanded <= {c["name"] for c in CASES}
    assert len(DATA["source_sha256"]) == 10
