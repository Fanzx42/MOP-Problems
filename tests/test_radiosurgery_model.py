# ruff: noqa: E402 -- the application has an optional SciPy dependency
"""Physical/LP checks that run without a COPT installation or license."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("scipy")
from scipy.optimize import linprog

from mop_problems.applications.radiosurgery.model import Case, make_case, weight_grid


def test_four_objectives_against_hand_computed_doses():
    case = Case(
        problem=None,
        K=np.array([[0.4], [0.6], [0.8], [0.4], [0.1]]),
        groups={"TS": [0], "TI": [1], "R": [2], "LD": [3], "O": [4]},
        thresholds={"TS": 1.0, "TI": 1.0, "R": 1.0, "LD": 0.5, "O": 0.35},
        C=np.zeros((1, 1)),
        T=np.ones((1, 1)),
        dual_A=np.zeros((0, 0)),
        dual_c=np.zeros(0),
        seed=0,
    )
    # Dose=(.8,1.2,1.6,.8,.2); the low-dose threshold .5 changes its scaling.
    assert_allclose(case.objectives(np.array([2.0, 0.7])), [0.2, 0.6, 0.6, 0.7], atol=1e-14)


@pytest.mark.parametrize("isocenters", [1, 2])
def test_physical_primal_and_explicit_dual_lp_agree_without_copt(isocenters):
    case = make_case(11, isocenters)
    for weight in weight_grid()[[0, 4, 8]]:
        primal = case.lp(weight)
        rhs, upper = case.dual_bounds(weight)
        dual = linprog(
            case.dual_c,
            A_ub=case.dual_A,
            b_ub=rhs,
            bounds=list(zip(np.zeros(len(upper)), upper, strict=True)),
            method="highs-ds",
        )
        assert dual.success
        assert abs(primal["value"] + dual.fun) < 2e-8
        x = primal["x"]
        z = case.problem.A @ x
        assert_allclose(case.problem.values(x, z), case.objectives(x), atol=1e-12)
        assert case.problem.violation(x, z) < 1e-8


def test_archived_evidence_hashes_and_state_shapes():
    evidence = Path(__file__).resolve().parents[1] / "docs/experiments/radiosurgery"
    provenance = json.loads((evidence / "provenance.json").read_text(encoding="utf-8"))
    for name, count in (("numerics", 444), ("three-methods", 351)):
        directory = evidence / name
        rows = json.loads((directory / "runs.json").read_text(encoding="utf-8"))
        assert len(rows) == count
        assert len({r["id"] for r in rows}) == count
        for filename, digest in provenance[name]["files"].items():
            assert hashlib.sha256((directory / filename).read_bytes()).hexdigest() == digest
        with np.load(directory / "states.npz", allow_pickle=False) as state:
            assert state["x"].shape == (count, 25)
            assert state["z"].shape == (count, 32)
            assert state["w"].shape == (count, 4)
            assert np.isfinite(state["x"]).all()
            assert_allclose(state["w"].sum(axis=1), 1.0, atol=1e-12)
