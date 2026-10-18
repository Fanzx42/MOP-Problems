"""Check application-inspired affine splits; no optimizer or real dataset is used.

Run from the repository root:
    python examples/admm_application_splits.py

See docs/admm-applications.zh-CN.md for source models and the distinctions between
original formulations, catalogue vectorizations, and synthetic fixtures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from numpy.testing import assert_allclose


@dataclass(frozen=True)
class Fixture:
    """An explicit point, objective decomposition, and affine coupling."""

    name: str
    catalogue_id: str
    x: np.ndarray
    z: np.ndarray
    A: np.ndarray
    B: np.ndarray
    c: np.ndarray
    original: np.ndarray
    f: np.ndarray
    g: np.ndarray
    expected: np.ndarray


def tv_fixture() -> Fixture:
    """B02: one-dimensional, forward differences with no wraparound."""
    x = np.array([0.0, 2.0, 1.0])
    b = np.ones(3)
    D = np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]])
    z = D @ x
    fidelity = 0.5 * np.sum((x - b) ** 2)
    original = np.array([fidelity, np.abs(np.diff(x)).sum()])
    return Fixture(
        "tv",
        "B02",
        x,
        z,
        D,
        -np.eye(2),
        np.zeros(2),
        original,
        np.array([fidelity, 0.0]),
        np.array([0.0, np.abs(z).sum()]),
        # (1/2) * (1 + 1 + 0) = 1; |2 - 0| + |1 - 2| = 3.
        np.array([1.0, 3.0]),
    )


def rpca_fixture() -> Fixture:
    """B03: nuclear norm is singular-value sum, L1 is entrywise."""
    L = np.array([[3.0, 0.0], [0.0, 0.0]])
    S = np.array([[0.0, -2.0], [0.0, 0.0]])
    M = np.array([[3.0, -2.0], [0.0, 0.0]])
    x, z = L.ravel(), S.ravel()
    nuclear = np.linalg.svd(L, compute_uv=False).sum()
    sparse = np.abs(S).sum()
    return Fixture(
        "rpca",
        "B03",
        x,
        z,
        np.eye(4),
        np.eye(4),
        M.ravel(),
        np.array([nuclear, sparse]),
        np.array([nuclear, 0.0]),
        np.array([0.0, sparse]),
        # L has singular values (3, 0); S has one entry of magnitude 2.
        np.array([3.0, 2.0]),
    )


def unmixing_fixture() -> Fixture:
    """B04: fixed dictionary, nonnegative abundances, no sum-to-one.

    Two pixels on a one-dimensional open chain; anisotropic TV.
    This boundary choice differs from the cyclic two-dimensional reference code.
    Flattening is row-major for both U and the derivative operator.
    """
    H = np.diag([1.0, 2.0])
    U = np.array([[1.0, 0.0], [0.0, 2.0]])
    Y = np.array([[1.0, 1.0], [0.0, 2.0]])
    x = U.ravel()
    D = np.kron(np.eye(2), np.array([[-1.0, 1.0]]))
    A = np.vstack((np.eye(4), D))
    # Form the auxiliary variables directly from physical values, not via A.
    z = np.concatenate((x, np.diff(U, axis=1).ravel()))
    fidelity = 0.5 * np.sum((H @ U - Y) ** 2)
    original = np.array([fidelity, np.abs(U).sum(), np.abs(np.diff(U, axis=1)).sum()])
    if not np.all(U >= 0):
        raise AssertionError("Abundance fixture violates its local domain.")
    return Fixture(
        "unmixing",
        "B04",
        x,
        z,
        A,
        -np.eye(6),
        np.zeros(6),
        original,
        np.array([fidelity, 0.0, 0.0]),
        np.array([0.0, np.abs(z[:4]).sum(), np.abs(z[4:]).sum()]),
        # HU-Y = [[0,-1],[0,2]]; L1(U)=3; differences are (-1,2).
        np.array([2.5, 3.0, 3.0]),
    )


def portfolio_fixture() -> Fixture:
    """B05: simplified risk/return/turnover, with a hard local budget."""
    h = np.array([0.5, 0.5])
    x = np.array([0.25, 0.75])
    z = x - h
    covariance = np.diag([2.0, 1.0])
    mean = np.array([0.1, 0.2])
    risk = x @ covariance @ x
    negative_return = -mean @ x
    original = np.array([risk, negative_return, np.abs(x - h).sum()])
    assert_allclose(x.sum(), 1.0, rtol=0, atol=1e-14)
    if not np.all(x >= 0):
        raise AssertionError("Portfolio fixture violates its local domain.")
    return Fixture(
        "portfolio",
        "B05",
        x,
        z,
        np.eye(2),
        -np.eye(2),
        h,
        original,
        np.array([risk, negative_return, 0.0]),
        np.array([0.0, 0.0, np.abs(z).sum()]),
        # 2*(1/4)^2+(3/4)^2=11/16; return=7/40; turnover=1/2.
        np.array([11.0 / 16.0, -7.0 / 40.0, 0.5]),
    )


def check_fixture(case: Fixture) -> dict:
    """Compare against hand values, then check feasible/infeasible residuals."""
    split = case.f + case.g
    assert_allclose(case.original, case.expected, rtol=0, atol=1e-14)
    assert_allclose(split, case.expected, rtol=0, atol=1e-14)
    residual = case.A @ case.x + case.B @ case.z - case.c
    assert_allclose(residual, 0.0, rtol=0, atol=1e-14)

    perturbed_z = case.z.copy()
    perturbed_z[0] += 0.25
    bad_residual = case.A @ case.x + case.B @ perturbed_z - case.c
    # Every fixture has B = +/-I; a single-coordinate perturbation has norm 1/4.
    assert_allclose(np.linalg.norm(bad_residual), 0.25, rtol=0, atol=1e-14)
    return {
        "name": case.name,
        "catalogue_id": case.catalogue_id,
        "synthetic": True,
        "n_x": case.x.size,
        "n_z": case.z.size,
        "n_equalities": case.c.size,
        "objectives": split.tolist(),
        "split_error": float(np.max(np.abs(case.original - split))),
        "residual": float(np.linalg.norm(residual)),
        "perturbed_residual": float(np.linalg.norm(bad_residual)),
    }


def main() -> None:
    cases = [tv_fixture(), rpca_fixture(), unmixing_fixture(), portfolio_fixture()]
    results = [check_fixture(case) for case in cases]
    # NumPy's matrix ord=1 would give 2 here; the entrywise L1 is 4.
    matrix = np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert_allclose(np.abs(matrix).sum(), 4.0, rtol=0, atol=1e-14)
    assert_allclose(np.linalg.norm(matrix, ord=1), 2.0, rtol=0, atol=1e-14)
    # Demonstrate why nonnegative per-pixel sum-to-one trivializes abundance L1.
    simplex_abundances = np.array([[0.2, 0.7], [0.8, 0.3]])
    assert_allclose(simplex_abundances.sum(axis=0), 1.0, rtol=0, atol=1e-14)
    assert_allclose(np.abs(simplex_abundances).sum(), 2.0, rtol=0, atol=1e-14)
    print(json.dumps({"construction_checks": results}, indent=2))


if __name__ == "__main__":
    main()
