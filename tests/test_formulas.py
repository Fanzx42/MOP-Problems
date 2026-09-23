"""Analytical anchors and Pareto-set identities independent of the implementation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mop_problems import get_problem, list_problems


@pytest.mark.parametrize(
    "name,expected",
    [
        ("ZDT1", [0.25, 0.5]),
        ("ZDT2", [0.25, 0.9375]),
        ("ZDT3", [0.25, 0.25]),
        ("ZDT4", [0.25, 0.5]),
        ("ZDT6", [1.0, 0.0]),
    ],
)
def test_zdt_analytical_front_anchors(name, expected):
    p = get_problem(name)
    X = np.zeros(p.n_var)
    X[0] = 0 if name == "ZDT6" else 0.25
    assert_allclose(p.evaluate(X).F[0], expected, atol=1e-12)


@pytest.mark.parametrize("number", range(1, 8))
@pytest.mark.parametrize("n_obj", [2, 3, 5])
def test_dtlz_distance_minima(number, n_obj):
    p = get_problem(f"DTLZ{number}", n_obj=n_obj)
    X = np.zeros((4, p.n_var))
    X[:, : n_obj - 1] = np.linspace(0, 1, 4)[:, None]
    X[:, n_obj - 1 :] = 0 if number in (6, 7) else 0.5
    F = p.evaluate(X).F
    if number == 1:
        assert_allclose(F.sum(axis=1), 0.5, atol=1e-12)
    elif number == 7:
        # With position coordinates zero and g=1, the last objective is 2*M.
        assert_allclose(F[0], [0] * (n_obj - 1) + [2 * n_obj])
    else:
        assert_allclose(np.linalg.norm(F, axis=1), 1, atol=1e-12)


def test_dtlz_dimension_dependent_defaults():
    assert get_problem("DTLZ1", n_obj=5).n_var == 9
    assert get_problem("DTLZ2", n_obj=5).n_var == 14
    assert get_problem("DTLZ7", n_obj=5).n_var == 24


@pytest.mark.parametrize(
    "name,x,F,G",
    [
        ("SCH", [1], [1, 1], []),
        ("SCH", [0], [0, 4], []),
        ("POL", [1, 2], [1, 25], []),
        ("BNH", [1, 1], [8, 32], [-8, -57.3]),
        ("BNH", [0, 0], [0, 50], [0, -65.3]),
        ("TNK", [1, 1], [1, 1], [-0.9, 0]),
        ("TNK", [0, 0], [0, 0], [1.1, 0]),
        ("OSY", [2, 2, 3, 0, 3, 4], [-24, 42], [-2, -2, -2, -6, -4, 0]),
        ("SRN", [0, 0], [7, -1], [-225, 10]),
        ("SRN", [-2.5, 2.5], [24.5, -24.75], [-212.5, 0]),
    ],
)
def test_classic_hand_calculated_values(name, x, F, G):
    out = get_problem(name).evaluate(x)
    assert_allclose(out.F[0], F, atol=1e-12)
    assert_allclose(out.G[0], G, atol=1e-12)
    assert_allclose(out.CV, [sum(max(0, g) for g in G)], atol=1e-12)


@pytest.mark.parametrize("n", [2, 3, 11])
def test_fon_known_values(n):
    p = get_problem("FON", n_var=n)
    assert_allclose(p.evaluate(np.zeros(n)).F, [[1 - np.exp(-1)] * 2])
    assert_allclose(p.evaluate(np.full(n, 1 / np.sqrt(n))).F, [[0, 1 - np.exp(-4)]])


def test_constraint_signs_distinguish_feasible_and_infeasible():
    assert get_problem("BNH").evaluate([1, 1]).feasible[0]
    assert not get_problem("BNH").evaluate([0, 3]).feasible[0]
    assert get_problem("TNK").evaluate([1, 1]).feasible[0]
    assert not get_problem("TNK").evaluate([0.5, 0.5]).feasible[0]
    assert get_problem("OSY").evaluate([2, 2, 3, 0, 3, 4]).feasible[0]
    assert not get_problem("OSY").evaluate([0, 0, 1, 0, 1, 0]).feasible[0]
    assert get_problem("SRN").evaluate([-2.5, 2.5]).feasible[0]
    assert not get_problem("SRN").evaluate([0, 0]).feasible[0]


FRONTS = ["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6", *[f"DTLZ{i}" for i in range(1, 7)], "SCH", "FON"]


@pytest.mark.parametrize("name", FRONTS)
def test_reference_front_shape_finiteness_and_reproducibility(name):
    p = get_problem(name)
    for n in [2, 7, 101]:
        F = p.pareto_front(n, seed=19)
        assert F.shape == (n, p.n_obj)
        assert np.isfinite(F).all()
        assert_array_equal(F, p.pareto_front(n, seed=19))
    with pytest.raises(ValueError):
        p.pareto_front(1)


def test_disconnected_zdt3_front_has_no_dominated_bridges():
    F = get_problem("ZDT3").pareto_front(201)
    for i, point in enumerate(F):
        dominated = np.all(F <= point + 1e-9, axis=1) & np.any(F < point - 1e-8, axis=1)
        dominated[i] = False
        assert not dominated.any()
    assert np.count_nonzero(np.diff(F[:, 0]) > 0.05) == 4


def test_zdt6_front_starts_at_true_minimum():
    F = get_problem("ZDT6").pareto_front(100)
    assert_allclose(F[0, 0], 0.2807753191, atol=1e-10)
    assert_allclose(F[:, 1], 1 - F[:, 0] ** 2)


@pytest.mark.parametrize("name", sorted(set(list_problems()) - set(FRONTS)))
def test_unsupported_fronts_raise_instead_of_fabricating_data(name):
    with pytest.raises(NotImplementedError):
        get_problem(name).pareto_front()
