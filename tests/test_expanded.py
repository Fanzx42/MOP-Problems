"""Off-manifold UF checks and independently known identities for new families."""

import math
from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mop_problems import get_problem


def uf_scalar(number, x):
    """Scalar oracle from CES-487, using explicit one-based groups and math.

    This is a separate formula transcription, not an executed external library.
    Its loop/index structure intentionally differs from the vectorized kernel.
    """
    n, t = len(x), float(x[0])
    pi, sin, cos, sqrt = math.pi, math.sin, math.cos, math.sqrt
    objectives = 2 if number <= 7 else 3
    errors = [[] for _ in range(objectives)]
    products = [[] for _ in range(objectives)]
    for j in range(objectives, n + 1):
        group = (j - 1) % objectives
        phase = 6 * pi * t + j * pi / n
        if number == 2:
            amplitude = 0.3 * t**2 * cos(24 * pi * t + 4 * j * pi / n) + 0.6 * t
            target = amplitude * (cos(phase) if j % 2 else sin(phase))
        elif number == 3:
            target = t ** (0.5 + 1.5 * (j - 2) / (n - 2))
        elif number >= 8:
            target = 2 * x[1] * sin(2 * pi * t + j * pi / n)
        else:
            target = sin(phase)
        y = x[j - 1] - target
        if number == 4:
            loss = abs(y) / (1 + math.exp(2 * abs(y)))
        elif number == 5:
            loss = 2 * y * y + 1 - cos(4 * pi * y)
        elif number == 10:
            loss = 4 * y * y + 1 - cos(8 * pi * y)
        else:
            loss = y * y
        errors[group].append(loss)
        products[group].append(cos(20 * pi * y / sqrt(j)))
    if number <= 3:
        front = [t, 1 - sqrt(t)]
    elif number == 4:
        front = [t, 1 - t * t]
    elif number in (5, 6):
        bump = 0.15 * abs(sin(20 * pi * t)) if number == 5 else max(0, 0.7 * sin(4 * pi * t))
        front = [t + bump, 1 - t + bump]
    elif number == 7:
        front = [t**0.2, 1 - t**0.2]
    elif number == 9:
        bump = max(0, 1.1 * (1 - 4 * (2 * t - 1) ** 2))
        front = [(bump / 2 + t) * x[1], (bump / 2 - t + 1) * x[1], 1 - x[1]]
    else:
        front = [
            cos(pi * t / 2) * cos(pi * x[1] / 2),
            cos(pi * t / 2) * sin(pi * x[1] / 2),
            sin(pi * t / 2),
        ]
    for i, group in enumerate(errors):
        loss = 4 * sum(group) - 2 * math.prod(products[i]) + 2 if number in (3, 6) else sum(group)
        front[i] += 2 * loss / len(group)
    return front


@pytest.mark.parametrize("number", range(1, 11))
@pytest.mark.parametrize("n_var", [5, 7, 30])
def test_uf_scalar_oracle_with_uneven_index_groups(number, n_var):
    p = get_problem(f"UF{number}", n_var=n_var)
    X = np.vstack((p.sample(17, seed=803), p.xl, p.xu))
    expected = [uf_scalar(number, row) for row in X]
    assert_allclose(p.evaluate(X).F, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("number", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_uf_linkage_minima(number):
    p = get_problem(f"UF{number}")
    # t=0 makes all UF2/3 links zero; other links retain index dependence.
    X = np.zeros(p.n_var)
    start = p.n_obj
    j = np.arange(start, p.n_var + 1)
    if number not in (2, 3) and number <= 7:
        X[start - 1 :] = np.sin(j * np.pi / p.n_var)
    if number >= 8:
        X[1] = 0.5
        X[2:] = np.sin(j * np.pi / p.n_var)
        expected = [0, 0.5, 0.5] if number == 9 else [np.sqrt(0.5), np.sqrt(0.5), 0]
    else:
        expected = [0, 1]
    assert_allclose(p.evaluate(X).F[0], expected, atol=1e-12)


@pytest.mark.parametrize("number", [1, 2, 3, 4, 5, 6, 7])
def test_uf_smallest_two_objective_dimension(number):
    p = get_problem(f"UF{number}", n_var=3)
    X = p.sample(9, seed=31)
    assert_allclose(p.evaluate(X).F, [uf_scalar(number, x) for x in X], atol=1e-12)


@pytest.mark.parametrize("name", ["WFG1", "WFG2", "WFG3", "WFG4", "WFG5", "WFG6", "WFG7"])
def test_wfg_known_axis_endpoint(name):
    p = get_problem(name)
    if name == "WFG1":
        # Polynomial exponent 0.02 magnifies rounding near the distance minimum;
        # use the exactly representable lower bound for this analytical anchor.
        assert_allclose(p.evaluate(p.xl).F, [[1, 1, 7]], atol=1e-12)
        return
    normalized = np.full(p.n_var, 0.35)
    if name not in {"WFG4", "WFG5"}:
        normalized[: p.k] = 0
    assert_allclose(p.evaluate(normalized * p.xu).F, [[0, 0, 6]], atol=1e-10)


def test_wfg_minimal_valid_partition():
    p = get_problem("WFG6", n_var=2, n_obj=2, k=1)
    assert_allclose(p.evaluate([0, 1.4]).F, [[0, 4]], atol=1e-12)


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("WFG1", {"n_obj": 1}),
        ("WFG1", {"n_obj": 4, "k": 4}),
        ("WFG1", {"k": 0}),
        ("WFG1", {"n_var": 4, "k": 4}),
        ("WFG2", {"n_var": 7, "k": 4}),
        ("WFG3", {"n_var": 7, "k": 4}),
        ("UF1", {"n_var": 2}),
        ("UF8", {"n_var": 4}),
        ("Omni-test", {"n_var": 1}),
        ("C1-DTLZ1", {"n_var": 2, "n_obj": 3}),
        ("C3-DTLZ4", {"alpha": 0}),
        ("C1-DTLZ3", {"r": float("nan")}),
        ("C2-DTLZ2", {"r": 0}),
        ("SYM-PART-Rotated", {"angle": float("inf")}),
    ],
)
def test_expanded_parameter_validation(name, kwargs):
    with pytest.raises(ValueError):
        get_problem(name, **kwargs)


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("WFG1", {"k": 2.5}),
        ("UF1", {"n_var": True}),
        ("C2-DTLZ2", {"n_obj": 3.0}),
        ("Omni-test", {"n_var": 2.5}),
    ],
)
def test_expanded_integral_parameters(name, kwargs):
    with pytest.raises(TypeError):
        get_problem(name, **kwargs)


def test_constrained_dtlz_signs_and_radii():
    X = np.full(12, 0.5)
    assert_allclose(get_problem("C1-DTLZ1").evaluate(X).G, [[-1 / 12]])
    assert_allclose(get_problem("C1-DTLZ3").evaluate(X).G, [[-1200]])
    assert get_problem("C2-DTLZ2").evaluate(X).feasible[0]
    assert_array_equal(get_problem("C3-DTLZ1").evaluate(X).G, [[0.375, 0.375, 0.25]])
    X[2:] = 0.6
    assert get_problem("C3-DTLZ1").evaluate(X).feasible[0]
    assert get_problem("C1-DTLZ3", n_obj=5).r == 12.5
    assert get_problem("C1-DTLZ3", n_obj=13).r == 15
    assert get_problem("C2-DTLZ2", n_obj=2).r == 0.2
    assert get_problem("C2-DTLZ2", n_obj=5).r == 0.5
    assert get_problem("C1-DTLZ3", r=7).r == 7
    assert get_problem("C2-DTLZ2", r=0.3).r == 0.3


def test_omni_distinct_decision_sets_share_one_front():
    p = get_problem("Omni-test")
    base = np.linspace(1, 1.5, 21)[:, None]
    fronts = []
    for offsets in product([0, 2, 4], repeat=2):
        fronts.append(p.evaluate(base + offsets).F)
    for front in fronts:
        assert_allclose(front, p.pareto_front(21), atol=1e-13)
    assert_allclose(p.evaluate([1, 1]).F, [[0, -2]], atol=1e-13)


@pytest.mark.parametrize("name", ["SYM-PART", "SYM-PART-Rotated"])
def test_all_nine_sympart_sets(name):
    p = get_problem(name)
    c, s = np.cos(p.angle), np.sin(p.angle)
    rotation = np.array([[c, -s], [s, c]])
    t = np.linspace(-1, 1, 31)
    for row, col in product([-1, 0, 1], repeat=2):
        X = np.column_stack((10 * row + t, np.full(len(t), 10 * col))) @ rotation.T
        assert_allclose(p.evaluate(X).F, p.pareto_front(len(t)), atol=1e-12)


def test_welded_beam_cost_deflection_and_feasibility():
    p = get_problem("Welded-Beam")
    assert_allclose(p.evaluate([1, 1, 1, 1]).F, [[1.82636, 2.1952]])
    assert not p.evaluate(p.xl).feasible[0]
    assert p.evaluate(p.xu).feasible[0]
    assert_allclose(
        p.evaluate([1, 1, 1, 1]).G[0, 1:], [474000, 0, 6000 - 64746.022 * (1 - 0.0282346)]
    )
