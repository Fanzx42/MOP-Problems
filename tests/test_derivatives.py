"""Independent difference, closed-form and exceptional-point derivative checks."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mop_problems import (
    NonDifferentiableError,
    Problem,
    check_derivatives,
    finite_difference,
    get_problem,
    list_problems,
)


@pytest.mark.parametrize("name", list_problems())
@pytest.mark.parametrize("seed", [238, 714])
def test_all_problem_derivatives_against_independent_differences(name, seed):
    p = get_problem(name)
    x = p.sample(4, seed=seed)
    if name in ("WFG9", "MaF12", "DASCMaOP9"):
        # Extreme parameter bias can collapse different transformed coordinates
        # to a floating-point tie. Validate regular branches without that loss.
        x = p.xl + (0.35 + 0.3 * (x - p.xl) / (p.xu - p.xl)) * (p.xu - p.xl)
    if name == "C1-DTLZ3":
        # G is quartic in the highly multimodal radius. At arbitrary points its
        # value can exceed 1e12, making position differences pure cancellation.
        # Here independently check all entries at a better-conditioned radius.
        x[:, p.n_obj - 1 :] = 0.5 + 1e-4 * (x[:, p.n_obj - 1 :] - 0.5)
    if name.startswith("NCTP"):
        # Rosenbrock tails near one keep the rotated sine resolvable by differences.
        x[:, 1:] = 1 + 0.01 * (x[:, 1:] - 2.5)
    if name in ("LSMOP6", "LSMOP7"):
        coefficient = 1 + np.cos(np.pi * np.arange(p.n_obj, p.n_var + 1) / (2 * p.n_var))
        x[:, 0] *= 0.8
        x[:, p.n_obj - 1 :] = (10 * x[:, 0, None] + 0.01 * x[:, p.n_obj - 1 :]) / coefficient
    if name.startswith("LSCMOP"):
        # The CEC2006 g6 cubic can dominate other objective terms by 1e6.
        # Exercise nonstationary but well-conditioned points near its small end.
        for number, cs, *_ in p._groups:
            if number == 6:
                desired = np.array([0.02, 0.02])
                control = x[:, 0, None]
                linked = (1 - desired) / 2 if p._config[2] == 1 else np.arccos(desired) / np.pi
                x[:, cs] = np.mod(linked - control, 0.5)
    if name == "CF3":
        x[:, 1:] *= 0.1
    if name == "MW13":
        x[:, 1:] = 0.5 + 0.01 * x[:, 1:]
    if name == "DOC4":
        # CEC2006 g9 optimum neighbourhood avoids aliasing in objective-space sin.
        optimum = np.array([2.330499, 1.951372, -0.477541, 4.365726, -0.624487, 1.038131, 1.594227])
        x[:, 1:] = optimum + 0.001 * x[:, 1:]
    if name == "DOC5":
        # Stay away from equality-band abs switches rather than straddle a kink.
        x[:, 5:8] = [6.4, 6.0, 5.0]
    before = x.copy()
    ev, jac = p.evaluate_with_jacobian(x)
    ordinary = p.evaluate(x)
    for label in ("F", "G", "H"):
        assert_allclose(getattr(ev, label), getattr(ordinary, label), rtol=2e-14, atol=2e-12)
        a = getattr(jac, "d" + label)
        assert a.shape == (len(x), getattr(ev, label).shape[1], p.n_var)
        assert not a.flags.writeable
    assert_array_equal(x, before)
    assert jac.regular.all() and not jac.regular.flags.writeable
    # DOC5/7's oscillatory constraints require a finer perturbation.
    step = 1e-8 if name == "DOC5" else 1e-7 if name == "DOC7" else 5e-7
    report = check_derivatives(p, x, rtol=3e-4, atol=3e-6, relative_step=step)
    assert report.passed, report
    assert report.finite_difference_evaluations == len(x) * (1 + 2 * p.n_var)
    one = p.jacobian(x[0])
    assert_allclose(one.dF[0], jac.dF[0], atol=1e-10)
    empty = p.jacobian(np.empty((0, p.n_var)))
    assert empty.dF.shape == (0, p.n_obj, p.n_var)


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("DTLZ1", {"n_obj": 5, "n_var": 8}),
        ("DTLZ4", {"n_obj": 2, "alpha": 0.7}),
        ("DTLZ5", {"n_obj": 8, "n_var": 9}),
        ("DTLZ6", {"n_obj": 5}),
        ("WFG1", {"n_obj": 2, "n_var": 2, "k": 1}),
        ("WFG2", {"n_obj": 5, "n_var": 10, "k": 4}),
        ("WFG9", {"n_obj": 8, "n_var": 18, "k": 14}),
        ("UF3", {"n_var": 3}),
        ("UF8", {"n_var": 5}),
        ("C3-DTLZ4", {"n_obj": 5, "alpha": 2}),
        ("SYM-PART-Rotated", {"angle": 0.31}),
    ],
)
def test_changed_dimensions_and_parameters(name, kwargs):
    p = get_problem(name, **kwargs)
    report = check_derivatives(p, p.sample(4, seed=192), rtol=2e-3, atol=1e-5, relative_step=2e-7)
    assert report.passed, report


def test_closed_form_objective_and_raw_constraint_derivatives():
    p = get_problem("FON", n_var=4)
    x = np.array([0.2, -0.1, 0.3, -0.2])
    differences = np.stack((x - 0.5, x + 0.5))
    expected = 2 * differences * np.exp(-np.sum(differences**2, axis=1))[:, None]
    assert_allclose(p.jacobian(x).dF[0], expected, atol=1e-15)
    p = get_problem("BNH")
    jac = p.jacobian([2, 1])
    assert_array_equal(jac.dF[0], [[16, 8], [-6, -8]])
    assert_array_equal(jac.dG[0], [[-6, 2], [12, -8]])
    p = get_problem("Omni-test", n_var=4)
    assert_allclose(
        p.jacobian(x + 1).dF[0], [np.pi * np.cos(np.pi * (x + 1)), -np.pi * np.sin(np.pi * (x + 1))]
    )


def test_product_derivatives_at_zero_and_dtlz_axis():
    p = get_problem("DTLZ1", n_obj=3)
    x = np.full(p.n_var, 0.5)
    x[:2] = 0
    assert_allclose(p.jacobian(x).dF[0, :, :2], [[0, 0], [0.5, 0], [-0.5, 0]])
    assert_allclose(p.jacobian(x).dF[0, :, 2:], 0)
    p = get_problem("DTLZ2")
    x = np.full(p.n_var, 0.5)
    x[:2] = 0
    assert_allclose(p.jacobian(x).dF[0, :, :2], [[0, 0], [0, np.pi / 2], [np.pi / 2, 0]])


@pytest.mark.parametrize(
    "name,point",
    [
        ("ZDT1", [0, 0.5]),
        ("ZDT6", [0.2, 0]),
        ("DTLZ6", [0.2, 0]),
        ("TNK", [0, 0]),
        ("UF7", [0, 0.4, 0.3]),
    ],
)
def test_singular_derivatives_never_silently_regularized(name, point):
    kwargs = {} if name == "TNK" else {"n_var": len(point)}
    if name == "DTLZ6":
        kwargs["n_obj"] = 2
    p = get_problem(name, **kwargs)
    for policy in ("raise", "branch"):
        with pytest.raises(NonDifferentiableError, match="singular"):
            p.jacobian(point, nondifferentiable=policy)


@pytest.mark.parametrize(
    "name,point",
    [
        ("UF5", [0.5, 0.2, 0.4]),
        ("SYM-PART", [6, 1]),
        ("SYM-PART", [1, 5]),
    ],
)
def test_kinks_require_explicit_branch_derivative_opt_in(name, point):
    p = get_problem(name, **({"n_var": 3} if name == "UF5" else {}))
    with pytest.raises(NonDifferentiableError):
        p.jacobian(point)
    branch = p.jacobian(point, nondifferentiable="branch")
    assert not branch.regular[0]
    assert np.isfinite(branch.dF).all()


def test_numerical_derivatives_at_bounds_and_equality_contract():
    class Equalities(Problem):
        def __init__(self):
            super().__init__("equalities", 2, 2, -2, 2, n_eq_constr=1)

        def _evaluate(self, x):
            return np.column_stack((x[:, 0] ** 2, x[:, 1] ** 2)), None, (x.sum(axis=1) - 1)[:, None]

    p = Equalities()
    x = np.array([[-2, 2], [0.3, -0.2]])
    exact, difference = p.jacobian(x), finite_difference(p, x)
    assert_allclose(exact.dH, np.ones((2, 1, 2)))
    assert_allclose(exact.dF, difference.dF, atol=1e-9)
    assert_allclose(exact.dH, difference.dH, atol=1e-9)
    assert difference.method == "finite_difference" and difference.regular is None
    assert p.derivative_info["smoothness"] == "unknown"


@pytest.mark.parametrize(
    "name,kind",
    [
        ("SCH", "smooth"),
        ("ZDT1", "interior"),
        ("WFG9", "piecewise"),
        ("C2-DTLZ2", "piecewise"),
        ("SYM-PART", "piecewise_discontinuous"),
    ],
)
def test_smoothness_metadata(name, kind):
    assert get_problem(name).derivative_info["smoothness"] == kind
    assert get_problem("DTLZ4", alpha=0.1).derivative_info["smoothness"] == "interior"


def test_derivative_validation_and_wrong_shape():
    p = get_problem("SCH")
    for x in ([np.inf], [1j], [[1, 2]], [1001]):
        with pytest.raises((ValueError, TypeError)):
            p.jacobian(x)
    with pytest.raises(ValueError):
        p.jacobian([1], nondifferentiable="guess")
    for h in (0, -1, np.inf):
        with pytest.raises(ValueError):
            finite_difference(p, [1], relative_step=h)
    for kwargs in ({"rtol": -1}, {"atol": -1}, {"rtol": 0, "atol": 0}):
        with pytest.raises(ValueError):
            check_derivatives(p, [1], **kwargs)


def test_chain_rule_zero_powers_and_batch_local_regularity():
    from mop_problems._autodiff import Jet

    x = Jet.variables(np.array([[0.0, 2.0], [1.0, 3.0]]))
    with np.errstate(all="ignore"):
        powered = x[:, 0] ** x[:, 1]
    assert_allclose(powered.d, [[0, 0], [3, 0]])
    x.r[0, 0] = False
    product = x @ np.ones(2)
    assert_array_equal(product.r, [False, True])
    assert_allclose(x.sum(axis=-1).d, [[1, 1], [1, 1]])


def test_sqrt_of_squared_variable_does_not_silently_hide_a_kink():
    class RootSquare(Problem):
        def __init__(self):
            super().__init__("root-square", 1, 2, -1, 1)

        def _evaluate(self, x):
            return np.column_stack((np.sqrt(x[:, 0] ** 2), x[:, 0] ** 2)), None, None

    with pytest.raises(NonDifferentiableError):
        RootSquare().jacobian([0])
