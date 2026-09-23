"""Mathematical oracles, regression trajectories, budget and failure semantics."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mop_problems import Problem, get_problem
from mop_problems.algorithms import (
    BBDMO,
    GD,
    L1,
    MGDA,
    NesterovAcceleratedProximalGradient,
    ProximalGradient,
    SolverOptions,
    get_algorithm,
    list_algorithms,
    minimize,
    multistart,
    nondominated_mask,
)
from mop_problems.algorithms._subproblem import SubproblemError, minimax_step
from mop_problems.algorithms.first_order import bb_scales
from mop_problems.algorithms.proximal import Zero


class Quadratics(Problem):
    def __init__(self, scales=(1.0, 1.0), lower=-10, upper=10):
        super().__init__("quadratics", 2, 2, lower, upper)
        self.scales = np.array(scales)

    def _evaluate(self, x):
        return (
            np.column_stack((np.sum((x - 1) ** 2, axis=1), np.sum((x + 1) ** 2, axis=1)))
            * self.scales,
            None,
            None,
        )


def test_mgda_direction_matches_closed_form_two_objective_projection():
    J = np.array([[3, 1, 2], [1, 4, -1.0]])
    difference = J[0] - J[1]
    weight = np.clip(-J[1] @ difference / (difference @ difference), 0, 1)
    expected = -(weight * J[0] + (1 - weight) * J[1])
    result = minimax_step(
        np.zeros(3), J, np.zeros(2), Zero(2), 1, np.full(3, -np.inf), np.full(3, np.inf)
    )
    assert_allclose(result.x, expected, atol=1e-10)
    assert_allclose(result.weights, [weight, 1 - weight], atol=1e-10)
    assert np.all(J @ result.x < 0)
    assert result.gap <= 1e-11


def test_simplex_solver_three_objectives_with_known_interior_optimum():
    pytest.importorskip("scipy")
    J = np.diag([1.0, 2.0, 3.0])
    weights = 1 / np.diag(J) ** 2
    weights /= weights.sum()
    result = minimax_step(
        np.zeros(3), J, np.zeros(3), Zero(3), 1, np.full(3, -np.inf), np.full(3, np.inf)
    )
    assert_allclose(result.weights, weights, atol=1e-7)
    assert_allclose(result.x, -weights @ J, atol=1e-7)
    assert result.gap < 1e-10


def test_box_is_inside_subproblem_and_changes_optimal_weights():
    # The minimax model incorporates feasibility before selecting the mixture.
    J = np.array([[1.0, 1.0], [-1.0, 3.0]])
    result = minimax_step(
        np.zeros(2), J, np.zeros(2), Zero(2), 1, np.array([0, -10.0]), np.array([10, 10.0])
    )
    assert_allclose(result.x, [0, -1], atol=1e-9)
    assert_allclose(result.weights, [1, 0], atol=1e-9)


@pytest.mark.parametrize("name", list_algorithms())
def test_all_baselines_stationarity_and_nontrivial_common_descent(name):
    p = Quadratics()
    x0 = np.array([3.0, -2.0])
    before = x0.copy()
    result = minimize(p, x0, algorithm=name, step_size=0.2, max_iter=150)
    assert result.success, result.message
    assert result.stationarity_bound <= 1e-5
    assert np.all(result.fun < p.evaluate(x0).F[0])
    assert_allclose(result.x[0], result.x[1], atol=3e-5)
    assert -1.00001 <= result.x[0] <= 1.00001
    assert result.history[-1].iteration == result.n_iter
    assert_allclose(result.history[-1].x, result.x)
    assert result.nfev >= result.njev >= 1
    assert result.nprox >= result.nsubproblem >= 1
    assert_array_equal(x0, before)


def test_mgda_fixed_step_hand_trajectory_and_armijo_inequality():
    p = Quadratics()
    r = GD(step_rule="fixed", step_size=0.1, max_iter=1).solve(p, [3, -3])
    assert_allclose(r.x, [2.4, -2.4], atol=1e-12)
    assert r.status == "max_iter" and not r.success
    r = MGDA(armijo=0.1, step_size=1.0).solve(p, [5, 3])
    for old, new in zip(r.history, r.history[1:], strict=False):
        slope = p.jacobian(old.x).dF[0] @ (new.x - old.x)
        assert np.all(new.fun <= old.fun + 0.1 * slope + 1e-11)
    assert any(h.step_size < 1 for h in r.history[1:])


@pytest.mark.parametrize("rule", ["fixed", "armijo", "nonmonotone", "nonmonotone_average"])
def test_gradient_step_rules_are_explicit_and_reproducible(rule):
    method = BBDMO(step_rule=rule, step_size=0.1, max_iter=20)
    a = method.solve(Quadratics(scales=(1, 20)), [4, 3])
    b = method.solve(Quadratics(scales=(1, 20)), [4, 3])
    assert a.options["step_rule"] == rule
    assert_array_equal(a.x, b.x)
    assert a.status == b.status and a.nfev == b.nfev


def test_bbdmo_three_curvature_cases_and_first_step_is_mgda():
    s = np.array([1.0, 0])
    y = np.array([[2.0, 0], [-3.0, 4], [0, 2], [1e6, 0]])
    assert_allclose(bb_scales(s, y), [2, 5, 1e-3, 1e3])
    assert_array_equal(bb_scales(np.zeros(2), y), np.ones(4))
    p = Quadratics(scales=(2, 5))
    a = MGDA(max_iter=1).solve(p, [3, 4])
    b = BBDMO(max_iter=1).solve(p, [3, 4])
    assert_allclose(a.x, b.x)
    r = BBDMO(max_iter=2, step_rule="fixed", step_size=0.1).solve(p, [3, -2])
    assert_allclose(r.history[-1].bb_scales, [4, 10], atol=1e-8)


def test_l1_prox_against_independent_scalar_minimization():
    scipy = pytest.importorskip("scipy.optimize")
    penalty = L1([[0.2, 0.5], [1, 0.3], [0.6, 1.1]], [[-1, 1], [0.5, -2], [3, 0]]).bind(3, 2)
    weights, v, step = np.array([0.1, 0.3, 0.6]), np.array([-3.0, 1.7]), 0.7
    z = penalty.prox(v, step, weights, np.full(2, -2.0), np.full(2, 2.0))
    for j in range(2):

        def objective(t, j=j):
            return (t - v[j]) ** 2 / (2 * step) + np.sum(
                weights * penalty.coefficients[:, j] * np.abs(t - penalty.centres[:, j])
            )

        optimum = scipy.minimize_scalar(
            objective, method="bounded", bounds=(-2, 2), options={"xatol": 1e-13}
        )
        assert objective(z[j]) <= optimum.fun + 1e-8
    zero = L1(0).bind(2, 2)
    assert_array_equal(zero.prox(v, 1, [0.5, 0.5], [-2, -2], [2, 2]), [-2, 1.7])


@pytest.mark.parametrize("method", [ProximalGradient, NesterovAcceleratedProximalGradient])
def test_proximal_objectives_include_distinct_penalties(method):
    p = Quadratics()
    reg = L1([0.3, 0.7], [0, 0.5])
    r = method(regularizer=reg, step_size=0.2, max_iter=200).solve(p, [3, -2])
    assert r.success, r.message
    assert_allclose(r.fun, p.evaluate(r.x).F[0] + reg.bind(2, 2).values(r.x))
    assert np.any(np.abs(r.fun - r.smooth_fun) > 0)
    assert r.stationarity_bound <= 1e-5


def test_accelerated_model_includes_objective_difference_offsets():
    # Independent epigraph optimization distinguishes the TFY model from the
    # tempting but incorrect extrapolate-then-MGDA implementation.
    scipy = pytest.importorskip("scipy.optimize")
    y, J, offsets = np.array([0.8, -0.4]), np.array([[2.0, 1], [-1, 3]]), np.array([3.0, -2.0])
    result = minimax_step(y, J, offsets, Zero(2), 0.2, [-10, -10], [10, 10])

    def objective(zt):
        d = zt[:2] - y
        return zt[-1] + d @ d / 0.4

    oracle = scipy.minimize(
        objective,
        np.r_[y, 3.0],
        method="SLSQP",
        constraints={"type": "ineq", "fun": lambda zt: zt[-1] - J @ (zt[:2] - y) - offsets},
        options={"ftol": 1e-13},
    )
    assert oracle.success
    assert_allclose(result.x, oracle.x[:2], atol=2e-7)
    naive = minimax_step(y, J, np.zeros(2), Zero(2), 0.2, [-10, -10], [10, 10])
    assert np.linalg.norm(result.x - naive.x) > 0.1


def test_nesterov_recurrence_matches_independent_two_step_derivation():
    p = Quadratics(scales=(1, 3))
    step = 0.05
    r = minimize(p, [4, -2], algorithm="nesterov", step_rule="fixed", step_size=step, max_iter=3)
    x, y, t = np.array([4.0, -2]), np.array([4.0, -2]), 1.0
    for k in range(3):
        J = np.stack((2 * (y - 1), 6 * (y + 1)))
        offset = p.evaluate(y).F[0] - p.evaluate(x).F[0]
        z = minimax_step(y, J, offset, Zero(2), step, p.xl, p.xu).x
        next_t = (1 + np.sqrt(1 + 4 * t * t)) / 2
        y, x, t = z + (t - 1) / next_t * (z - x), z, next_t
        assert_allclose(r.history[k + 1].x, x, atol=1e-10)


@pytest.mark.parametrize(
    "kwargs,status",
    [
        ({"max_nfev": 1}, "max_nfev"),
        ({"max_nfev": 3}, "max_nfev"),
        ({"max_njev": 1}, "max_njev"),
        ({"max_iter": 0}, "max_iter"),
        ({"max_line_search": 1, "step_size": 100}, "line_search_failed"),
    ],
)
def test_budgets_and_failure_are_not_success(kwargs, status):
    r = MGDA(**kwargs).solve(Quadratics(), [4, 3])
    assert r.status == status and not r.success
    assert r.nfev <= kwargs.get("max_nfev", 10000)
    assert r.njev <= kwargs.get("max_njev", 2000)
    assert_allclose(r.history[-1].x, r.x)
    assert_allclose(r.fun, Quadratics().evaluate(r.x).F[0])
    if r.stationarity is None:
        assert r.stationarity_bound is None


def test_nonlinear_constraints_and_nonsmooth_failures_are_explicit():
    with pytest.raises(ValueError, match="nonlinear"):
        MGDA().solve(get_problem("BNH"), [1, 1])
    result = MGDA().solve(get_problem("ZDT1", n_var=2), [0, 0.5])
    assert not result.success and result.status == "nondifferentiable"
    with pytest.raises(ValueError):
        MGDA(regularizer=L1())
    with pytest.raises(ValueError):
        ProximalGradient(step_rule="armijo")
    with pytest.raises(ValueError):
        MGDA().solve(Quadratics(), [[1, 1]])


def test_insufficient_inner_solve_cannot_pass_as_stationarity():
    r = MGDA(subproblem_max_iter=1, subproblem_tol=1e-16).solve(Quadratics(), [3, -2])
    assert not r.success and r.status == "subproblem_failed"
    assert r.nprox > 0
    with pytest.raises(SubproblemError):
        minimax_step(
            np.zeros(2),
            np.array([[1, 1], [-1, 2]]),
            np.zeros(2),
            Zero(2),
            1,
            [-10, -10],
            [10, 10],
            max_iter=1,
            tol=1e-16,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tol": 0},
        {"tol": np.inf},
        {"step_size": -1},
        {"step_size": True},
        {"armijo": 1},
        {"contraction": 0},
        {"average_eta": 2},
        {"max_iter": -1},
        {"max_nfev": 1.5},
        {"bb_min": 5, "bb_max": 1},
        {"bounds": "clip"},
        {"step_rule": "magic"},
        {"domain_restart": 1},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        SolverOptions(**kwargs)


def test_multistart_pairing_aliases_and_dominance():
    p = Quadratics()
    starts = np.array([[4, -3], [-3, 4]])
    a = multistart(p, MGDA(step_size=0.1), starts=starts)
    b = multistart(p, get_algorithm("gd", step_size=0.1), starts=starts)
    for ra, rb in zip(a, b, strict=True):
        assert_array_equal(ra.x, rb.x)
    assert_array_equal(nondominated_mask([[1, 2], [2, 1], [3, 3], [1, 2]]), [1, 1, 0, 1])
    assert nondominated_mask(np.empty((0, 2))).shape == (0,)
    with pytest.raises(KeyError):
        get_algorithm("invented")


def test_joint_and_value_traversals_are_counted_exactly():
    class Counted(Quadratics):
        calls = 0

        def _evaluate(self, x):
            self.calls += 1
            return super()._evaluate(x)

    p = Counted()
    r = MGDA(step_size=1).solve(p, [5, 3])
    assert p.calls == r.nfev
    assert r.nfev > r.njev
    assert r.nfev >= 2 * r.njev


def test_projected_stationarity_and_explicit_unbounded_extension():
    class Linear(Problem):
        def __init__(self):
            super().__init__("linear", 1, 2, -1, 1)

        def _evaluate(self, x):
            return np.column_stack((x[:, 0], 2 * x[:, 0])), None, None

    p = Linear()
    boxed = MGDA().solve(p, [0.5])
    assert boxed.success and boxed.feasible
    assert_allclose(boxed.x, [-1])
    unbounded = MGDA(bounds="none", max_iter=3, store_history=False).solve(p, [0.5])
    assert not unbounded.success and not unbounded.feasible
    assert unbounded.history == ()
    assert unbounded.stationarity > 0.9


def test_acceleration_restarts_at_domain_boundary():
    method = get_algorithm("nag", step_rule="fixed", step_size=2, max_iter=5)
    r = method.solve(Quadratics(lower=-1, upper=1), [0.9, -0.9])
    assert r.domain_restarts > 0
    assert all(np.all(np.abs(h.x) <= 1) for h in r.history)
    assert r.status == "max_iter"


def test_three_objective_shifted_l1_inner_accuracy_regression():
    pytest.importorskip("scipy")
    p = get_problem("DTLZ2", n_var=8, n_obj=3)
    reg = L1([0.05, 0.1, 0.15], [0.0, 0.3, 0.7])
    x = p.sample(1, seed=42)[0]
    for method in (ProximalGradient, NesterovAcceleratedProximalGradient):
        result = method(regularizer=reg, max_iter=150).solve(p, x)
        assert result.success, result.message
        assert result.stationarity_bound <= 1e-5


@pytest.mark.parametrize("reg", [L1(-1), L1([1, 2, 3]), L1(np.inf), L1(centres=1j)])
def test_invalid_l1_parameters_are_rejected(reg):
    with pytest.raises(ValueError):
        ProximalGradient(regularizer=reg).solve(Quadratics(), [0, 0])
