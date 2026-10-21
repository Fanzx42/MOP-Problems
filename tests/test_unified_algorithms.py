"""Independent formula and KKT oracles for the unified deterministic families."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mop_problems import Problem
from mop_problems.algorithms import (
    L1,
    AugmentedLagrangian,
    ConjugateGradient,
    FrankWolfe,
    MultiobjectiveADMM,
    NesterovAcceleratedGradient,
    NesterovAcceleratedProximalGradient,
    Polytope,
    SplitProblem,
    get_algorithm,
    list_algorithms,
    minimize,
)
from mop_problems.algorithms.augmented import alm_penalty_update
from mop_problems.algorithms.conditional import coefficient_cap, simplex_minimax
from mop_problems.algorithms.conjugate import conjugate_beta
from mop_problems.algorithms.splitting import balanced_penalty, simplex_projection

pytest.importorskip("scipy")


class Quadratic(Problem):
    def __init__(self, centres=((1.0, 1.0), (-1.0, -1.0)), curvature=(2.0, 6.0)):
        self.centres = np.asarray(centres)
        self.curvature = np.asarray(curvature)
        super().__init__("quadratic", self.centres.shape[1], len(self.centres), -20, 20)

    def _evaluate(self, x):
        return (
            np.column_stack(
                [
                    0.5 * a * np.sum((x - c) ** 2, axis=1)
                    for a, c in zip(self.curvature, self.centres, strict=True)
                ]
            ),
            None,
            None,
        )


def test_registry_and_explicit_algorithm_migration():
    assert len(list_algorithms()) == 13
    assert get_algorithm("nag").name == "nesterov"
    assert get_algorithm("tfy").name == "tfy"
    assert get_algorithm("amg-qp").options.bounds == "none"
    assert get_algorithm("gd", L=5).options.step_size == 0.2
    with pytest.raises(ValueError, match="conflicting"):
        get_algorithm("gd", L=5, step_size=0.3)


@pytest.mark.parametrize("rule", ["FR", "CD", "DY", "PRP", "PRP+", "HS", "HS+", "LS", "HZ", "HZ+"])
def test_vector_cg_reduces_to_classic_scalar_beta(rule):
    g, old_g, old_d = np.array([2.0, -1.0]), np.array([3.0, 4.0]), np.array([-2.0, -3.0])
    y = g - old_g
    formulas = {
        "FR": g @ g / (old_g @ old_g),
        "CD": g @ g / (-old_g @ old_d),
        "DY": g @ g / (old_d @ y),
        "PRP": g @ y / (old_g @ old_g),
        "HS": g @ y / (old_d @ y),
        "LS": g @ y / (-old_g @ old_d),
        "HZ": (y - 2 * old_d * (y @ y) / (old_d @ y)) @ g / (old_d @ y),
    }
    expected = formulas[rule.rstrip("+")]
    if rule.endswith("+"):
        expected = max(0.0, expected)
    beta, reason = conjugate_beta(
        rule, np.tile(g, (2, 1)), np.tile(old_g, (2, 1)), -g, -old_g, old_d
    )
    assert_allclose(beta, expected, atol=1e-14)
    assert reason == ""


@pytest.mark.parametrize("rule", ["FR", "CD", "DY", "PRP", "PRP+", "HS", "HS+", "LS", "HZ", "HZ+"])
def test_cg_common_descent_restart_and_wolfe(rule):
    p = Quadratic()
    r = ConjugateGradient(beta_rule=rule, restart_interval=1, powell_restart=None).solve(p, [4, -3])
    assert r.success, r.message
    for old, new in zip(r.history, r.history[1:], strict=False):
        d = new.diagnostics["direction"]
        slope = np.max(p.jacobian(old.x).dF[0] @ d)
        assert slope < 0
        assert np.all(new.fun <= old.fun + 1e-4 * new.step_size * slope + 1e-10)
        if new.diagnostics["restart"] != "line_search":
            assert abs(np.max(p.jacobian(new.x).dF[0] @ d)) <= 0.1 * abs(slope) + 1e-10
    assert r.diagnostics["restart_counts"].get("periodic", 0) > 0


def test_cg_denominator_restart():
    J = np.ones((2, 2))
    beta, reason = conjugate_beta("DY", J, J, -J[0], -J[0], -J[0])
    assert beta == 0 and reason == "denominator"


def test_fw_requires_minimax_mixture_not_one_vertex():
    # Neither vertex descends in both objectives; their midpoint does.
    payoff = np.array([[-3.0, 1.0], [1.0, -3.0]])
    weights, primal, dual, _ = simplex_minimax(payoff)
    assert_allclose(weights, [0.5, 0.5])
    assert_allclose([primal, dual], [-1.0, -1.0])
    assert_allclose(coefficient_cap(np.array([0.2, 0.8]), np.array([-0.5, 0.5])), 0.4)


@pytest.mark.parametrize("method", ["fw", "away_fw", "pairwise_fw"])
@pytest.mark.parametrize("step_rule", ["adaptive", "armijo"])
def test_fw_variants_preserve_convex_coefficients_and_reach_known_solution(method, step_rule):
    p = Quadratic(centres=((0.2, 0.2), (0.2, 0.2)), curvature=(2.0, 2.0))
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    r = minimize(
        p,
        [0.8, 0.1],
        algorithm=method,
        domain=Polytope(vertices),
        step_rule=step_rule,
        max_iter=1000,
        max_njev=3000,
        tol=1e-5,
    )
    assert r.success, r.message
    assert_allclose(r.x, [0.2, 0.2], atol=3e-5)
    for h in r.history:
        weights = h.diagnostics["weights"]
        assert np.all(weights >= 0)
        assert_allclose(weights.sum(), 1, atol=1e-13)
        assert_allclose(weights @ vertices, h.x, atol=1e-13)
        if h.iteration:
            assert h.step_size <= h.diagnostics["step_cap"] + 1e-14


def test_fw_diminishing_schedule_and_box_oracle():
    p = Quadratic()
    r = FrankWolfe(step_rule="diminishing", max_iter=4).solve(p, [4, -3])
    assert_allclose([h.step_size for h in r.history[1:]], [1.0, 2 / 3, 0.5, 0.4])
    assert all(np.all(h.x >= p.xl) and np.all(h.x <= p.xu) for h in r.history)
    with pytest.raises(ValueError, match="explicit"):
        get_algorithm("away_fw")


def test_away_step_is_selected_and_can_drop_an_active_vertex():
    p = Quadratic(centres=((0.8, 0.8), (0.8, 0.8)), curvature=(2.0, 2.0))
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    r = get_algorithm("away_fw", domain=Polytope(V)).solve(p, [0.4, 0.4])
    assert r.success, r.message
    assert r.history[1].diagnostics["direction_type"] == "away"
    assert_allclose(r.x, [0.5, 0.5], atol=1e-10)
    assert_allclose(r.history[1].step_size, r.history[1].diagnostics["step_cap"])
    assert r.diagnostics["weights"][0] == 0


@pytest.mark.parametrize("mu", [0.0, 0.5])
def test_full_amg_qp_clock_against_independent_closed_form_projection(mu):
    p, L = Quadratic(), 8.0
    r = NesterovAcceleratedGradient(L=L, mu=mu, max_iter=4, tol=1e-10).solve(p, [4, -3])
    x = z = np.array([4.0, -3.0])
    gamma = 1.0
    assert r.n_iter == 4, r.message
    for h in r.history[1:]:
        tau = (gamma + np.sqrt(gamma**2 + 4 * L * gamma)) / (2 * L)
        y = (x + tau * z) / (1 + tau)
        J = p.curvature[:, None] * (y - p.centres)
        target = mu * (y - x) + gamma * (z - x) / tau
        difference = J[0] - J[1]
        weight = np.clip((target - J[1]) @ difference / (difference @ difference), 0, 1)
        mixture = weight * J[0] + (1 - weight) * J[1]
        z = (gamma * z + mu * tau * y - tau * mixture) / (gamma + mu * tau)
        x = (x + tau * z) / (1 + tau)
        gamma = (gamma + mu * tau) / (1 + tau)
        assert_allclose(h.x, x, atol=1e-10)
        assert_allclose(h.diagnostics["z"], z, atol=1e-10)
        assert_allclose(h.diagnostics["gamma"], gamma, atol=1e-14)


@pytest.mark.parametrize("mu", [0.0, 0.5])
def test_amg_proximal_reformulation_matches_shifted_qp_with_inactive_box(mu):
    # bounds='box' forces the independently derived proximal path even with g=0.
    p = Quadratic()
    options = dict(L=8, mu=mu, max_iter=4)
    smooth = NesterovAcceleratedGradient(**options).solve(p, [4, -3])
    composite = NesterovAcceleratedProximalGradient(bounds="box", **options).solve(p, [4, -3])
    assert smooth.n_iter == composite.n_iter == 4
    for a, b in zip(smooth.history, composite.history, strict=True):
        assert_allclose(a.x, b.x, atol=3e-8)


def test_amg_zero_regularizer_identical_and_proximal_model_independent_oracle():
    from scipy.optimize import minimize as scalar_minimize

    p, x, L = Quadratic(), np.array([4.0, -3.0]), 8.0
    a = NesterovAcceleratedGradient(L=L, max_iter=4).solve(p, x)
    b = NesterovAcceleratedProximalGradient(L=L, regularizer=L1(0), max_iter=4).solve(p, x)
    assert_array_equal(a.x, b.x)
    reg = L1([0.3, 0.7], [0.0, 0.5])
    penalty = reg.bind(2, 2)
    r = NesterovAcceleratedProximalGradient(L=L, regularizer=reg, max_iter=2).solve(p, x)
    z, gamma = x.copy(), 1.0
    for record in r.history[1:]:
        tau = (gamma + np.sqrt(gamma**2 + 4 * L * gamma)) / (2 * L)
        y = (x + tau * z) / (1 + tau)
        J = p.jacobian(y).dF[0]

        def objective(ut, y=y):
            return ut[-1] + 0.5 * L * np.sum((ut[:2] - y) ** 2)

        def constraint(ut, J=J, x=x):
            return ut[-1] - J @ (ut[:2] - x) - penalty.values(ut[:2]) + penalty.values(x)

        def constraint_jac(ut, J=J):
            derivative = penalty.coefficients * np.sign(ut[:2] - penalty.centres)
            return np.column_stack((-J - derivative, np.ones(2)))

        oracle = scalar_minimize(
            objective,
            np.r_[x, 0.0],
            method="SLSQP",
            jac=lambda ut, y=y: np.r_[L * (ut[:2] - y), 1.0],
            constraints={"type": "ineq", "fun": constraint, "jac": constraint_jac},
            options={"ftol": 1e-11, "maxiter": 500},
        )
        assert oracle.success, oracle.message
        assert_allclose(record.x, oracle.x[:2], atol=2e-6)
        z = ((1 + tau) * record.x - x) / tau
        x, gamma = record.x, gamma / (1 + tau)


class ConstrainedQuadratic(Problem):
    def __init__(self):
        super().__init__("constrained", 2, 2, -10, 10, n_ieq_constr=1, n_eq_constr=1)

    def _evaluate(self, x):
        # x=1, z>=2, unique optimum (1,2), lambda=2, nu=-1.
        f = 0.5 * np.sum(x * x, axis=1)
        return np.column_stack((f, f)), 2 - x[:, 1:2], x[:, :1] - 1


def test_alm_phr_multipliers_feasibility_and_penalty_update():
    assert alm_penalty_update(2, 0.6, 1) == 4
    assert alm_penalty_update(2, 0.4, 1) == 2
    p = ConstrainedQuadratic()
    r = AugmentedLagrangian().solve(p, [-2.0, -1.0])
    assert r.success, r.message
    assert r.feasible and r.stationarity_bound <= 1e-5
    assert_allclose(r.x, [1, 2], atol=1e-5)
    assert_allclose(r.diagnostics["inequality_multipliers"], [2], atol=2e-5)
    assert_allclose(r.diagnostics["equality_multipliers"], [-1], atol=2e-5)
    assert r.diagnostics["complementarity"] <= 1e-5
    assert all(h.diagnostics["penalty"] > 0 for h in r.history)


def test_alm_composite_kkt_and_inner_failure_are_explicit():
    p = ConstrainedQuadratic()
    r = AugmentedLagrangian(regularizer=L1(0.3)).solve(p, [-2.0, -1.0])
    assert r.success, r.message
    assert_allclose(r.x, [1.0, 2.0], atol=1e-5)
    assert_allclose(r.diagnostics["inequality_multipliers"], [2.3], atol=2e-5)
    assert_allclose(r.diagnostics["equality_multipliers"], [-1.3], atol=2e-5)
    bad = AugmentedLagrangian(max_inner=1, step_size=0.001).solve(p, [-2.0, -1.0])
    assert bad.status == "inner_iterations_limit" and not bad.success
    assert_array_equal(bad.x, [-2.0, -1.0])


def split_quadratics():
    first = Quadratic(centres=((0.0,), (0.0,)), curvature=(1.0, 1.0))
    second = Quadratic(centres=((2.0,), (2.0,)), curvature=(1.0, 1.0))
    return SplitProblem(first, second, [[1]], [[-1]], [0])


def test_admm_split_contract_known_kkt_and_unscaled_dual_with_adaptive_rho():
    p = split_quadratics()
    ev, jac = p.evaluate_with_jacobian([[0, 2], [1, 1]])
    assert_allclose(ev.F, [[0, 0], [1, 1]])
    assert_allclose(jac.dH, [[[1, -1]], [[1, -1]]])
    r = MultiobjectiveADMM(penalty=0.01).solve(p, [0.0, 2.0])
    assert r.success, r.message
    assert_allclose(r.x, [1, 1], atol=1e-5)
    assert_allclose(r.diagnostics["multiplier"], [-1], atol=1e-5)
    assert r.diagnostics["final_penalty"] > 0.01
    for old, new in zip(r.history, r.history[1:], strict=False):
        expected = (
            old.diagnostics["multiplier"] + new.diagnostics["penalty_used"] * p.evaluate(new.x).H[0]
        )
        assert_allclose(new.diagnostics["multiplier"], expected, atol=1e-12)
        assert_allclose(new.diagnostics["alpha"], [0.5, 0.5])


def test_admm_updates_objective_weights_by_saddle_envelope():
    first = Quadratic(centres=((0.0,), (1.0,)), curvature=(1.0, 2.0))
    second = Quadratic(centres=((2.0,), (3.0,)), curvature=(2.0, 1.0))
    p = SplitProblem(first, second, [[1]], [[-1]], [0])
    r = MultiobjectiveADMM(max_iter=1, weight_prox=0.7).solve(p, [0.0, 2.0])
    assert r.n_iter == 1, r.message
    a = simplex_projection(
        np.array([0.5, 0.5]) + (first.evaluate(r.x[:1]).F[0] - first.evaluate([0]).F[0]) / 0.7
    )
    b = simplex_projection(a + (second.evaluate(r.x[1:]).F[0] - second.evaluate([2]).F[0]) / 0.7)
    assert_allclose(r.diagnostics["alpha"], a, atol=1e-10)
    assert_allclose(r.diagnostics["beta"], b, atol=1e-10)
    assert not np.allclose(a, [0.5, 0.5])
    assert not r.success


def test_admm_residual_balancing_and_initial_feasibility_is_not_success():
    assert balanced_penalty(1, 100, 1) == 2
    assert balanced_penalty(1, 1, 100) == 0.5
    assert balanced_penalty(1, 1, 1) == 1
    r = MultiobjectiveADMM(max_iter=0).solve(split_quadratics(), [0.0, 0.0])
    assert r.feasible and not r.success
    assert r.stationarity > 0


def test_admm_rejects_unsplit_or_constrained_blocks():
    with pytest.raises(TypeError, match="SplitProblem"):
        MultiobjectiveADMM().solve(Quadratic(), [0.0, 0.0])
    with pytest.raises(ValueError, match="G/H"):
        SplitProblem(ConstrainedQuadratic(), Quadratic(), np.eye(2), -np.eye(2), [0, 0])


@pytest.mark.parametrize("name", ["cg", "fw", "nesterov", "nesterov_proximal", "alm", "admm"])
def test_new_solvers_budget_failure_returns_last_accepted_point(name):
    p = split_quadratics() if name == "admm" else Quadratic()
    x0 = np.array([4.0, -3.0])
    r = minimize(p, x0, algorithm=name, max_nfev=1)
    assert not r.success and r.status == "max_nfev"
    assert r.nfev == 1 and r.njev == 0
    assert_array_equal(r.x, x0)
    assert_allclose(r.fun, p.evaluate(x0).F[0])


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("nesterov", {"step_rule": "backtracking"}),
        ("nesterov", {"mu": 2, "L": 1}),
        ("cg", {"beta_rule": "invented"}),
        ("cg", {"restart_interval": -1}),
        ("alm", {"penalty_growth": 1}),
        ("admm", {"penalty": 0}),
        ("admm", {"weight_prox": -1}),
        ("fw", {"bounds": "none"}),
    ],
)
def test_new_options_reject_invalid_mathematics(name, kwargs):
    with pytest.raises((ValueError, TypeError)):
        get_algorithm(name, **kwargs)
