"""Domain, version and derivative contracts beyond the frozen numerical oracles."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mop_problems import NonDifferentiableError, get_problem, get_problem_info
from mop_problems.problems.composite.fda import benchmark_time


@pytest.mark.parametrize(
    "name", [*[f"FDA{i}" for i in range(1, 6)], *[f"JY{i}" for i in range(1, 9)]]
)
def test_dynamic_time_is_explicit_and_does_not_depend_on_evaluation_order(name):
    p = get_problem(name, time=0.3)
    q = p.at_time(0.7)
    x = p.sample(3, seed=82)
    expected = p.evaluate(x).F.copy()
    q.evaluate(x)
    assert p.time == 0.3 and q.time == 0.7
    assert_array_equal(p.evaluate(x).F, expected)
    assert_allclose(q.evaluate(x).F, get_problem(name, time=0.7).evaluate(x).F)
    assert p.jacobian(x).dF.shape == (3, p.n_obj, p.n_var)
    with pytest.raises(AttributeError):
        p.time = 2
    for invalid in (-1, np.inf, np.nan, True):
        with pytest.raises(ValueError):
            p.at_time(invalid)


def test_benchmark_time_and_fda_dimensions():
    assert benchmark_time(29, frequency=10, severity=5) == 0.4
    with pytest.raises(ValueError):
        benchmark_time(0, frequency=0)
    with pytest.raises(ValueError):
        get_problem("FDA2", n_var=10)
    with pytest.raises(ValueError):
        get_problem("FDA1", n_obj=3)


@pytest.mark.parametrize("count", (1, 2))
def test_equality_residuals_have_signed_analytical_jacobians(count):
    p = get_problem("Eq-DTLZ2", n_obj=3, n_equalities=count)
    x = np.full(p.n_var, 0.5)
    if count == 1:
        x[0] += 0.4
    else:
        x[0] += np.sqrt(3) / 4
        x[1] += 0.25
    value, jac = p.evaluate_with_jacobian(x)
    assert value.feasible[0]
    assert value.G.shape == (1, 0)
    assert_allclose(value.H, 0, atol=1e-15)
    expected = np.zeros((count, p.n_var))
    expected[:, :2] = 2 * (x[:2] - 0.5)
    if count == 2:
        expected[1, 1] -= 1
    assert_allclose(jac.dH[0], expected)
    x[0] += 1e-4
    tight = p.evaluate(x, equality_tol=0)
    loose = p.evaluate(x, equality_tol=1e-3)
    assert not tight.feasible[0] and loose.feasible[0]
    assert_array_equal(tight.H, loose.H)


def test_nctp_alias_bounds_and_unclipped_residuals():
    p = get_problem("NCTP-III-6")
    assert p.name == "NCTP18"
    assert_array_equal(p.xu, np.full(p.n_var, 5))
    p = get_problem("NCTP1", n_var=3)
    x = np.array([[0.4, 1, 1], [0.6, 1, 1]])
    both = p.evaluate(x)
    assert np.any(both.G < 0)
    for j in range(len(x)):
        assert_array_equal(p.evaluate(x[j]).G[0], both.G[j])


def test_maf9_excluded_regions_never_trigger_random_repair():
    p = get_problem("MaF9")
    axis = np.linspace(-3, 3, 41)
    mesh = np.array(np.meshgrid(axis, axis)).reshape(2, -1).T
    bad = mesh[~p.domain_mask(mesh)][0]
    for check in (True, False):
        with pytest.raises(ValueError, match="excluded polygon"):
            p.evaluate(bad, check_bounds=check)
    assert p.domain_mask(p.sample(500, seed=3)).all()
    assert_array_equal(p.sample(20, seed=3), p.sample(20, seed=3))


def test_lscmop_uses_declared_aggregate_and_never_rounds_dimensions():
    p = get_problem("LSCM12")
    assert p.name == "LSCMOP12"
    assert "aggregate" in p.derivative_info["constraint_convention"]
    assert np.all(p.evaluate(p.sample(3, seed=42)).G >= 0)
    with pytest.raises(ValueError, match="multiple"):
        get_problem("LSCMOP1", n_var=101)


def test_rcm50_batch_output_equals_individual_evaluation():
    p = get_problem("RWMOP50")
    x = p.sample(4, seed=38)
    values = p.evaluate(x)
    assert_allclose(values.F, np.vstack([p.evaluate(row).F for row in x]))
    assert_allclose(values.H, np.vstack([p.evaluate(row).H for row in x]))
    for omitted in (1, 6, 7, 13, 15, 19, 25, 26, 27, 28, 29):
        with pytest.raises(KeyError):
            get_problem(f"RCM{omitted}")
    for relaxed in (5, 14):
        assert "relax" in get_problem_info(f"RCM{relaxed}")["notes"].lower()


@pytest.mark.parametrize("name", ("RCM20", "RCM47", "RCM48", "RCM49"))
def test_undefined_engineering_points_are_explicit_errors(name):
    p = get_problem(name)
    point = p.xl if name == "RCM20" else (p.xl + p.xu) / 2
    with np.errstate(all="ignore"), pytest.raises(FloatingPointError):
        p.evaluate(point)


def test_dascmaop_disabled_difficulty_has_zero_constraints():
    p = get_problem("DASCMaOP2", difficulty=(0, 0, 0), n_obj=5)
    x = p.sample(3, seed=532)
    assert_array_equal(p.evaluate(x).G, np.zeros((3, 11)))
    assert_array_equal(p.jacobian(x).dG, np.zeros((3, 11, 30)))
    with pytest.raises(ValueError):
        get_problem("DASCMaOP2", difficulty=13)


def test_ctp_linear_and_multimodal_are_explicit_options():
    linear = get_problem("CTP1", n_var=3, n_constraints=4)
    multimodal = get_problem("CTP1", n_var=3, n_constraints=4, distance="multimodal")
    assert linear.n_ieq_constr == multimodal.n_ieq_constr == 4
    x = [0.3, 0.4, 0.2]
    assert not np.allclose(linear.evaluate(x).F, multimodal.evaluate(x).F)
    with pytest.raises(ValueError):
        get_problem("CTP3", n_constraints=4)


def test_maf8_distance_at_vertex_has_no_classical_derivative():
    p = get_problem("MaF8")
    assert_allclose(p.evaluate(p.vertices[0]).F[0, 0], 0)
    with pytest.raises((NonDifferentiableError, FloatingPointError)):
        p.jacobian(p.vertices[0])
