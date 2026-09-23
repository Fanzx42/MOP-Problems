"""Validate the numerical contract, including empty batches and equality handling."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mop_problems import Problem, get_problem, get_problem_info, list_problems


@pytest.mark.parametrize("name", list_problems())
def test_batch_contract_and_metadata(name):
    problem = get_problem(name)
    info = get_problem_info(name)
    assert (problem.n_var, problem.n_obj) == (info["n_var"], info["n_obj"])
    assert_array_equal(problem.xl, np.broadcast_to(info["lower_bounds"], (problem.n_var,)))
    assert_array_equal(problem.xu, np.broadcast_to(info["upper_bounds"], (problem.n_var,)))
    X = problem.sample(7, seed=41)
    before = X.copy()
    result = problem.evaluate(X)
    assert_array_equal(X, before)
    assert result.F.shape == (7, problem.n_obj)
    assert result.G.shape == (7, problem.n_ieq_constr)
    assert result.H.shape == (7, problem.n_eq_constr)
    assert result.CV.shape == result.feasible.shape == (7,)
    assert np.all(np.isfinite(result.F))
    for i, row in enumerate(X):
        single = problem.evaluate(row)
        assert single.F.shape == (1, problem.n_obj)
        assert_allclose(single.F[0], result.F[i])
        assert_allclose(single.G[0], result.G[i])
    empty = problem.evaluate(np.empty((0, problem.n_var)))
    assert empty.F.shape == (0, problem.n_obj)
    assert empty.CV.shape == (0,)
    assert empty.G.shape == (0, problem.n_ieq_constr)


@pytest.mark.parametrize("name", list_problems())
def test_boundaries_sampling_and_immutability(name):
    p = get_problem(name)
    out = p.evaluate(np.vstack((p.xl, p.xu, (p.xl + p.xu) / 2)))
    assert np.isfinite(out.F).all()
    assert np.isfinite(out.G).all()
    assert_array_equal(p.sample(5, seed=18), p.sample(5, seed=18))
    assert p.sample(0).shape == (0, p.n_var)
    sample = p.sample(5, seed=np.random.default_rng(2))
    assert np.all(sample >= p.xl) and np.all(sample <= p.xu)
    assert not out.F.flags.writeable and not p.xl.flags.writeable
    assert p.name in repr(p)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_input_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        get_problem("SCH").evaluate([bad])


@pytest.mark.parametrize("bad", [[], 1.0, [[1, 2]], [[[1]]]])
def test_bad_shape_rejected(bad):
    with pytest.raises(ValueError, match="shape"):
        get_problem("SCH").evaluate(bad)


def test_complex_and_outside_bounds_rejected():
    with pytest.raises(TypeError, match="real"):
        get_problem("SCH").evaluate([1j])
    with pytest.raises(ValueError, match="bounds"):
        get_problem("SCH").evaluate([1001])
    out = get_problem("SCH").evaluate([1001], check_bounds=False)
    assert_allclose(out.CV, [1])
    assert not out.feasible[0]


@pytest.mark.parametrize("kwargs", [{"n_var": 1}, {"n_var": 0}, {"n_var": -1}])
def test_zdt_dimension_restrictions(kwargs):
    with pytest.raises(ValueError):
        get_problem("ZDT1", **kwargs)


@pytest.mark.parametrize("bad", [3.0, True, "3", np.bool_(True)])
def test_dimensions_must_be_integers(bad):
    with pytest.raises(TypeError):
        get_problem("ZDT1", n_var=bad)


@pytest.mark.parametrize("kwargs", [{"n_obj": 1}, {"n_obj": 4, "n_var": 3}])
def test_dtlz_dimension_restrictions(kwargs):
    with pytest.raises(ValueError):
        get_problem("DTLZ2", **kwargs)


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf, [2, 3]])
def test_dtlz_alpha_restrictions(bad):
    with pytest.raises(ValueError):
        get_problem("DTLZ4", alpha=bad)


def test_unknown_parameters_do_not_silently_change_problem():
    with pytest.raises(TypeError):
        get_problem("BNH", n_var=3)
    with pytest.raises(TypeError):
        get_problem("ZDT1", n_obj=3)
    with pytest.raises(ValueError):
        get_problem("FON", n_var=1)
    with pytest.raises(ValueError):
        get_problem("SCH").sample(-1)


class EqualityFixture(Problem):
    """Analytical fixture for generic equality handling; not a registered benchmark."""

    def __init__(self, **kwargs):
        super().__init__("fixture", 1, 2, kwargs.get("xl", -1), kwargs.get("xu", 1), 1, 1)

    def _evaluate(self, X):
        return np.column_stack((X[:, 0] ** 2, (X[:, 0] - 1) ** 2)), X - 0.5, X


def test_equality_tolerance_raw_residual_and_violation_sum():
    p = EqualityFixture()
    out = p.evaluate([[0], [1e-5], [0.75]], equality_tol=1e-4)
    assert_array_equal(out.feasible, [True, True, False])
    assert_allclose(out.H[:, 0], [0, 1e-5, 0.75])
    assert_allclose(out.CV, [0, 0, 0.9999])
    strict = p.evaluate([1e-5], equality_tol=0)
    assert_allclose(strict.CV, [1e-5])
    assert_allclose(p.evaluate([1.5], check_bounds=False, equality_tol=0).CV, [3.0])


@pytest.mark.parametrize("bad", [-1, np.nan, np.inf, [1]])
def test_bad_tolerance_rejected(bad):
    with pytest.raises(ValueError):
        EqualityFixture().evaluate([0], equality_tol=bad)


@pytest.mark.parametrize("kwargs", [{"xl": [0, 1]}, {"xl": np.nan}, {"xl": 1}, {"xu": -2}])
def test_invalid_problem_bounds(kwargs):
    with pytest.raises(ValueError):
        EqualityFixture(**kwargs)


@pytest.mark.parametrize("mode", ["shape", "missing", "nonfinite"])
def test_bad_implementation_fails_loudly(mode):
    class Broken(EqualityFixture):
        def _evaluate(self, X):
            F, G, H = super()._evaluate(X)
            if mode == "shape":
                return F[:, 0], G, H
            if mode == "missing":
                return F, None, H
            F[:] = np.inf
            return F, G, H

    error = FloatingPointError if mode == "nonfinite" else RuntimeError
    with pytest.raises(error):
        Broken().evaluate([0])
