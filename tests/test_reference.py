"""Independent cross-library checks, pinned to pymoo 0.6.2 by the reference extra."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mop_problems import get_problem

pytest.importorskip("pymoo")
from pymoo.problems import get_problem as reference_problem  # noqa: E402

pytestmark = pytest.mark.reference


@pytest.mark.parametrize("name", ["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6"])
@pytest.mark.parametrize("n_var", [2, 10, 30])
def test_zdt_against_pymoo(name, n_var):
    p = get_problem(name, n_var=n_var)
    ref = reference_problem(name.lower(), n_var=n_var)
    X = np.vstack((p.sample(97, seed=728), p.xl, p.xu, (p.xl + p.xu) / 2))
    assert_allclose(p.evaluate(X).F, ref.evaluate(X), rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("i", range(1, 8))
@pytest.mark.parametrize("n_obj", [2, 3, 5])
def test_dtlz_against_pymoo(i, n_obj):
    # Check k=1 as well as the non-default n_var=2*M+3.
    for n_var in [n_obj, 2 * n_obj + 3]:
        kwargs = {"n_var": n_var, "n_obj": n_obj}
        if i == 4:
            kwargs["alpha"] = 7.5
        p = get_problem(f"DTLZ{i}", **kwargs)
        ref = reference_problem(f"dtlz{i}", **kwargs)
        X = np.vstack((p.sample(97, seed=617), p.xl, p.xu, (p.xl + p.xu) / 2))
        assert_allclose(p.evaluate(X).F, ref.evaluate(X), rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize(
    "name,scale",
    [
        ("BNH", [25, 7.7]),
        ("TNK", [1, 0.5]),
        ("OSY", [2, 6, 2, 2, 4, 4]),
        ("SRN", [1, 1]),
    ],
)
def test_constrained_raw_residuals_against_scaled_reference(name, scale):
    p = get_problem(name)
    ref = reference_problem(name.lower())
    X = p.sample(300, seed=923)
    ours = p.evaluate(X)
    expected_F, expected_G = ref.evaluate(X, return_values_of=["F", "G"])
    assert_allclose(ours.F, expected_F, rtol=1e-12, atol=1e-12)
    assert_allclose(ours.G, expected_G * scale, rtol=1e-12, atol=1e-12)
    assert_array_equal(ours.feasible, np.all(expected_G <= 0, axis=1))
