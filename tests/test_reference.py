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


@pytest.mark.parametrize("number", range(1, 10))
@pytest.mark.parametrize("n_obj", [2, 3, 5, 8])
def test_wfg_all_stages_against_pymoo(number, n_obj):
    for k, distance in [(4 * (n_obj - 1), 2), (2 * (n_obj - 1), 20)]:
        kwargs = dict(n_var=k + distance, n_obj=n_obj, k=k)
        p = get_problem(f"WFG{number}", **kwargs)
        # Original WFG permits k=2 for M=2; pymoo adds a k>=4 restriction.
        if k < 4:
            continue
        ref = reference_problem(f"wfg{number}", **kwargs)
        transitions = np.array([0, 0.349, 0.35, 0.351, 0.75, 0.85, 1])
        X = np.vstack((p.sample(71, seed=819), transitions[:, None] * p.xu))
        assert_allclose(p.evaluate(X).F, ref.evaluate(X), rtol=3e-11, atol=3e-11)


@pytest.mark.parametrize("name", ["C1-DTLZ1", "C1-DTLZ3", "C2-DTLZ2", "C3-DTLZ1", "C3-DTLZ4"])
@pytest.mark.parametrize("n_obj", [2, 3, 5, 13])
def test_constrained_dtlz_against_pymoo(name, n_obj):
    for distance in [1, 10]:
        kwargs = dict(n_var=n_obj + distance - 1, n_obj=n_obj)
        if name == "C3-DTLZ4":
            kwargs["alpha"] = 7.5
        p = get_problem(name, **kwargs)
        ref = reference_problem(name.lower().replace("-", ""), **kwargs)
        X = np.vstack((p.sample(71, seed=632), p.xl, p.xu, np.full(p.n_var, 0.5)))
        actual = p.evaluate(X)
        F, G = ref.evaluate(X, return_values_of=["F", "G"])
        assert_allclose(actual.F, F, rtol=1e-11, atol=1e-11)
        assert_allclose(actual.G, G, rtol=1e-11, atol=1e-9)


@pytest.mark.parametrize("n_var", [2, 3, 8])
def test_omni_against_pymoo(n_var):
    from pymoo.problems.multi.omnitest import OmniTest

    p = get_problem("Omni-test", n_var=n_var)
    X = np.vstack((p.sample(100, seed=231), p.xl, p.xu))
    assert_allclose(p.evaluate(X).F, OmniTest(n_var=n_var).evaluate(X), atol=1e-12)


@pytest.mark.parametrize("angle", [0, np.pi / 4, -0.37])
def test_sympart_against_pymoo(angle):
    from pymoo.problems.multi.sympart import SYMPARTRotated

    p = get_problem("SYM-PART-Rotated", angle=angle)
    X = np.vstack((p.sample(100, seed=777), p.xl, p.xu, [6, 5], [-6, -5]))
    assert_allclose(p.evaluate(X).F, SYMPARTRotated(angle=angle).evaluate(X), atol=1e-10)


def test_welded_beam_reference_variant_and_raw_scales():
    p = get_problem("Welded-Beam")
    ref = reference_problem("welded_beam")
    X = np.vstack((p.sample(300, seed=923), p.xl, p.xu))
    actual = p.evaluate(X)
    F, G = ref.evaluate(X, return_values_of=["F", "G"])
    assert_allclose(actual.F, F, rtol=1e-12, atol=1e-12)
    assert_allclose(actual.G, G * [13600, 30000, 4.875, 6000], rtol=1e-11, atol=1e-9)
    assert_array_equal(actual.feasible, np.all(G <= 0, axis=1))
