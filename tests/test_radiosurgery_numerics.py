# ruff: noqa: E402 -- optional solver/license checks precede application imports
"""Algebraic equivalence, changing-rho state handling, and quality audits."""

import unittest

import pytest

pytest.importorskip("scipy")
pytest.importorskip("coptpy")
from mop_problems.applications.radiosurgery._support import create_environment

try:
    _test_environment = create_environment()
except RuntimeError as error:
    pytest.skip(str(error), allow_module_level=True)
else:
    _test_environment.close()
from time import perf_counter

import numpy as np

from mop_problems.applications.radiosurgery._engine import Oracle, solve_admm
from mop_problems.applications.radiosurgery.model import initial, make_case, weight_grid
from mop_problems.applications.radiosurgery.numerics import (
    NumericalOracle,
    ResidualBalance,
    solve_numeric_admm,
)
from mop_problems.applications.radiosurgery.oracle import CachedOracle


class NumericalRadiosurgeryTests(unittest.TestCase):
    def test_compiled_values_and_prox_match_dense_model_at_boundaries(self):
        case = make_case(11, 2)
        p = case.problem
        fast, reference = NumericalOracle(p), CachedOracle(p)
        rng = np.random.default_rng(51)
        try:
            for scale in (0.0, 0.2, 2.0):
                x = rng.normal(size=p.nx) * scale
                z = rng.normal(size=p.nz) * scale
                np.testing.assert_allclose(
                    fast.block_values("x", x), p.block_values("x", x), atol=1e-14
                )
                np.testing.assert_allclose(
                    fast.block_values("z", z), p.block_values("z", z), atol=1e-14
                )
                for w in weight_grid()[[0, 4, 8]]:
                    linear = rng.normal(size=p.nz) * 0.1
                    np.testing.assert_allclose(
                        fast.proximal_gradient("z", z, linear, w, np.inf),
                        reference.proximal_gradient("z", z, linear, w, np.inf),
                        atol=1e-14,
                    )
                    np.testing.assert_allclose(fast.time_gradient(w), fast.F.T @ w, atol=1e-14)
        finally:
            fast.close()
            reference.close()

    def test_local_models_match_reference_when_rho_changes_and_reverts(self):
        case = make_case(17, 2)
        p = case.problem
        x, z = initial(case, 1)
        fast, reference = NumericalOracle(p), CachedOracle(p)
        y = np.random.default_rng(2).normal(scale=0.03, size=p.nz)
        alpha, beta = weight_grid()[2], weight_grid()[6]
        eta, kappa = 0.7, 1.3
        try:
            for rho in (0.3, 2.0, 0.3):
                for block, anchor, other in [("x", x, z), ("z", z, x)]:
                    args = (
                        block,
                        anchor,
                        other,
                        y,
                        alpha,
                        beta,
                        rho,
                        eta,
                        kappa,
                        2e-6,
                        perf_counter() + 20,
                    )
                    u, w, gap = fast.local(*args)
                    v, w0, _ = reference.local(*args)
                    np.testing.assert_allclose(w, w0, atol=3e-6)
                    delta = p.block_values(block, u) - p.block_values(block, v)
                    ru = p.residual(u, other) if block == "x" else p.residual(other, u)
                    rv = p.residual(v, other) if block == "x" else p.residual(other, v)
                    difference = w @ delta + y @ (ru - rv) + 0.5 * rho * (ru @ ru - rv @ rv)
                    self.assertLess(abs(difference), 3e-7)
                    self.assertLessEqual(gap, 2e-6)
                    C = p.A if block == "x" else p.B
                    mapped = reference.proximal_gradient(block, u, C.T @ (y + rho * ru), w, np.inf)
                    self.assertLess(np.max(np.abs(mapped - u)), 4e-6)
            self.assertEqual(fast.work["gram_builds"], 1)
            self.assertEqual(fast.work["time_quadratic_builds"], 1)
            self.assertEqual(fast.work["time_quadratic_resets"], 3)
        finally:
            fast.close()
            reference.close()

    def test_cached_projection_updates_linear_objective(self):
        case = make_case(31)
        p = case.problem
        fast, reference = NumericalOracle(p), Oracle(p)
        try:
            for point in (np.ones(p.nx), np.linspace(-0.5, 0.6, p.nx)):
                a = fast.project(p.X, point, perf_counter() + 10)
                b = reference.project(p.X, point, perf_counter() + 10)
                np.testing.assert_allclose(a, b, atol=2e-6)
            self.assertEqual(fast.work["projection_objective_builds"], 1)
        finally:
            fast.close()
            reference.close()

    def test_fixed_penalty_matches_existing_iteration(self):
        for seed in (101, 202):
            case = make_case(seed)
            p = case.problem
            x, z = initial(case, 0)
            w = weight_grid()[4]
            args = dict(penalty=0.3, max_iterations=15, seconds_limit=15.0)
            a = solve_numeric_admm(p, x, z, w, balance=None, **args)
            b = solve_admm(p, x, z, w, oracle_factory=CachedOracle, **args)
            self.assertEqual(a["iterations"], b["iterations"])
            self.assertEqual(a["status"], b["status"])
            for key in ("y", "w", "beta", "z"):
                np.testing.assert_allclose(a[key], b[key], atol=3e-6)
            np.testing.assert_allclose(
                p.values(a["x"], a["z"]), p.values(b["x"], b["z"]), atol=3e-6
            )
            self.assertEqual(a["operator_work"]["time_quadratic_resets"], 1)

    def test_residual_balance_bounds_freeze_and_zero_residuals(self):
        policy = ResidualBalance(minimum=0.25, maximum=4.0, until=30)
        self.assertEqual(policy.update(1.0, 100.0, 1.0, 10), 2.0)
        self.assertEqual(policy.update(1.0, 1.0, 100.0, 10), 0.5)
        self.assertEqual(policy.update(4.0, 100.0, 1.0, 20), 4.0)
        self.assertEqual(policy.update(0.25, 1.0, 100.0, 20), 0.25)
        self.assertEqual(policy.update(1.0, 0.0, 0.0, 10), 1.0)
        self.assertEqual(policy.update(1.0, 100.0, 1.0, 11), 1.0)
        self.assertEqual(policy.update(1.0, 100.0, 1.0, 40), 1.0)
        with self.assertRaises(ValueError):
            ResidualBalance(factor=1.0)

    def test_changing_penalty_keeps_unscaled_multiplier_and_quality(self):
        case = make_case(101)
        p = case.problem
        x, z = initial(case, 0)
        w = weight_grid()[4]
        trace = []

        class TraceOracle(NumericalOracle):
            def local(
                self,
                block,
                anchor,
                other,
                y,
                old,
                other_weight,
                rho,
                eta,
                kappa,
                inner_tol,
                deadline,
            ):
                result = super().local(
                    block, anchor, other, y, old, other_weight, rho, eta, kappa, inner_tol, deadline
                )
                if block == "z":
                    trace.append((y.copy(), rho, p.residual(other, result[0])))
                return result

        result = solve_numeric_admm(p, x, z, w, oracle_factory=TraceOracle, seconds_limit=15.0)
        self.assertEqual(result["status"], "converged")
        self.assertTrue(result["penalty_updates"])
        for previous, current in zip(trace, trace[1:], strict=False):
            np.testing.assert_allclose(
                current[0], previous[0] + previous[1] * previous[2], atol=1e-13
            )
        reference = CachedOracle(p)
        try:
            certificate = reference.certificate(result["x"], result["z"], result["y"], result["w"])
            self.assertLessEqual(certificate["kkt"], 1e-4)
        finally:
            reference.close()
        physical, _ = case.physical(result["x"])
        objectives = case.objectives(physical)
        scalar_lp = case.lp(result["w"])
        self.assertLess(abs(result["w"] @ objectives - scalar_lp["value"]), 5e-5)
        witness = case.lp(np.ones(p.m), caps=objectives, cap_slack=1e-9)
        self.assertLess(objectives.sum() - witness["objectives"].sum(), 1e-3)


if __name__ == "__main__":
    unittest.main()
