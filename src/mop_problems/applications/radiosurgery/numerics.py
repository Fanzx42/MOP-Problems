"""Numerical implementation of the existing local models and residual balancing.

The specialized oracle accepts linear time objectives, disjoint coordinate
hinges, and B=-I. It changes algebra/model assembly, not the local minimax model.
Residual balancing is a heuristic borrowed from scalar ADMM, not a new
convergence claim for the multiobjective iteration.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ._engine import infnorm
from ._support import diagonal_simplex, simplex
from .oracle import CachedOracle


@dataclass(frozen=True)
class ResidualBalance:
    interval: int = 10
    ratio: float = 10.0
    factor: float = 2.0
    minimum: float = 1e-4
    maximum: float = 1e4
    until: int = 200

    def __post_init__(self):
        if (
            self.interval < 1
            or self.ratio <= 1
            or self.factor <= 1
            or not 0 < self.minimum <= self.maximum
            or self.until < 0
        ):
            raise ValueError("Invalid residual-balance configuration")

    def update(self, rho, primal, dual, iteration):
        if iteration > self.until or iteration % self.interval:
            return rho
        if primal > self.ratio * dual:
            return min(self.maximum, rho * self.factor)
        if dual > self.ratio * primal:
            return max(self.minimum, rho / self.factor)
        return rho


class NumericalOracle(CachedOracle):
    def __init__(self, problem):
        p = problem
        if p.B.shape != (p.nz, p.nz) or not np.array_equal(p.B, -np.eye(p.nz)):
            raise ValueError("NumericalOracle requires B=-I")
        if any(t.kind != "linear" for terms in p.f for t in terms):
            raise ValueError("NumericalOracle requires linear time objectives")
        # Compile selectors once. No dense selector products in response loops.
        ids, slopes, offsets, coefficients, groups = [], [], [], [], []
        for i, terms in enumerate(p.g):
            for term in terms:
                if term.kind != "hinge":
                    raise ValueError("NumericalOracle requires coordinate hinges")
                for row, offset in zip(term.matrix, term.offset, strict=False):
                    index = np.flatnonzero(row)
                    if len(index) != 1 or int(index[0]) in ids:
                        raise ValueError("Dose hinges must use disjoint coordinates")
                    ids.append(int(index[0]))
                    slopes.append(row[index[0]])
                    offsets.append(offset)
                    groups.append(i)
                    coefficients.append(term.coefficient / p.scales[i])
        self.ids = np.asarray(ids, dtype=int)
        self.slopes = np.asarray(slopes)
        self.offsets = np.asarray(offsets)
        self.coefficients = np.asarray(coefficients)
        self.groups = np.asarray(groups, dtype=int)
        self.signs = np.sign(self.slopes)
        self.abs_slopes = np.abs(self.slopes)
        self.hinge_scale = self.coefficients * self.abs_slopes
        self.F = np.zeros((p.m, p.nx))
        self.f_offset = np.zeros(p.m)
        for i, terms in enumerate(p.f):
            for term in terms:
                coefficient = term.coefficient / p.scales[i]
                self.F[i] += coefficient * term.matrix.sum(axis=0)
                self.f_offset[i] -= coefficient * term.offset.sum()
        self.f_groups, self.f_ids = np.nonzero(self.F)
        self.f_coefficients = self.F[self.f_groups, self.f_ids]
        self.gram = p.A.T @ p.A
        self._ax_point = None
        self._ax_value = None
        self.fast_cache = {}
        self.work = dict(
            gram_builds=1,
            ax_products=0,
            at_products=0,
            time_quadratic_builds=0,
            time_quadratic_resets=0,
            dose_evaluations=0,
            dose_newton_steps=0,
            projection_objective_builds=0,
        )
        super().__init__(p)

    def close(self):
        for data in self.fast_cache.values():
            data["model"].dispose()
        self.fast_cache.clear()
        super().close()

    def ax(self, x):
        # Solver iterates are replaced, not mutated. Keep the array reference
        # itself so a recycled Python id can never produce a false cache hit.
        if x is not self._ax_point:
            self._ax_point = x
            self._ax_value = self.p.A @ x
            self.work["ax_products"] += 1
        return self._ax_value

    def at(self, value):
        self.work["at_products"] += 1
        return self.p.A.T @ value

    def residual(self, x, z):
        return self.ax(x) - z - self.p.c

    def block_values(self, block, point):
        if block == "x":
            return self.time_values(point) + self.f_offset
        values = self.coefficients * np.maximum(self.slopes * point[self.ids] - self.offsets, 0.0)
        return np.bincount(self.groups, weights=values, minlength=self.p.m)

    def time_values(self, point):
        return np.bincount(
            self.f_groups, weights=self.f_coefficients * point[self.f_ids], minlength=self.p.m
        )

    def time_gradient(self, weights):
        return np.bincount(
            self.f_ids, weights=self.f_coefficients * weights[self.f_groups], minlength=self.p.nx
        )

    def project(self, domain, point, deadline):
        if not len(domain.rhs) and not len(domain.limits):
            return np.clip(point, domain.lower, domain.upper)
        key = ("projection", id(domain))
        if key not in self.fast_cache:
            import coptpy as cp

            m = self.model()
            u = m.addMVar(
                len(point),
                lb=np.where(np.isfinite(domain.lower), domain.lower, -cp.COPT.INFINITY),
                ub=np.where(np.isfinite(domain.upper), domain.upper, cp.COPT.INFINITY),
            )
            if len(domain.rhs):
                m.addConstr(domain.equalities @ u == domain.rhs)
            if len(domain.limits):
                m.addConstr(domain.inequalities @ u <= domain.limits)
            m.setObjective(0.5 * (u @ u))
            self.fast_cache[key] = dict(model=m, u=u)
            self.work["projection_objective_builds"] += 1
        data = self.fast_cache[key]
        m, u = data["model"], data["u"]
        u.setInfo("Obj", -point)
        self.optimize(m, deadline)
        return np.asarray(u.x.tonumpy())

    def proximal_gradient(self, block, point, linear, weights, deadline):
        if block == "x":
            return self.project(self.p.X, point - linear - self.time_gradient(weights), deadline)
        return self.dose_response(point - linear, weights, 1.0)

    def dose_response(self, value, weights, rho):
        result = value.copy()
        hinge = self.signs * value[self.ids] - self.offsets / self.abs_slopes
        threshold = weights[self.groups] * self.hinge_scale / rho
        hinge = np.where(hinge >= 0, np.maximum(hinge - threshold, 0.0), hinge)
        result[self.ids] = self.signs * (hinge + self.offsets / self.abs_slopes)
        return np.clip(result, self.p.Z.lower, self.p.Z.upper)

    def certificate(self, x, z, y, w, deadline=np.inf):
        up = self.proximal_gradient("x", x, self.at(y), w, deadline)
        vp = self.proximal_gradient("z", z, -y, w, deadline)
        stationarity = max(infnorm(up - x), infnorm(vp - z))
        feasibility = infnorm(self.residual(x, z))
        domain = self.p.violation(x, z)
        return dict(
            feasibility=feasibility,
            stationarity=stationarity,
            domain_violation=domain,
            kkt=max(feasibility, stationarity, domain),
        )

    def local(
        self, block, anchor, other, y, old, other_weight, rho, eta, kappa, inner_tol, deadline
    ):
        if block == "z":
            return self.dose_local(
                anchor, other, y, old, other_weight, rho, eta, kappa, inner_tol, deadline
            )
        import coptpy as cp

        p = self.p
        mu = 1 / eta + kappa
        center = (old / eta + kappa * other_weight) / mu
        base = self.time_values(anchor)
        linear = self.at(y - rho * (other + p.c))
        key = ("time", float(mu))
        if key not in self.fast_cache:
            m = self.model()
            u, _ = self.block(m, "x")
            threshold = m.addVar(lb=-cp.COPT.INFINITY)
            surplus = m.addMVar(p.m, lb=-mu * center)
            constraints = m.addConstr(self.F @ u - threshold - surplus <= base)
            # Numeric Gram construction once, symbolic quadratic once.
            quadratic = 0.5 * (u @ (self.gram @ u))
            remainder = threshold + (surplus @ surplus) / (2 * mu)
            self.fast_cache[key] = dict(
                model=m,
                u=u,
                surplus=surplus,
                constraints=constraints,
                quadratic=quadratic,
                remainder=remainder,
                rho=None,
            )
            self.work["time_quadratic_builds"] += 1
        data = self.fast_cache[key]
        m, u, surplus = data["model"], data["u"], data["surplus"]
        if data["rho"] != rho:
            m.setObjective(rho * data["quadratic"] + data["remainder"])
            data["rho"] = rho
            self.work["time_quadratic_resets"] += 1
        u.setInfo("Obj", linear)
        surplus.setInfo("Obj", center)
        surplus.setInfo("LB", -mu * center)
        data["constraints"].setInfo("UB", base)
        _, lower = self.optimize(m, deadline)
        un = np.asarray(u.x.tonumpy())
        delta = self.time_values(un) - base
        w = simplex(center + delta / mu)
        ax = self.ax(un)
        gap = float(
            w @ delta
            - 0.5 * mu * np.sum((w - center) ** 2)
            + 0.5 * rho * (ax @ ax)
            + linear @ un
            - lower
        )
        if not -2e-6 <= gap <= inner_tol:
            raise RuntimeError(f"Numerical time response gap {gap:g}")
        return un, w, max(0.0, gap)

    def dose_local(self, anchor, other, y, old, other_weight, rho, eta, kappa, inner_tol, deadline):
        p = self.p
        mu = 1 / eta + kappa
        center = (old / eta + kappa * other_weight) / mu
        base = self.block_values("z", anchor)
        value = self.ax(other) - p.c + y / rho
        lo, hi = p.Z.lower, p.Z.upper

        def evaluate(w):
            self.work["dose_evaluations"] += 1
            u = self.dose_response(value, w, rho)
            active = (
                (self.slopes * u[self.ids] - self.offsets > 1e-12)
                & (lo[self.ids] + 1e-12 < u[self.ids])
                & (u[self.ids] < hi[self.ids] - 1e-12)
            )
            curvature = mu + np.bincount(
                self.groups, weights=active * self.hinge_scale**2 / rho, minlength=p.m
            )
            delta = self.block_values("z", u) - base
            negative_dual = float(
                0.5 * mu * np.sum((w - center) ** 2)
                - w @ delta
                - 0.5 * rho * np.sum((u - value) ** 2)
            )
            gradient = mu * (w - center) - delta
            best = simplex(center + delta / mu)
            gap = float(
                (best - w) @ delta
                - 0.5 * mu * (np.sum((best - center) ** 2) - np.sum((w - center) ** 2))
            )
            return negative_dual, gradient, curvature, u, gap

        w = simplex(center)
        for _ in range(100):
            if perf_counter() >= deadline:
                raise TimeoutError("Dose weight solve time limit")
            objective, gradient, curvature, u, gap = evaluate(w)
            if gap <= 1e-12:
                return u, w, max(0.0, gap)
            self.work["dose_newton_steps"] += 1
            direction = diagonal_simplex(w - gradient / curvature, curvature) - w
            step = 1.0
            for _ in range(50):
                trial = w + step * direction
                if evaluate(trial)[0] <= objective + 1e-4 * step * (gradient @ direction) + 1e-15:
                    w = trial
                    break
                step *= 0.5
            else:
                raise RuntimeError("Dose dual Newton line search failed")
        raise RuntimeError(f"Dose dual Newton iteration limit; gap={gap:g}")


_DEFAULT_BALANCE = ResidualBalance()


def solve_numeric_admm(
    p,
    x0,
    z0,
    w0,
    penalty=1.0,
    tol=1e-4,
    inner_tol=2e-6,
    seconds_limit=25.0,
    max_iterations=3000,
    kappa=1.0,
    eta=1.0,
    balance=_DEFAULT_BALANCE,
    oracle_factory=NumericalOracle,
):
    """Existing multiobjective block updates with optional finite rho adaptation.

    y is unscaled and never rescaled when rho changes. Classical physical
    residuals control rho; common-weight KKT and weight agreement still control
    stopping. Adaptation freezes after `balance.until` iterations.
    """
    if penalty <= 0 or eta <= 0 or kappa < 0:
        raise ValueError("Invalid ADMM parameter")
    if balance is not None and not balance.minimum <= penalty <= balance.maximum:
        raise ValueError("Initial penalty is outside adaptation bounds")
    start = perf_counter()
    oracle = oracle_factory(p)
    x, z, alpha, beta = x0.copy(), z0.copy(), w0.copy(), w0.copy()
    y = np.zeros(len(p.c))
    rho = float(penalty)
    history, updates, gaps = [], [], []
    status, reason, iterations = "iterations_limit", None, 0
    residual_info = {}
    try:
        cert = oracle.certificate(x, z, y, alpha)
        for k in range(1, max_iterations + 1):
            if max(cert["kkt"], infnorm(alpha - beta)) <= tol:
                status = "converged"
                break
            if perf_counter() - start >= seconds_limit:
                status = "time_limit"
                break
            try:
                xn, an, gx = oracle.local(
                    "x", x, z, y, alpha, beta, rho, eta, kappa, inner_tol, start + seconds_limit
                )
                zn, bn, gz = oracle.local(
                    "z", z, xn, y, beta, an, rho, eta, kappa, inner_tol, start + seconds_limit
                )
            except (RuntimeError, TimeoutError) as error:
                status = "time_limit" if isinstance(error, TimeoutError) else "inner_failure"
                reason = str(error)
                break
            previous_z = z
            x, z, alpha, beta = xn, zn, an, bn
            residual = oracle.residual(x, z)
            y += rho * residual
            iterations = k
            gaps.extend((gx, gz))
            if k % 5 == 0:
                cert = oracle.certificate(x, z, y, alpha)
                history.append(
                    dict(
                        iteration=k,
                        seconds=perf_counter() - start,
                        **cert,
                        consensus=infnorm(alpha - beta),
                        penalty=rho,
                    )
                )
            else:
                cert = {"kkt": np.inf}
            if balance is not None and k <= balance.until and k % balance.interval == 0:
                # B=-I, so s=-rho*A.T@(z-z_previous), evaluated only at checkpoints.
                dual_residual = -rho * oracle.at(z - previous_z)
                primal_scale = max(
                    1.0, np.linalg.norm(oracle.ax(x)), np.linalg.norm(z), np.linalg.norm(p.c)
                )
                dual_scale = max(1.0, np.linalg.norm(oracle.at(y)))
                primal = float(np.linalg.norm(residual) / primal_scale)
                dual = float(np.linalg.norm(dual_residual) / dual_scale)
                residual_info = dict(
                    residual_iteration=k,
                    primal_norm=float(np.linalg.norm(residual)),
                    dual_norm=float(np.linalg.norm(dual_residual)),
                    primal_relative=primal,
                    dual_relative=dual,
                )
                next_rho = balance.update(rho, primal, dual, k)
                if next_rho != rho and max(cert["kkt"], infnorm(alpha - beta)) > tol:
                    updates.append(
                        dict(
                            iteration=k,
                            before=rho,
                            after=next_rho,
                            unscaled_y_norm=float(np.linalg.norm(y)),
                            **residual_info,
                        )
                    )
                    rho = next_rho
        cert = oracle.certificate(x, z, y, alpha)
        if max(cert["kkt"], infnorm(alpha - beta)) <= tol:
            status = "converged"
        elapsed = perf_counter() - start
        history.append(
            dict(
                iteration=iterations,
                seconds=elapsed,
                **cert,
                consensus=infnorm(alpha - beta),
                penalty=rho,
                **residual_info,
            )
        )
        return dict(
            method="Numerical-Consensus-ADMM",
            x=x,
            z=z,
            y=y,
            w=alpha,
            beta=beta,
            status=status,
            reason=reason,
            iterations=iterations,
            inner_steps=2 * iterations,
            seconds=elapsed,
            history=history,
            **cert,
            consensus=infnorm(alpha - beta),
            residual=max(cert["kkt"], infnorm(alpha - beta)),
            penalty=penalty,
            final_penalty=rho,
            penalty_updates=updates,
            max_inner_gap=max(gaps, default=0.0),
            solves=oracle.solves,
            barrier_iterations=oracle.barrier_iterations,
            native_status_counts=oracle.native_status_counts,
            max_native_gap=oracle.max_native_gap,
            operator_work=oracle.work.copy(),
        )
    finally:
        oracle.close()
