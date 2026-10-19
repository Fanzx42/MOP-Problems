"""Cached exact small convex subproblems; original outer algorithms unchanged."""

from time import perf_counter

import numpy as np

from ._engine import Oracle as BaseOracle
from ._support import diagonal_simplex, simplex


class CachedOracle(BaseOracle):
    def __init__(self, problem):
        super().__init__(problem)
        self.cache = {}

    def close(self):
        for data in self.cache.values():
            data[0].dispose()
        self.cache.clear()
        super().close()

    def project(self, domain, point, deadline):
        if not len(domain.rhs) and not len(domain.limits):
            return np.clip(point, domain.lower, domain.upper)
        import coptpy as cp

        key = ("projection", id(domain))
        if key not in self.cache:
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
            self.cache[key] = (m, u)
        m, u = self.cache[key]
        m.setObjective(0.5 * (u @ u) - point @ u)
        self.optimize(m, deadline)
        return np.asarray(u.x.tonumpy())

    def proximal_gradient(self, block, point, linear, weights, deadline):
        if block == "x":
            return super().proximal_gradient(block, point, linear, weights, deadline)
        # Dose thresholds rescale selector rows (e.g. LD/.5), unlike the older
        # certificate that only allowed +/- unit selectors. Each voxel belongs
        # to exactly one hinge group in this fixed-pass model.
        out = point - linear
        seen = set()
        for i, terms in enumerate(self.p.g):
            for term in terms:
                for row, offset in zip(term.matrix, term.offset, strict=False):
                    ids = np.flatnonzero(row)
                    if term.kind != "hinge" or len(ids) != 1 or ids[0] in seen:
                        raise ValueError("Dose certificate requires disjoint coordinate hinges")
                    j = int(ids[0])
                    seen.add(j)
                    a = row[j]
                    threshold = weights[i] * term.coefficient * abs(a) / self.p.scales[i]
                    value = np.sign(a) * out[j] - offset / abs(a)
                    value = max(value - threshold, 0.0) if value >= 0 else value
                    out[j] = np.sign(a) * (value + offset / abs(a))
        return np.clip(out, self.p.Z.lower, self.p.Z.upper)

    def weighted(self, block, rho, linear, w, deadline):
        key = ("weighted", block)
        if key not in self.cache:
            m = self.model()
            u, terms = self.block(m, block)
            self.cache[key] = (m, u, terms)
        m, u, terms = self.cache[key]
        C = self.p.A if block == "x" else self.p.B
        v = C @ u
        m.setObjective(
            0.5 * rho * (v @ v) + linear @ u + sum(float(w[i]) * terms[i] for i in range(self.p.m))
        )
        value, _ = self.optimize(m, deadline)
        return value

    def local(
        self, block, anchor, other_point, y, old, other_weight, rho, eta, kappa, inner_tol, deadline
    ):
        import coptpy as cp

        p = self.p
        mu = 1 / eta + kappa
        center = (old / eta + kappa * other_weight) / mu
        if block == "z":
            return self.dose_local(
                anchor, other_point, y, old, other_weight, rho, eta, kappa, inner_tol, deadline
            )
        C, offset = (
            (p.A, p.B @ other_point - p.c) if block == "x" else (p.B, p.A @ other_point - p.c)
        )
        linear = C.T @ (y + rho * offset)
        base = p.block_values(block, anchor)
        key = ("local", block)
        if key not in self.cache:
            m = self.model()
            u, terms = self.block(m, block)
            threshold = m.addVar(lb=-cp.COPT.INFINITY)
            surplus = m.addMVar(p.m, lb=-mu * center)
            constraints = [
                m.addConstr(terms[i] - threshold - surplus[i].item() <= float(base[i]))
                for i in range(p.m)
            ]
            self.cache[key] = (m, u, terms, threshold, surplus, constraints)
        m, u, terms, threshold, surplus, constraints = self.cache[key]
        surplus.setInfo("LB", -mu * center)
        for i, c in enumerate(constraints):
            c.setInfo("UB", float(base[i]))
        v = C @ u
        m.setObjective(
            0.5 * rho * (v @ v)
            + linear @ u
            + threshold
            + center @ surplus
            + (surplus @ surplus) / (2 * mu)
        )
        _, lower = self.optimize(m, deadline)
        un = np.asarray(u.x.tonumpy())
        vals = p.block_values(block, un)
        w = simplex(center + (vals - base) / mu)
        # Exact conjugate epigraph lower bound vs actual eliminated envelope.
        # This is a primal subproblem gap, not the fixed-weight saddle gap.
        gap = float(
            w @ (vals - base)
            - 0.5 * mu * np.sum((w - center) ** 2)
            + 0.5 * rho * np.sum((C @ un) ** 2)
            + linear @ un
            - lower
        )
        if not -2e-6 <= gap <= inner_tol:
            raise RuntimeError(f"Local response gap {gap:g}")
        return un, w, max(0.0, gap)

    def dose_local(self, anchor, other, y, old, other_weight, rho, eta, kappa, inner_tol, deadline):
        p = self.p
        mu = 1 / eta + kappa
        center = (old / eta + kappa * other_weight) / mu
        base = p.block_values("z", anchor)
        v = p.A @ other - p.c + y / rho
        lo, hi = p.Z.lower, p.Z.upper

        def evaluate(w):
            u = v.copy()
            curvature = np.full(p.m, mu)
            for i, terms in enumerate(p.g):
                for term in terms:
                    for row, offset in zip(term.matrix, term.offset, strict=False):
                        j = int(np.flatnonzero(row)[0])
                        a = row[j]
                        coefficient = term.coefficient / p.scales[i]
                        value = np.sign(a) * v[j] - offset / abs(a)
                        threshold = w[i] * coefficient * abs(a) / rho
                        value = max(value - threshold, 0.0) if value >= 0 else value
                        u[j] = np.sign(a) * (value + offset / abs(a))
                        if a * u[j] - offset > 1e-12 and lo[j] + 1e-12 < u[j] < hi[j] - 1e-12:
                            curvature[i] += (coefficient * a) ** 2 / rho
            u = np.clip(u, lo, hi)
            delta = p.block_values("z", u) - base
            negative_dual = float(
                0.5 * mu * np.sum((w - center) ** 2) - w @ delta - 0.5 * rho * np.sum((u - v) ** 2)
            )
            gradient = mu * (w - center) - delta
            best = simplex(center + delta / mu)
            gap = float(
                (best - w) @ delta
                - 0.5 * mu * (np.sum((best - center) ** 2) - np.sum((w - center) ** 2))
            )
            return negative_dual, gradient, curvature, u, gap

        w = simplex(center)
        for _iteration in range(100):
            if perf_counter() >= deadline:
                raise TimeoutError("Dose weight solve time limit")
            value, g, h, u, gap = evaluate(w)
            if gap <= 1e-12:
                return u, w, max(0.0, gap)
            direction = diagonal_simplex(w - g / h, h) - w
            step = 1.0
            for _ in range(50):
                trial = w + step * direction
                if evaluate(trial)[0] <= value + 1e-4 * step * (g @ direction) + 1e-15:
                    w = trial
                    break
                step *= 0.5
            else:
                raise RuntimeError("Dose dual Newton line search failed")
        raise RuntimeError(f"Dose dual Newton iteration limit; gap={gap:g}")

    def alm_step(self, x, z, y, rho, proximal, deadline):
        import coptpy as cp

        p = self.p
        key = ("joint",)
        if key not in self.cache:
            m = self.model()
            u, f = self.block(m, "x")
            v, g = self.block(m, "z")
            t = m.addVar(lb=-cp.COPT.INFINITY)
            constraints = [m.addConstr(f[i] + g[i] - t <= 0.0) for i in range(p.m)]
            self.cache[key] = (m, u, v, t, constraints)
        m, u, v, t, constraints = self.cache[key]
        base = p.values(x, z)
        for i, c in enumerate(constraints):
            c.setInfo("UB", float(base[i]))
        r = p.A @ u + p.B @ v
        m.setObjective(
            t
            + 0.5 * rho * (r @ r)
            + (p.A.T @ (y - rho * p.c) - proximal * x) @ u
            + (p.B.T @ (y - rho * p.c) - proximal * z) @ v
            + 0.5 * proximal * ((u @ u) + (v @ v))
        )
        value, bound = self.optimize(m, deadline)
        w = np.maximum(-np.array([c.pi for c in constraints]), 0.0)
        w /= w.sum()
        return np.asarray(u.x.tonumpy()), np.asarray(v.x.tonumpy()), w, max(0.0, value - bound)
