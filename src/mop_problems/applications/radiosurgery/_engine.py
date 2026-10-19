"""Constrained extensions of Consensus-ADMM and a multiobjective ALM.

The ALM uses a joint proximal-minimax inner method, not the frozen smooth
minimum-norm/Armijo implementation. Both algorithms retain exact hard domains
and original nonsmooth losses. COPT solves epigraph QCPs, or pure SDPs for PCP.
No statement of the old finite/full-space coercivity theorem is made here.
"""

from time import perf_counter

import numpy as np

from ._support import create_environment, simplex


def infnorm(a):
    return float(np.max(np.abs(a), initial=0.0))


class Oracle:
    def __init__(self, problem, solver_tol=1e-9):
        self.p = problem
        self.env = create_environment()
        self.solver_tol = solver_tol
        self.solves = 0
        self.barrier_iterations = 0
        self.native_status_counts = {}
        self.max_native_gap = 0.0

    def close(self):
        self.env.close()

    def model(self):
        m = self.env.createModel("application_convex")
        m.setParam("Logging", 0)
        m.setParam("Threads", 1)
        m.setParam("FeasTol", self.solver_tol)
        m.setParam("DualTol", self.solver_tol)
        return m

    def scalar(self, model, expression):
        """Bridge COPT's multidimensional linear API to a native scalar Var."""
        import coptpy as cp

        u = model.addMVar(1, lb=-cp.COPT.INFINITY)[0]
        model.addConstr(u == expression)
        return u.item()

    def quadratic_objective(self, model, vector):
        """Direct quadratic objective except when a pure SDP is required."""
        if self.p.sdp:
            return self.square(model, vector)
        entries = [self.scalar(model, vector[i]) for i in range(vector.size)]
        return sum(v * v for v in entries)

    def square(self, model, vector):
        """Epigraph of ||v||^2, using a Schur complement for pure SDP models.

        Native quadratic objectives mixed with PSD variables are not used.
        The SDP form [t v'; v I] >= 0 is exact, not a norm smoothing.
        """
        n = vector.size
        mt = model.addMVar(1, lb=0.0)[0]
        t = mt.item()
        if self.p.sdp:
            P = model.addPsdVar(n + 1)

            def pick(i, j):
                return P.pick([(i, j)])[0].item()

            model.addConstr(pick(0, 0) == t)
            for i in range(n):
                model.addConstr(pick(i + 1, 0) == self.scalar(model, vector[i]))
                for j in range(i + 1):
                    model.addConstr(pick(i + 1, j + 1) == float(i == j))
        else:
            model.addQConstr(vector @ vector <= mt)
        return t

    def block(self, model, block):
        import coptpy as cp

        p = self.p
        domain = p.X if block == "x" else p.Z
        n = p.nx if block == "x" else p.nz
        lower = np.where(np.isfinite(domain.lower), domain.lower, -cp.COPT.INFINITY)
        upper = np.where(np.isfinite(domain.upper), domain.upper, cp.COPT.INFINITY)
        u = model.addMVar(n, lb=lower, ub=upper, nameprefix=block)
        if len(domain.rhs):
            model.addConstr(domain.equalities @ u == domain.rhs)
        if len(domain.limits):
            model.addConstr(domain.inequalities @ u <= domain.limits)
        objectives = p.f if block == "x" else p.g
        expressions = []
        for i, terms in enumerate(objectives):
            expression = 0.0
            for term in terms:
                v = term.matrix @ u - term.offset
                if term.kind == "quadratic":
                    component = 0.5 * self.square(model, v)
                elif term.kind == "linear":
                    component = self.scalar(model, v.sum())
                elif term.kind in ("l1", "hinge", "squared_hinge"):
                    a = model.addMVar(v.size, lb=0.0)
                    model.addConstr(a >= v)
                    if term.kind == "l1":
                        model.addConstr(a >= -v)
                    component = (
                        0.5 * self.square(model, a)
                        if term.kind == "squared_hinge"
                        else cp.quicksum(a.tolist())
                    )
                elif term.kind == "nuclear":
                    rows, cols = term.shape
                    P = model.addPsdVar(rows + cols)
                    for row in range(rows):
                        for col in range(cols):
                            model.addConstr(
                                P.pick([(rows + col, row)])[0].item()
                                == self.scalar(model, v[row * cols + col])
                            )
                    component = model.addVar(lb=0.0)
                    model.addConstr(0.5 * P.diag().sum() <= component)
                else:
                    raise ValueError(term.kind)
                expression += (term.coefficient / p.scales[i]) * component
            expressions.append(expression)
        return u, expressions

    def optimize(self, model, deadline=np.inf):
        import coptpy as cp

        remaining = deadline - perf_counter()
        if remaining <= 0:
            raise TimeoutError("Application solver time limit")
        model.setParam("TimeLimit", min(remaining, 1e20))
        self.solves += 1
        model.solve()
        self.barrier_iterations += model.BarrierIter
        if model.status == cp.COPT.TIMEOUT:
            raise TimeoutError("COPT application subproblem time limit")
        self.native_status_counts[int(model.status)] = (
            self.native_status_counts.get(int(model.status), 0) + 1
        )
        if model.status not in (cp.COPT.OPTIMAL, cp.COPT.IMPRECISE) or not model.HasSol:
            raise RuntimeError(f"Convex subproblem not optimal: COPT status={model.status}")
        if not np.isfinite(model.ObjVal) or not np.isfinite(model.ObjBound):
            raise RuntimeError("Missing finite objective/bound")
        gap = float(model.ObjVal - model.ObjBound)
        self.max_native_gap = max(self.max_native_gap, gap)
        if gap > 2e-6 or gap < -2e-6:
            raise RuntimeError(f"Native convex primal/bound gap too large: {gap:g}")
        return float(model.ObjVal), float(model.ObjBound)

    def local(
        self, block, anchor, other_point, y, old, other_weight, rho, eta, kappa, inner_tol, deadline
    ):
        """Full regularized local minimax plus an independent weighted check."""
        import coptpy as cp

        p = self.p
        mu = 1 / eta + kappa
        center = (old / eta + kappa * other_weight) / mu
        base = p.block_values(block, anchor)
        objectives = p.f if block == "x" else p.g
        if not any(objectives):
            # The primal and weight parts decouple for an identically zero
            # block; removing its simplex epigraph avoids degeneracy.
            model = self.model()
            try:
                u, _ = self.block(model, block)
                r = (
                    p.A @ u + p.B @ other_point - p.c
                    if block == "x"
                    else p.A @ other_point + p.B @ u - p.c
                )
                model.setObjective(
                    self.scalar(model, y @ r) + 0.5 * rho * self.quadratic_objective(model, r)
                )
                _, lower = self.optimize(model, deadline)
                un = np.asarray(u.x.tonumpy())
            finally:
                model.dispose()
            rr = p.residual(un, other_point) if block == "x" else p.residual(other_point, un)
            gap = float(y @ rr + 0.5 * rho * (rr @ rr) - lower)
            if gap > inner_tol or gap < -2e-6:
                raise RuntimeError(f"Zero-cost block numerical gap too large: {gap:g}")
            return un, simplex(center), max(0.0, gap)
        model = self.model()
        try:
            u, expressions = self.block(model, block)
            r = (
                p.A @ u + p.B @ other_point - p.c
                if block == "x"
                else p.A @ other_point + p.B @ u - p.c
            )
            threshold = model.addVar(lb=-cp.COPT.INFINITY)
            surplus = model.addMVar(p.m, lb=-mu * center)
            for i in range(p.m):
                model.addConstr(expressions[i] - base[i] <= threshold + surplus[i].item())
            objective = (
                self.scalar(model, y @ r)
                + 0.5 * rho * self.quadratic_objective(model, r)
                + threshold
            )
            objective += self.scalar(model, center @ surplus) + self.quadratic_objective(
                model, surplus
            ) / (2 * mu)
            model.setObjective(objective)
            self.optimize(model, deadline)
            un = np.asarray(u.x.tonumpy())
        finally:
            model.dispose()
        vals = p.block_values(block, un)
        weight = simplex(center + (vals - base) / mu)
        # Unlike a curvature-based gradient bound, this works for singular A,
        # hard domains, L1/TV, and the nuclear norm. It is a numerical solver
        # bound, not an interval-arithmetic or exact-real certificate.
        checker = self.model()
        try:
            v, terms = self.block(checker, block)
            rr = (
                p.A @ v + p.B @ other_point - p.c
                if block == "x"
                else p.A @ other_point + p.B @ v - p.c
            )
            checker.setObjective(
                cp.quicksum(float(weight[i]) * terms[i] for i in range(p.m))
                + self.scalar(checker, y @ rr)
                + 0.5 * rho * self.quadratic_objective(checker, rr)
            )
            _, lower = self.optimize(checker, deadline)
        finally:
            checker.dispose()
        actual_r = p.residual(un, other_point) if block == "x" else p.residual(other_point, un)
        upper = float(weight @ vals + y @ actual_r + 0.5 * rho * (actual_r @ actual_r))
        gap = upper - lower
        domain = p.X if block == "x" else p.Z
        if domain.violation(un) > 2e-6 or gap < -2e-6 or gap > inner_tol:
            raise RuntimeError(
                f"Uncertified {block} subproblem: gap={gap:g}, domain={domain.violation(un):g}"
            )
        return un, weight, max(0.0, gap)

    def project(self, domain, point, deadline):
        if not len(domain.rhs) and not len(domain.limits):
            return np.clip(point, domain.lower, domain.upper)
        if (
            not len(domain.limits)
            and domain.equalities.shape == (1, len(point))
            and np.all(domain.equalities == 1)
            and domain.rhs[0] == 1
            and np.all(domain.lower == 0)
            and np.all(domain.upper >= 1)
        ):
            return simplex(point)
        model = self.model()
        try:
            import coptpy as cp

            u = model.addMVar(
                len(point),
                lb=np.where(np.isfinite(domain.lower), domain.lower, -cp.COPT.INFINITY),
                ub=np.where(np.isfinite(domain.upper), domain.upper, cp.COPT.INFINITY),
            )
            if len(domain.rhs):
                model.addConstr(domain.equalities @ u == domain.rhs)
            if len(domain.limits):
                model.addConstr(domain.inequalities @ u <= domain.limits)
            model.setObjective(0.5 * (u @ u) - point @ u)
            self.optimize(model, deadline)
            return np.asarray(u.x.tonumpy())
        finally:
            model.dispose()

    def proximal_gradient(self, block, point, linear, weights, deadline):
        """Independent exact norm prox and smooth-gradient/projected mapping."""
        p = self.p
        objectives = p.f if block == "x" else p.g
        domain = p.X if block == "x" else p.Z
        gradient = linear.copy()
        nonsmooth = []
        for i, terms in enumerate(objectives):
            for t in terms:
                coefficient = weights[i] / p.scales[i]
                if t.kind in ("quadratic", "linear", "squared_hinge"):
                    gradient += coefficient * t.gradient(point)
                elif coefficient > 0:
                    nonsmooth.append((coefficient, t))
        prox = point - gradient
        for coefficient, t in nonsmooth:
            threshold = coefficient * t.coefficient
            if t.kind == "nuclear":
                U, s, V = np.linalg.svd(prox.reshape(t.shape), full_matrices=False)
                prox = ((U * np.maximum(s - threshold, 0.0)) @ V).ravel()
            else:
                for row, offset in zip(t.matrix, t.offset, strict=False):
                    ids = np.flatnonzero(row)
                    if len(ids) != 1 or abs(row[ids[0]]) != 1:
                        raise ValueError(
                            "Certificate prox requires disjoint signed coordinate losses"
                        )
                    j = ids[0]
                    sign = row[j]
                    v = sign * prox[j] - offset
                    if t.kind == "l1":
                        v = np.sign(v) * max(abs(v) - threshold, 0.0)
                    elif t.kind == "hinge":
                        v = max(v - threshold, 0.0) if v >= 0 else v
                    else:
                        raise ValueError(t.kind)
                    prox[j] = sign * (v + offset)
        return self.project(domain, prox, deadline)

    def certificate(self, x, z, y, w, deadline=np.inf):
        """Common-weight proximal-gradient KKT mapping, affine and domain errors.

        Nonsmooth prox formulas are independent of the COPT epigraph builder.
        Smooth blocks use a unit gradient step and projection onto the domain.
        This diagnostic is not invariant to variable/objective rescaling.
        """
        p = self.p
        up = self.proximal_gradient("x", x, p.A.T @ y, w, deadline)
        vp = self.proximal_gradient("z", z, p.B.T @ y, w, deadline)
        stationarity = max(infnorm(up - x), infnorm(vp - z))
        feasibility = infnorm(p.residual(x, z))
        domain = p.violation(x, z)
        return dict(
            feasibility=feasibility,
            stationarity=stationarity,
            domain_violation=domain,
            kkt=max(feasibility, stationarity, domain),
        )

    def alm_step(self, x, z, y, rho, proximal, deadline):
        """Joint proximal minimax step for the vector augmented Lagrangian."""
        import coptpy as cp

        p = self.p
        model = self.model()
        try:
            u, f = self.block(model, "x")
            v, g = self.block(model, "z")
            r = p.A @ u + p.B @ v - p.c
            t = model.addVar(lb=-cp.COPT.INFINITY)
            base = p.values(x, z)
            constraints = [model.addConstr(f[i] + g[i] - base[i] <= t) for i in range(p.m)]
            model.setObjective(
                t
                + self.scalar(model, y @ r)
                + 0.5 * rho * self.quadratic_objective(model, r)
                + 0.5 * proximal * self.quadratic_objective(model, u - x)
                + 0.5 * proximal * self.quadratic_objective(model, v - z)
            )
            self.optimize(model, deadline)
            un, zn = np.asarray(u.x.tonumpy()), np.asarray(v.x.tonumpy())
            # The epigraph inequalities use <=, hence their minimization
            # dual multipliers have the opposite sign to simplex weights.
            weights = np.maximum(-np.array([c.pi for c in constraints]), 0.0)
            if abs(weights.sum() - 1) > 1e-5:
                raise RuntimeError(f"Invalid ALM epigraph dual weights: {weights}")
            weights /= weights.sum()
            gap = float(model.ObjVal - model.ObjBound)
            if gap > 2e-6:
                raise RuntimeError(f"ALM proximal model gap too large: {gap:g}")
            return un, zn, weights, gap
        finally:
            model.dispose()

    def reference(self, x, z, w):
        """Independent feasible scalarized optimum and nearest-feasible repair."""
        import coptpy as cp

        p = self.p
        deadline = perf_counter() + 30.0
        model = self.model()
        try:
            u, f = self.block(model, "x")
            v, g = self.block(model, "z")
            equality = model.addConstr(p.A @ u + p.B @ v == p.c)
            model.setObjective(cp.quicksum(float(w[i]) * (f[i] + g[i]) for i in range(p.m)))
            optimum, lower = self.optimize(model, deadline)
            xr, zr = np.asarray(u.x.tonumpy()), np.asarray(v.x.tonumpy())
            reference_y = -np.asarray(equality.pi.tonumpy())
        finally:
            model.dispose()
        repair = self.model()
        try:
            u, _ = self.block(repair, "x")
            v, _ = self.block(repair, "z")
            repair.addConstr(p.A @ u + p.B @ v == p.c)
            repair.setObjective(
                0.5 * self.quadratic_objective(repair, u - x)
                + 0.5 * self.quadratic_objective(repair, v - z)
            )
            self.optimize(repair, deadline)
            xp, zp = np.asarray(u.x.tonumpy()), np.asarray(v.x.tonumpy())
        finally:
            repair.dispose()
        value = float(w @ p.values(xp, zp))
        reference_certificate = self.certificate(xr, zr, reference_y, w, deadline)
        return dict(
            reference_objectives=p.values(xr, zr, False).tolist(),
            repaired_objectives=p.values(xp, zp, False).tolist(),
            reference_scalar_value=optimum,
            reference_lower_bound=lower,
            repaired_scalar_gap=value - lower,
            reference_model_gap=optimum - lower,
            reference_kkt=reference_certificate["kkt"],
            reference_x=xr.tolist(),
            reference_z=zr.tolist(),
            reference_y=reference_y.tolist(),
            repair_distance=max(infnorm(xp - x), infnorm(zp - z)),
            reference_feasibility=infnorm(p.residual(xr, zr)),
            repaired_feasibility=infnorm(p.residual(xp, zp)),
        )


def solve_admm(
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
    oracle_factory=Oracle,
):
    start = perf_counter()
    oracle = oracle_factory(p)
    x, z, alpha, beta = x0.copy(), z0.copy(), w0.copy(), w0.copy()
    y = np.zeros(len(p.c))
    history, status, reason, iterations = [], "iterations_limit", None, 0
    gaps = []
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
                    "x", x, z, y, alpha, beta, penalty, eta, kappa, inner_tol, start + seconds_limit
                )
                zn, bn, gz = oracle.local(
                    "z", z, xn, y, beta, an, penalty, eta, kappa, inner_tol, start + seconds_limit
                )
            except (TimeoutError, RuntimeError) as error:
                status = "time_limit" if isinstance(error, TimeoutError) else "inner_failure"
                reason = str(error)
                break
            x, z, alpha, beta = xn, zn, an, bn
            y += penalty * p.residual(x, z)
            iterations = k
            gaps.extend((gx, gz))
            # A certificate belongs only to the exact state where it was
            # computed. Never reuse a previous state's successful residual.
            if k % 5 == 0:
                cert = oracle.certificate(x, z, y, alpha)
                history.append(
                    dict(
                        iteration=k,
                        seconds=perf_counter() - start,
                        **cert,
                        consensus=infnorm(alpha - beta),
                    )
                )
            else:
                cert = {"kkt": np.inf}
        cert = oracle.certificate(x, z, y, alpha)
        if max(cert["kkt"], infnorm(alpha - beta)) <= tol:
            status = "converged"
        elapsed = perf_counter() - start
        history.append(
            dict(iteration=iterations, seconds=elapsed, **cert, consensus=infnorm(alpha - beta))
        )
        return dict(
            method="Consensus-ADMM-domain",
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
            max_inner_gap=max(gaps, default=0.0),
            solves=oracle.solves,
            barrier_iterations=oracle.barrier_iterations,
            native_status_counts=oracle.native_status_counts,
            max_native_gap=oracle.max_native_gap,
        )
    finally:
        oracle.close()


def solve_alm(
    p,
    x0,
    z0,
    w0,
    penalty=1.0,
    tol=1e-4,
    seconds_limit=25.0,
    max_outer=80,
    max_inner=500,
    proximal=1.0,
    oracle_factory=Oracle,
):
    """MO-ALM with proximal-minimax inner descent and safeguarded multipliers.

    Inner tolerances decrease geometrically; rho doubles when affine residual
    fails to contract. Both blocks are optimized jointly in each inner step.
    """
    start = perf_counter()
    oracle = oracle_factory(p)
    x, z, w, y = x0.copy(), z0.copy(), w0.copy(), np.zeros(len(p.c))
    rho, previous_feasibility, inner_tol = penalty, np.inf, 0.1
    history, status, reason, steps, outer = [], "iterations_limit", None, 0, 0
    max_gap = 0.0
    try:
        cert = oracle.certificate(x, z, y, w)
        for outer in range(1, max_outer + 1):
            if cert["kkt"] <= tol:
                status = "converged"
                break
            for _inner in range(max_inner):
                if perf_counter() - start >= seconds_limit:
                    status = "time_limit"
                    break
                try:
                    xn, zn, wn, gap = oracle.alm_step(x, z, y, rho, proximal, start + seconds_limit)
                except (TimeoutError, RuntimeError) as error:
                    status = "time_limit" if isinstance(error, TimeoutError) else "inner_failure"
                    reason = str(error)
                    break
                mapping = proximal * max(infnorm(xn - x), infnorm(zn - z))
                x, z, w = xn, zn, wn
                steps += 1
                max_gap = max(max_gap, gap)
                trial_y = y + rho * p.residual(x, z)
                if steps % 5 == 0 or mapping <= inner_tol:
                    cert = oracle.certificate(x, z, trial_y, w)
                    history.append(
                        dict(
                            iteration=steps,
                            outer=outer,
                            seconds=perf_counter() - start,
                            **cert,
                            penalty=rho,
                        )
                    )
                    if cert["kkt"] <= tol:
                        y, status = trial_y, "converged"
                        break
                if mapping <= inner_tol:
                    break
            if status != "iterations_limit":
                break
            # Update only after the current inner solve has met its tolerance.
            if mapping > inner_tol:
                status, reason = "inner_iterations_limit", "ALM inner mapping above tolerance"
                break
            y = y + rho * p.residual(x, z)
            feasibility = infnorm(p.residual(x, z))
            if feasibility > 0.5 * previous_feasibility:
                rho = min(2 * rho, 1e6)
            previous_feasibility = feasibility
            inner_tol = max(0.1 * tol, 0.5 * inner_tol)
            cert = oracle.certificate(x, z, y, w)
        cert = oracle.certificate(x, z, y, w)
        if cert["kkt"] <= tol:
            status = "converged"
        elapsed = perf_counter() - start
        history.append(dict(iteration=steps, outer=outer, seconds=elapsed, **cert, penalty=rho))
        return dict(
            method="MO-ALM-PP-domain",
            x=x,
            z=z,
            y=y,
            w=w,
            beta=w.copy(),
            status=status,
            reason=reason,
            iterations=outer,
            inner_steps=steps,
            seconds=elapsed,
            history=history,
            **cert,
            consensus=0.0,
            residual=cert["kkt"],
            penalty=penalty,
            final_penalty=rho,
            max_inner_gap=max_gap,
            solves=oracle.solves,
            barrier_iterations=oracle.barrier_iterations,
            native_status_counts=oracle.native_status_counts,
            max_native_gap=oracle.max_native_gap,
        )
    finally:
        oracle.close()
