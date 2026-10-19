"""Paper equations (4),(6),(7): batched graph-form ADMM on weighted dual LPs."""

from time import perf_counter

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .oracle import CachedOracle


def solve_batch(
    case,
    weights,
    x0,
    penalty=1.0,
    tol=1e-4,
    seconds_limit=30.0,
    max_iterations=50000,
    check_every=25,
):
    tick = perf_counter()
    A0 = case.dual_A
    row_scale = 1 / np.maximum(np.linalg.norm(A0, axis=1), 1e-12)
    A = row_scale[:, None] * A0
    factor = cho_factor(np.eye(len(A)) + A @ A.T, lower=True, check_finite=False)
    setup = perf_counter() - tick
    checker = CachedOracle(case.problem)
    weights = np.atleast_2d(weights)
    batch = len(weights)
    bounds = [case.dual_bounds(w) for w in weights]
    b = np.array([q[0] for q in bounds]).T * row_scale[:, None]
    upper = np.array([q[1] for q in bounds]).T
    z1 = np.zeros((A.shape[1], batch))
    z2 = np.minimum(np.zeros_like(b), b)
    y1 = np.zeros_like(z1)
    # The paper recovers physical time variables from rho*y2 after undoing row scaling.
    y2 = np.tile(np.asarray(x0)[:, None] / (penalty * row_scale[:, None]), (1, batch))
    done = np.zeros(batch, dtype=bool)
    results = [None] * batch
    traces = [[] for _ in range(batch)]
    c = case.dual_c[:, None]
    reason = "iterations_limit"
    for iteration in range(1, max_iterations + 1):
        active = np.flatnonzero(~done)
        if not len(active):
            break
        if perf_counter() - tick >= seconds_limit:
            reason = "time_limit"
            break
        za = z1[:, active] - y1[:, active] - c / penalty
        v = cho_solve(factor, A @ za - z2[:, active] + y2[:, active], check_finite=False)
        x1 = za - A.T @ v
        x2 = v + z2[:, active] - y2[:, active]
        zz1 = np.clip(x1 + y1[:, active], 0.0, upper[:, active])
        zz2 = np.minimum(x2 + y2[:, active], b[:, active])
        y1[:, active] += x1 - zz1
        y2[:, active] += x2 - zz2
        z1[:, active] = zz1
        z2[:, active] = zz2
        if iteration % check_every == 0:
            for j in active:
                raw_x = row_scale * penalty * y2[:, j]
                phys_x, dose = case.physical(raw_x)
                objs = case.objectives(phys_x)
                dual_violation = float(
                    np.max(np.maximum(A0 @ z1[:, j] - bounds[j][0], 0.0), initial=0.0)
                )
                primal_violation = max(
                    case.problem.X.violation(raw_x),
                    case.problem.Z.violation(case.problem.A @ raw_x),
                )
                primal = float(weights[j] @ objs)
                dual = -float(case.dual_c @ z1[:, j])
                gap = abs(primal - dual) / (1 + abs(primal))
                criterion = max(primal_violation, dual_violation, gap)
                if criterion <= tol:
                    cert = checker.certificate(
                        raw_x,
                        case.problem.A @ raw_x,
                        case.coupling_multiplier(z1[:, j]),
                        weights[j],
                    )
                    criterion = max(criterion, cert["kkt"])
                traces[j].append(
                    dict(
                        iteration=iteration,
                        seconds=perf_counter() - tick,
                        residual=criterion,
                        primal_violation=primal_violation,
                        dual_violation=dual_violation,
                        gap=gap,
                    )
                )
                if criterion <= tol:
                    done[j] = True
                    results[j] = dict(
                        status="converged",
                        x=raw_x,
                        z=case.problem.A @ raw_x,
                        y=case.coupling_multiplier(z1[:, j]),
                        w=weights[j],
                        beta=weights[j],
                        dual=z1[:, j].copy(),
                        iterations=iteration,
                        residual=criterion,
                        primal_dual_gap=primal - dual,
                        history=traces[j],
                    )
    elapsed = perf_counter() - tick
    checker.close()
    for j in range(batch):
        if results[j] is None:
            raw_x = row_scale * penalty * y2[:, j]
            results[j] = dict(
                status=reason,
                x=raw_x,
                z=case.problem.A @ raw_x,
                y=case.coupling_multiplier(z1[:, j]),
                w=weights[j],
                beta=weights[j],
                dual=z1[:, j].copy(),
                iterations=iteration,
                residual=traces[j][-1]["residual"] if traces[j] else np.inf,
                history=traces[j],
            )
        results[j].update(
            seconds=elapsed / batch,
            batch_seconds=elapsed,
            setup_seconds=setup,
            batch_size=batch,
            method="Paper scalarization + dual graph ADMM",
        )
    return results
