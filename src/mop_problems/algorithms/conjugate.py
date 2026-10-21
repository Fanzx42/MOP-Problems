"""Vector nonlinear CG with explicit beta formula and restart policies."""

from dataclasses import dataclass

import numpy as np

from ..core import integer
from ._runtime import RUN_ERRORS, Run, feasible_step, options_with_L, positive
from .first_order import SolverOptions


def conjugate_beta(rule, J, old_J, v, old_v, old_d):
    """Directional-max generalizations; v and d include the descent minus sign.

    No silent absolute-value denominator. A degenerate denominator requests a restart.
    FR/CD/DY/PRP/HS follow the local vector-CG convention; LS extends its numerator.
    HZ uses the vector directional-max expression (not a changing scalarization).
    """
    a, b = float(np.max(J @ v)), float(np.max(old_J @ v))
    c, d = float(np.max(old_J @ old_v)), float(np.max(J @ old_d))
    e, f = float(np.max(old_J @ old_d)), float(np.max(J @ old_v))
    base = rule.rstrip("+")
    numerator = {
        "FR": -a,
        "CD": -a,
        "DY": -a,
        "PRP": -a + b,
        "HS": -a + b,
        "LS": -a + b,
        "HZ": -a + b,
    }[base]
    denominator = {"FR": -c, "CD": -e, "DY": d - e, "PRP": -c, "HS": d - e, "LS": -e, "HZ": d - e}[
        base
    ]
    scale = max(abs(a), abs(b), abs(c), abs(d), abs(e), np.finfo(float).tiny)
    if denominator <= 64 * np.finfo(float).eps * scale:
        return 0.0, "denominator"
    beta = numerator / denominator
    if base == "HZ":
        beta -= 2 * d * (-a + b + f - c) / denominator**2
    if rule.endswith("+"):
        beta = max(0.0, beta)
    if not np.isfinite(beta):
        return 0.0, "nonfinite_beta"
    return float(beta), ""


@dataclass(frozen=True)
class CGOptions(SolverOptions):
    step_rule: str = "strong_wolfe"
    bounds: str = "none"
    armijo: float = 1e-4
    beta_rule: str = "PRP+"
    wolfe_c2: float = 0.1
    restart_interval: int | None = None
    powell_restart: float | None = 0.2
    descent_fraction: float = 0.01

    def __post_init__(self):
        super().__post_init__()
        if self.step_rule not in ("armijo", "wolfe", "strong_wolfe"):
            raise ValueError("CG supports armijo, wolfe, strong_wolfe")
        if self.beta_rule not in ("FR", "CD", "DY", "PRP", "PRP+", "HS", "HS+", "LS", "HZ", "HZ+"):
            raise ValueError("Unknown CG beta_rule")
        if not self.armijo < self.wolfe_c2 < 1:
            raise ValueError("require armijo < wolfe_c2 < 1")
        if self.restart_interval is not None:
            integer(self.restart_interval, "restart_interval", 0)
        if self.powell_restart is not None:
            positive(self.powell_restart, "powell_restart")
        if not 0 < self.descent_fraction <= 1:
            raise ValueError("descent_fraction must be in (0,1]")


def vector_search(run, F, J, direction, *, armijo_only=False):
    o, x = run.o, run.x
    slope = float(np.max(J @ direction))
    if slope >= 0:
        return None
    cap = feasible_step(x, direction, run.lower, run.upper)
    alpha, low, high = min(o.step_size, cap), 0.0, np.inf
    for _ in range(o.max_line_search):
        if alpha < o.min_step or np.array_equal(x + alpha * direction, x):
            return None
        trial = np.clip(x + alpha * direction, run.lower, run.upper)
        try:
            if armijo_only or o.step_rule == "armijo":
                ev, jt = run.values(trial), None
            else:
                ev, jt = run.joint(trial)
        except FloatingPointError:
            high = alpha
            alpha = (low + high) / 2
            continue
        slack = 16 * np.finfo(float).eps * np.maximum(1, np.abs(F))
        if not np.all(ev.F[0] <= F + o.armijo * alpha * slope + slack):
            high = alpha
        elif jt is None:
            return trial, ev, alpha
        else:
            new_slope = float(np.max(jt.dF[0] @ direction))
            curvature = (
                abs(new_slope) <= o.wolfe_c2 * abs(slope)
                if o.step_rule == "strong_wolfe"
                else new_slope >= o.wolfe_c2 * slope
            )
            if curvature:
                return trial, ev, alpha
            if new_slope >= 0:
                high = alpha
            else:
                low = alpha
        next_alpha = (low + high) / 2 if np.isfinite(high) else min(2 * alpha, cap)
        if next_alpha == alpha:
            return None
        alpha = next_alpha
    return None


class ConjugateGradient:
    name = "conjugate_gradient"

    def __init__(self, **options):
        self.options = CGOptions(**options_with_L(options))

    def solve(self, problem, x0):
        r = Run(problem, x0, self.options, self.name)
        o = r.o
        interval = problem.n_var if o.restart_interval is None else o.restart_interval
        old_J = old_v = old_d = None
        restart_counts = {}
        try:
            for k in range(o.max_iter + 1):
                ev, jac = r.joint(r.x)
                J, F = jac.dF[0], ev.F[0]
                model = r.stationarity(J)
                r.record()
                if r.bound <= o.tol:
                    r.stop("stationary", "Unit-step Pareto stationarity residual below tolerance.")
                    break
                if k == o.max_iter:
                    break
                v = model.x - r.x
                beta, reason = 0.0, "initial"
                if old_J is not None:
                    beta, reason = conjugate_beta(o.beta_rule, J, old_J, v, old_v, old_d)
                    if interval and k % interval == 0:
                        beta, reason = 0.0, "periodic"
                    elif o.powell_restart is not None and abs(v @ old_v) >= o.powell_restart * (
                        v @ v
                    ):
                        beta, reason = 0.0, "powell"
                direction = v if old_d is None else v + beta * old_d
                if (
                    np.max(J @ direction) > o.descent_fraction * np.max(J @ v)
                    or feasible_step(r.x, direction, r.lower, r.upper) <= o.min_step
                ):
                    direction, beta, reason = v, 0.0, "descent_or_boundary"
                found = vector_search(r, F, J, direction)
                if found is None:
                    direction, beta, reason = v, 0.0, "line_search"
                    found = vector_search(r, F, J, v, armijo_only=True)
                if found is None:
                    r.stop(
                        "line_search_failed", "CG and restarted steepest-descent searches failed."
                    )
                    break
                if reason and reason != "initial":
                    restart_counts[reason] = restart_counts.get(reason, 0) + 1
                old_J, old_v, old_d = J.copy(), v.copy(), direction.copy()
                trial, trial_ev, alpha = found
                r.accept(trial, trial_ev, alpha, beta=beta, restart=reason, direction=direction)
        except RUN_ERRORS as error:
            r.failed(error)
        return r.result(restart_counts=restart_counts, total_restarts=sum(restart_counts.values()))
