"""Minimax FW and convex-hull away/pairwise directions for vector objectives."""

from dataclasses import dataclass

import numpy as np

from ..core import _readonly
from ._runtime import RUN_ERRORS, Run, options_with_L
from ._subproblem import SubproblemError
from .first_order import SolverOptions


@dataclass(frozen=True)
class Polytope:
    """C = conv(vertices), one vertex per row. No implicit vertex enumeration."""

    vertices: np.ndarray

    def __post_init__(self):
        V = np.asarray(self.vertices)
        if np.iscomplexobj(V) or V.ndim != 2 or min(V.shape) < 1 or not np.isfinite(V).all():
            raise ValueError("vertices must be a finite real (n_vertices,n_var) array")
        object.__setattr__(self, "vertices", _readonly(V))

    def decompose(self, x):
        from scipy.optimize import linprog

        V = self.vertices
        result = linprog(
            np.zeros(len(V)),
            A_eq=np.vstack((V.T, np.ones(len(V)))),
            b_eq=np.r_[x, 1],
            bounds=(0, None),
            method="highs",
        )
        if not result.success or np.max(np.abs(result.x @ V - x)) > 1e-8:
            raise ValueError("x0 is not in the supplied convex hull")
        weights = np.maximum(result.x, 0)
        return weights / weights.sum()


@dataclass(frozen=True)
class FWOptions(SolverOptions):
    step_rule: str = "adaptive"
    subproblem_tol: float = 1e-9

    def __post_init__(self):
        super().__post_init__()
        if self.step_rule not in ("diminishing", "adaptive", "armijo"):
            raise ValueError("FW supports diminishing, adaptive, armijo")


def simplex_minimax(payoff, groups=None, *, tol=1e-9):
    """LP on one simplex or a product of two simplexes, with a dual bound."""
    from scipy.optimize import linprog

    m, n = payoff.shape
    groups = [np.arange(n)] if groups is None else groups
    Aeq = np.zeros((len(groups), n + 1))
    for row, group in zip(Aeq, groups, strict=True):
        row[group] = 1
    result = linprog(
        np.r_[np.zeros(n), 1.0],
        A_ub=np.column_stack((payoff, -np.ones(m))),
        b_ub=np.zeros(m),
        A_eq=Aeq,
        b_eq=np.ones(len(groups)),
        bounds=[(0, None)] * n + [(None, None)],
        method="highs",
        options={
            "dual_feasibility_tolerance": max(tol, 1e-10),
            "primal_feasibility_tolerance": max(tol, 1e-10),
        },
    )
    if not result.success:
        raise SubproblemError(f"Minimax LP failed: {result.message}")
    w = np.maximum(result.x[:-1], 0)
    for group in groups:
        w[group] /= w[group].sum()
    lam = np.maximum(-result.ineqlin.marginals, 0)
    if lam.sum() == 0:
        raise SubproblemError("LP returned no objective simplex multiplier")
    lam /= lam.sum()
    theta = float(np.max(payoff @ w))
    dual = float(sum(np.min(lam @ payoff[:, group]) for group in groups))
    if theta - dual > 10 * tol * max(1, abs(theta), abs(dual)):
        raise SubproblemError("Minimax LP primal-dual gap is above tolerance")
    return w, theta, dual, int(result.nit)


def box_minimax(J, x, lower, upper, tol):
    from scipy.optimize import linprog

    m, n = J.shape
    result = linprog(
        np.r_[np.zeros(n), 1.0],
        A_ub=np.column_stack((J, -np.ones(m))),
        b_ub=J @ x,
        bounds=list(zip(lower, upper, strict=True)) + [(None, None)],
        method="highs",
        options={
            "dual_feasibility_tolerance": max(tol, 1e-10),
            "primal_feasibility_tolerance": max(tol, 1e-10),
        },
    )
    if not result.success:
        raise SubproblemError(f"Box minimax LP failed: {result.message}")
    s = np.clip(result.x[:-1], lower, upper)
    lam = np.maximum(-result.ineqlin.marginals, 0)
    if lam.sum() == 0:
        raise SubproblemError("Box LP returned no objective multiplier")
    lam /= lam.sum()
    combined = lam @ J
    extreme = np.where(combined >= 0, lower, upper)
    dual = float(combined @ (extreme - x))
    theta = float(np.max(J @ (s - x)))
    if theta - dual > 10 * tol * max(1, abs(theta), abs(dual)):
        raise SubproblemError("Box LP primal-dual gap is above tolerance")
    return s, theta, dual, int(result.nit)


def coefficient_cap(weights, change):
    negative = change < 0
    return float(np.min(-weights[negative] / change[negative], initial=np.inf))


class FrankWolfe:
    name = "frank_wolfe"
    variant = "fw"

    def __init__(self, *, domain=None, **options):
        if domain is not None and not isinstance(domain, Polytope):
            raise TypeError("domain must be Polytope, or None for the problem box")
        if self.variant != "fw" and domain is None:
            raise ValueError("Away/pairwise FW require an explicit Polytope(vertices)")
        self.domain = domain
        self.options = FWOptions(**options_with_L(options))
        if domain is None and self.options.bounds != "box":
            raise ValueError("FW requires a bounded domain")
        if self.variant != "fw" and self.options.step_rule == "diminishing":
            raise ValueError("Away/pairwise variants use adaptive or armijo steps")

    def solve(self, problem, x0):
        r = Run(problem, x0, self.options, self.name)
        o = r.o
        V = None if self.domain is None else self.domain.vertices
        weights = None
        if V is not None:
            if V.shape[1] != problem.n_var:
                raise ValueError("Polytope dimension differs from problem.n_var")
            if o.bounds == "box" and (np.any(V < problem.xl) or np.any(V > problem.xu)):
                raise ValueError("Polytope vertices must lie inside the problem box")
            weights = self.domain.decompose(r.x)
        M = 1 / o.step_size

        def lp(payoff, groups=None):
            r.counts["nsubproblem"] += 1
            answer = simplex_minimax(payoff, groups, tol=o.subproblem_tol)
            r.counts["subproblem_iterations"] += answer[-1]
            return answer

        try:
            for k in range(o.max_iter + 1):
                ev, jac = r.joint(r.x)
                J, F = jac.dF[0], ev.F[0]
                if V is None:
                    r.counts["nsubproblem"] += 1
                    point, theta, dual, nit = box_minimax(
                        J, r.x, r.lower, r.upper, o.subproblem_tol
                    )
                    r.counts["subproblem_iterations"] += nit
                    direction, change = point - r.x, None
                else:
                    forward, theta, dual, _ = lp(J @ (V - r.x).T)
                    change = forward - weights
                    direction = change @ V
                r.residual, r.bound, r.gap = (
                    max(0.0, -theta),
                    max(0.0, -dual),
                    max(0.0, theta - dual),
                )
                r.details.update(residual_kind="common_fw_gap", weights=weights)
                r.record()
                if r.bound <= o.tol:
                    r.stop("stationary", "Common FW gap upper bound below tolerance.")
                    break
                if k == o.max_iter:
                    break
                kind, cap = "fw", 1.0
                if self.variant != "fw":
                    active = np.flatnonzero(weights > 0)
                    if self.variant == "away":
                        away, away_theta, _, _ = lp(J @ (r.x - V[active]).T)
                        if away_theta < theta:
                            dist = np.zeros(len(V))
                            dist[active] = away
                            change, kind = weights - dist, "away"
                            cap = coefficient_cap(weights, change)
                    else:
                        payoff = np.column_stack((J @ V.T, -J @ V[active].T))
                        dist, _, _, _ = lp(
                            payoff, [np.arange(len(V)), np.arange(len(V), len(V) + len(active))]
                        )
                        change = dist[: len(V)].copy()
                        change[active] -= dist[len(V) :]
                        cap, kind = coefficient_cap(weights, change), "pairwise"
                    direction = change @ V
                slope = float(np.max(J @ direction))
                if slope >= 0 or not np.isfinite(cap) or cap <= 0:
                    r.stop("no_descent", "No feasible strict common descent direction.")
                    break
                norm2 = float(direction @ direction)
                trial_M, alpha, accepted = M * o.contraction, min(1.0, cap), False
                for _ in range(o.max_line_search):
                    if o.step_rule == "adaptive":
                        alpha = min(cap, -slope / (trial_M * norm2))
                    elif o.step_rule == "diminishing":
                        alpha = min(cap, 2 / (k + 2))
                    if alpha < o.min_step:
                        break
                    if V is None:
                        trial = r.x + alpha * direction
                        new_weights = None
                    else:
                        new_weights = np.maximum(weights + alpha * change, 0)
                        new_weights /= new_weights.sum()
                        trial = new_weights @ V
                    trial_ev = r.values(trial)
                    delta = trial - r.x
                    slack = 32 * np.finfo(float).eps * np.maximum(1, np.abs(F))
                    if o.step_rule == "diminishing":
                        accepted = True
                    elif o.step_rule == "adaptive":
                        accepted = bool(
                            np.all(
                                trial_ev.F[0]
                                <= F + J @ delta + 0.5 * trial_M * (delta @ delta) + slack
                            )
                        )
                    else:
                        accepted = bool(np.all(trial_ev.F[0] <= F + o.armijo * (J @ delta) + slack))
                    if accepted:
                        break
                    trial_M /= o.contraction
                    alpha *= o.contraction
                if not accepted or np.array_equal(trial, r.x):
                    r.stop(
                        "line_search_failed", "No acceptable distinct conditional-gradient step."
                    )
                    break
                M = trial_M if o.step_rule == "adaptive" else M
                weights = new_weights
                r.accept(
                    trial,
                    trial_ev,
                    alpha,
                    direction_type=kind,
                    step_cap=cap,
                    smoothness=M,
                    weights=weights,
                )
        except RUN_ERRORS as error:
            r.failed(error)
        return r.result(domain="box" if V is None else "convex_hull", weights=weights)


class AwayStepFrankWolfe(FrankWolfe):
    """Convex-hull away minimax direction (CH-AFW)."""

    name, variant = "away_frank_wolfe", "away"


class PairwiseFrankWolfe(FrankWolfe):
    """Joint minimax forward/away distributions (multiobjective PFW)."""

    name, variant = "pairwise_frank_wolfe", "pairwise"
