"""Two-block vector ADMM with consensus objective weights and residual balancing."""

from dataclasses import dataclass

import numpy as np

from ..core import Problem, _readonly, integer
from ._runtime import RUN_ERRORS, Run, options_with_L, positive
from ._subproblem import SubproblemError
from .augmented import infnorm
from .first_order import SolverOptions


class SplitProblem(Problem):
    """F_i(x,z)=first_i(x)+second_i(z), Ax+Bz=c, with each block's box.

    Both blocks are smooth Problem instances with the same objective count.
    Additional G/H constraints belong in ALM, not in the ADMM block oracle.
    """

    def __init__(self, first, second, A, B, c, *, name="split_problem"):
        if first.n_obj != second.n_obj:
            raise ValueError("ADMM blocks must have the same number of objectives")
        if any(p.n_ieq_constr or p.n_eq_constr for p in (first, second)):
            raise ValueError("ADMM blocks may have boxes but no additional G/H constraints")
        A, B, c = (np.asarray(t, dtype=float) for t in (A, B, c))
        if (
            c.ndim != 1
            or len(c) == 0
            or A.shape != (len(c), first.n_var)
            or B.shape != (len(c), second.n_var)
            or not all(np.isfinite(t).all() for t in (A, B, c))
        ):
            raise ValueError("Require finite A:(p,nx), B:(p,nz), c:(p,), p>=1")
        self.first, self.second = first, second
        self.A, self.B, self.c = map(_readonly, (A, B, c))
        self.n_first = first.n_var
        super().__init__(
            name,
            first.n_var + second.n_var,
            first.n_obj,
            np.r_[first.xl, second.xl],
            np.r_[first.xu, second.xu],
            n_eq_constr=len(c),
        )

    def _evaluate(self, X):
        x, z = X[:, : self.n_first], X[:, self.n_first :]
        f = self.first._evaluate(x)[0] + self.second._evaluate(z)[0]
        return f, None, x @ self.A.T + z @ self.B.T - self.c


def simplex_projection(v):
    """Euclidean simplex projection, with deterministic tie handling."""
    v = np.asarray(v, dtype=float)
    ordered = np.sort(v)[::-1]
    thresholds = (np.cumsum(ordered) - 1) / np.arange(1, len(v) + 1)
    last = np.flatnonzero(ordered > thresholds)[-1]
    return np.maximum(v - thresholds[last], 0.0)


def balanced_penalty(
    rho, primal, dual, *, ratio=10.0, increase=2.0, decrease=2.0, minimum=1e-8, maximum=1e8
):
    """Boyd residual balancing; the caller keeps its unscaled multiplier unchanged."""
    if primal > ratio * dual:
        return min(maximum, rho * increase)
    if dual > ratio * primal:
        return max(minimum, rho / decrease)
    return rho


@dataclass(frozen=True)
class ADMMOptions(SolverOptions):
    max_iter: int = 500
    max_nfev: int = 50000
    max_njev: int = 50000
    step_rule: str = "backtracking"
    penalty: float = 1.0
    penalty_min: float = 1e-8
    penalty_max: float = 1e8
    residual_ratio: float = 10.0
    penalty_increase: float = 2.0
    penalty_decrease: float = 2.0
    weight_prox: float = 1.0
    inner_tol: float = 1e-8
    max_inner: int = 300

    def __post_init__(self):
        super().__post_init__()
        if self.step_rule != "backtracking":
            raise ValueError("ADMM uses the bounded smooth inner solver's line search")
        for key in ("penalty", "penalty_min", "penalty_max", "weight_prox", "inner_tol"):
            positive(getattr(self, key), key)
        if not self.penalty_min <= self.penalty <= self.penalty_max:
            raise ValueError("Require penalty_min <= penalty <= penalty_max")
        for key in ("residual_ratio", "penalty_increase", "penalty_decrease"):
            if not getattr(self, key) > 1 or not np.isfinite(getattr(self, key)):
                raise ValueError(f"{key} must be finite and >1")
        integer(self.max_inner, "max_inner")


class MultiobjectiveADMM:
    """Implicit, consistency-only objective-weight consensus ADMM for smooth blocks.

    Each block solves the primal envelope of a simplex-concave saddle problem.
    This is not ADMM applied to one fixed weighted sum of the objectives.
    """

    name = "admm"

    def __init__(self, **options):
        self.options = ADMMOptions(**options_with_L(options))

    def solve(self, problem, x0):
        from scipy.optimize import minimize as scalar_minimize

        if not isinstance(problem, SplitProblem):
            raise TypeError("ADMM requires SplitProblem(first, second, A, B, c)")
        r = Run(problem, x0, self.options, self.name, constraints=True)
        o, nx = r.o, problem.n_first
        A, B, c = problem.A, problem.B, problem.c
        alpha = beta = np.full(problem.n_obj, 1 / problem.n_obj)
        y, rho = np.zeros(len(c)), o.penalty
        inner_iterations = block_calls = 0

        def block_values(block, u):
            nonlocal block_calls
            r.charge(jac=True)
            block_calls += 1
            ev, jac = block.evaluate_with_jacobian(u, check_bounds=False)
            return ev.F[0], jac.dF[0]

        def block_solve(block, old, M, fixed, reference, centre, lower, upper):
            nonlocal inner_iterations
            r.counts["nsubproblem"] += 1
            cache = {}

            def envelope(u):
                if "x" in cache and np.array_equal(cache["x"], u):
                    return cache["value"], cache["grad"]
                f, J = block_values(block, u)
                delta = f - reference
                weights = simplex_projection(centre + delta / o.weight_prox)
                residual = M @ u + fixed
                value = (
                    y @ residual
                    + 0.5 * rho * (residual @ residual)
                    + weights @ delta
                    - 0.5 * o.weight_prox * np.sum((weights - centre) ** 2)
                )
                grad = weights @ J + M.T @ (y + rho * residual)
                cache.update(x=u.copy(), value=float(value), grad=grad, weights=weights)
                return float(value), grad

            result = scalar_minimize(
                envelope,
                old,
                jac=True,
                method="L-BFGS-B",
                bounds=list(zip(lower, upper, strict=True)),
                options={
                    "gtol": 0.1 * min(o.inner_tol, o.tol),
                    "ftol": 1e-15,
                    "maxiter": o.max_inner,
                    "maxls": 40,
                },
            )
            inner_iterations += result.nit
            r.counts["subproblem_iterations"] += result.nit
            _, gradient = envelope(result.x)
            error = infnorm(result.x - np.clip(result.x - gradient, lower, upper))
            if error > min(o.inner_tol, 0.1 * o.tol):
                raise SubproblemError(
                    f"ADMM block projected KKT residual {error:.3g} exceeds tolerance"
                )
            return result.x, cache["weights"], error

        def check(ev, jac, dual_norm):
            weights = 0.5 * (alpha + beta)
            gradient = weights @ jac.dF[0] + y @ jac.dH[0]
            stationarity = infnorm(r.x - np.clip(r.x - gradient, r.lower, r.upper))
            primal = float(np.linalg.norm(ev.H[0]))
            consensus = infnorm(alpha - beta)
            r.residual = stationarity
            r.bound = max(stationarity, primal, dual_norm, consensus)
            r.details.update(
                optimality=stationarity,
                primal_residual=primal,
                dual_residual=dual_norm,
                weight_consensus=consensus,
                alpha=alpha,
                beta=beta,
                objective_weights=weights,
                multiplier=y,
                penalty=rho,
                residual_kind="split_kkt_and_consensus",
            )
            return r.bound <= o.tol

        try:
            ev, jac = r.joint(r.x)
            dual_norm = 0.0
            for outer in range(o.max_iter + 1):
                converged = check(ev, jac, dual_norm)
                r.record()
                if converged:
                    r.stop(
                        "stationary",
                        "Split KKT, primal/dual residuals and weight consensus passed.",
                    )
                    break
                if outer == o.max_iter:
                    break
                x, z = r.x[:nx], r.x[nx:]
                f, _ = block_values(problem.first, x)
                g, _ = block_values(problem.second, z)
                x_new, alpha_new, error_x = block_solve(
                    problem.first, x, A, B @ z - c, f, beta, r.lower[:nx], r.upper[:nx]
                )
                z_new, beta_new, error_z = block_solve(
                    problem.second, z, B, A @ x_new - c, g, alpha_new, r.lower[nx:], r.upper[nx:]
                )
                candidate = np.r_[x_new, z_new]
                # Complete all fallible oracle work before publishing an outer iterate.
                new_ev, new_jac = r.joint(candidate)
                residual = A @ x_new + B @ z_new - c
                new_y = y + rho * residual
                dual_norm = float(np.linalg.norm(rho * A.T @ B @ (z_new - z)))
                next_rho = balanced_penalty(
                    rho,
                    np.linalg.norm(residual),
                    dual_norm,
                    ratio=o.residual_ratio,
                    increase=o.penalty_increase,
                    decrease=o.penalty_decrease,
                    minimum=o.penalty_min,
                    maximum=o.penalty_max,
                )
                r.accept(
                    candidate,
                    new_ev,
                    1.0,
                    penalty_used=rho,
                    penalty_next=next_rho,
                    inner_residual_x=error_x,
                    inner_residual_z=error_z,
                )
                y, rho, alpha, beta = new_y, next_rho, alpha_new, beta_new
                ev, jac = new_ev, new_jac
        except RUN_ERRORS as error:
            r.failed(error)
        return r.result(
            inner_iterations=inner_iterations, block_oracle_calls=block_calls, final_penalty=rho
        )
