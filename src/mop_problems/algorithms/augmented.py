"""Vector PHR augmented Lagrangian with safeguarded multiplier/penalty updates."""

from dataclasses import dataclass

import numpy as np

from ..core import integer
from ._runtime import RUN_ERRORS, Run, options_with_L, positive
from .first_order import SolverOptions


def infnorm(x):
    return float(np.max(np.abs(x), initial=0.0))


def phr_terms(ev, jac, inequality, equality, rho):
    """Common PHR penalty and gradient; G<=0, H=0, unscaled multipliers."""
    G, H = ev.G[0], ev.H[0]
    new_inequality = np.maximum(inequality + rho * G, 0)
    new_equality = equality + rho * H
    penalty = (
        (new_inequality @ new_inequality - inequality @ inequality) / (2 * rho)
        + equality @ H
        + 0.5 * rho * (H @ H)
    )
    if jac is None:
        return float(penalty), None, new_inequality, new_equality
    derivative = new_inequality @ jac.dG[0] + new_equality @ jac.dH[0]
    return float(penalty), derivative, new_inequality, new_equality


def alm_penalty_update(rho, violation, previous, contraction=0.5, growth=2.0, maximum=1e8):
    return min(maximum, growth * rho) if violation > contraction * previous else rho


@dataclass(frozen=True)
class ALMOptions(SolverOptions):
    max_iter: int = 80
    max_nfev: int = 50000
    max_njev: int = 10000
    step_rule: str = "backtracking"
    penalty: float = 1.0
    penalty_growth: float = 2.0
    penalty_max: float = 1e8
    feasibility_contraction: float = 0.5
    multiplier_bound: float = 1e6
    inner_tol: float = 0.1
    inner_decay: float = 0.5
    max_inner: int = 300

    def __post_init__(self):
        super().__post_init__()
        if self.step_rule != "backtracking":
            raise ValueError("ALM uses backtracking for its vector proximal-gradient inner solve")
        for key in ("penalty", "penalty_max", "multiplier_bound", "inner_tol"):
            positive(getattr(self, key), key)
        if not self.penalty_growth > 1 or not np.isfinite(self.penalty_growth):
            raise ValueError("penalty_growth must be finite and >1")
        if self.penalty_max < self.penalty:
            raise ValueError("penalty_max must be >= penalty")
        if not 0 < self.feasibility_contraction < 1 or not 0 < self.inner_decay < 1:
            raise ValueError("contraction and inner_decay must lie in (0,1)")
        integer(self.max_inner, "max_inner")


class AugmentedLagrangian:
    """Solve a vector augmented objective to approximate Pareto stationarity each outer step."""

    name = "alm"

    def __init__(self, *, regularizer=None, **options):
        self.regularizer = regularizer
        self.options = ALMOptions(**options_with_L(options))

    def solve(self, problem, x0):
        r = Run(problem, x0, self.options, self.name, self.regularizer, constraints=True)
        o = r.o
        inequality = np.zeros(problem.n_ieq_constr)
        equality = np.zeros(problem.n_eq_constr)
        rho, previous, inner_steps = o.penalty, np.inf, 0

        def check(ev, jac, lam, nu):
            augmented_J = jac.dF[0] + lam @ jac.dG[0] + nu @ jac.dH[0]
            model = r.stationarity(augmented_J)
            optimality = r.bound
            feasibility = max(float(np.max(ev.G, initial=0.0)), infnorm(ev.H))
            complementarity = infnorm(np.minimum(-ev.G[0], lam))
            r.bound = max(optimality, feasibility, complementarity)
            r.details.update(
                optimality=optimality,
                feasibility=feasibility,
                complementarity=complementarity,
                penalty=rho,
                inequality_multipliers=lam,
                equality_multipliers=nu,
                objective_weights=model.weights,
                residual_kind="projected_pareto_kkt",
            )
            return r.bound <= o.tol

        try:
            ev, jac = r.joint(r.x)
            for outer in range(o.max_iter + 1):
                converged = check(ev, jac, inequality, equality)
                r.record()
                if converged:
                    r.stop(
                        "stationary", "Pareto stationarity, feasibility and complementarity passed."
                    )
                    break
                if outer == o.max_iter:
                    break
                lam_bar = np.clip(inequality, 0, o.multiplier_bound)
                nu_bar = np.clip(equality, -o.multiplier_bound, o.multiplier_bound)
                target_tol = max(0.1 * o.tol, o.inner_tol * o.inner_decay**outer)
                u, inner_ev, inner_jac = r.x.copy(), ev, jac
                step, solved = o.step_size, False
                for inner in range(o.max_inner + 1):
                    value, grad, _, _ = phr_terms(inner_ev, inner_jac, lam_bar, nu_bar, rho)
                    J = inner_jac.dF[0] + grad
                    gx = r.g(u)
                    model = r.subproblem(u, J, -gx)
                    mapping = infnorm(model.x - u) + np.sqrt(2 * model.gap)
                    if mapping <= target_tol:
                        solved = True
                        break
                    if inner == o.max_inner:
                        break
                    phi = inner_ev.F[0] + value
                    accepted = False
                    for _ in range(o.max_line_search):
                        if step < o.min_step:
                            break
                        proposal = r.subproblem(u, J, -gx, step)
                        trial_ev = r.values(proposal.x)
                        trial_value = phr_terms(trial_ev, None, lam_bar, nu_bar, rho)[0]
                        delta = proposal.x - u
                        bound = phi + J @ delta + (delta @ delta) / (2 * step)
                        slack = 32 * np.finfo(float).eps * np.maximum(1, np.abs(phi))
                        if np.all(trial_ev.F[0] + trial_value <= bound + slack):
                            accepted = True
                            break
                        step *= o.contraction
                    if not accepted or np.array_equal(u, proposal.x):
                        break
                    u = proposal.x
                    inner_ev, inner_jac = r.joint(u)
                    inner_steps += 1
                if not solved:
                    r.stop(
                        "inner_iterations_limit", "ALM inner stationarity tolerance was not met."
                    )
                    break
                _, _, lam_next, nu_next = phr_terms(inner_ev, inner_jac, lam_bar, nu_bar, rho)
                violation = max(
                    infnorm(inner_ev.H), infnorm(np.maximum(inner_ev.G[0], -lam_bar / rho))
                )
                next_rho = alm_penalty_update(
                    rho,
                    violation,
                    previous,
                    o.feasibility_contraction,
                    o.penalty_growth,
                    o.penalty_max,
                )
                r.accept(
                    u,
                    inner_ev,
                    step,
                    inner_tolerance=target_tol,
                    inner_residual=mapping,
                    penalty_used=rho,
                    penalty_next=next_rho,
                )
                inequality, equality = lam_next, nu_next
                ev, jac = inner_ev, inner_jac
                if (
                    violation > o.feasibility_contraction * previous
                    and rho == next_rho == o.penalty_max
                ):
                    if check(ev, jac, inequality, equality):
                        r.stop("stationary", "Pareto KKT residuals passed at the penalty cap.")
                    else:
                        r.stop("penalty_limit", "Penalty cap reached without residual reduction.")
                    break
                rho, previous = next_rho, violation
        except RUN_ERRORS as error:
            r.failed(error)
        return r.result(inner_iterations=inner_steps, final_penalty=rho)
