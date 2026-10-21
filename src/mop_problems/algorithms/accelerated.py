"""AMG-QP recurrence and an explicitly defined composite proximal extension.

The smooth recurrence matches LNAG/Algorithms.py:AMG_QP. The composite extension
is derived in docs/unified-algorithms.zh-CN.md; it is not the TFY/FISTA algorithm.
"""

from dataclasses import dataclass

import numpy as np

from ._runtime import RUN_ERRORS, Run, options_with_L, positive
from ._subproblem import SubproblemError
from .first_order import SolverOptions
from .proximal import Zero


def gradient_hull(J, target, *, tol, max_iter):
    """Uniform-start pairwise FW for min_w .5||w J-target||² on the simplex."""
    weights = np.full(len(J), 1 / len(J))
    for k in range(max_iter + 1):
        mixture = weights @ J
        scores = J @ (mixture - target)
        toward = int(np.argmin(scores))
        gap = max(0.0, float(weights @ scores - scores[toward]))
        if gap <= tol:
            return mixture, weights, gap, k
        if k == max_iter:
            break
        active = np.flatnonzero(weights > 0)
        away = int(active[np.argmax(scores[active])])
        difference = J[toward] - J[away]
        norm2 = float(difference @ difference)
        if norm2 == 0:
            break
        alpha = min(weights[away], max(0.0, -(difference @ (mixture - target)) / norm2))
        weights[toward] += alpha
        weights[away] -= alpha
    raise SubproblemError(f"AMG-QP pairwise FW gap {gap:.3g} exceeds {tol:.3g}")


@dataclass(frozen=True)
class AMGOptions(SolverOptions):
    step_rule: str = "fixed"
    bounds: str = "none"
    domain_restart: bool = False
    subproblem_max_iter: int = 1000
    mu: float = 0.0
    gamma0: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.step_rule != "fixed":
            raise ValueError("AMG-QP uses a supplied fixed L; select tfy for TFY backtracking")
        positive(self.mu, "mu", zero=True)
        positive(self.gamma0, "gamma0")
        if self.mu > 1 / self.step_size:
            raise ValueError("require 0 <= mu <= L")


class NesterovAcceleratedGradient:
    """Full smooth AMG-QP recurrence, with a common fixed smoothness constant L."""

    name = "nesterov"
    proximal = False

    def __init__(self, *, regularizer=None, **options):
        if regularizer is not None and not self.proximal:
            raise ValueError("Use NesterovAcceleratedProximalGradient for a regularizer")
        self.regularizer = regularizer
        self.options = AMGOptions(**options_with_L(options))

    def solve(self, problem, x0):
        r = Run(problem, x0, self.options, self.name, self.regularizer)
        o = r.o
        z, gamma, L = r.x.copy(), o.gamma0, 1 / o.step_size
        zero = isinstance(r.penalty, Zero) or (
            hasattr(r.penalty, "coefficients") and not np.any(r.penalty.coefficients)
        )
        try:
            for k in range(o.max_iter + 1):
                _, jac = r.joint(r.x)
                r.stationarity(jac.dF[0])
                r.record()
                if r.bound <= o.tol:
                    r.stop("stationary", "Unit-step Pareto proximal residual below tolerance.")
                    break
                if k == o.max_iter:
                    break
                ratio = gamma / L
                tau = ratio / 2 + np.sqrt(ratio + ratio**2 / 4)
                y = (r.x + tau * z) / (1 + tau)
                if o.domain_restart and (np.any(y < problem.xl) or np.any(y > problem.xu)):
                    z, gamma = r.x.copy(), o.gamma0
                    ratio = gamma / L
                    tau = ratio / 2 + np.sqrt(ratio + ratio**2 / 4)
                    y = r.x.copy()
                    r.restarts += 1
                _, jy = r.joint(y)
                J = jy.dF[0]
                h = tau**2 / ((1 + tau) * (gamma + o.mu * tau))
                centre = y + h * o.mu * (y - z)
                if zero and o.bounds == "none":
                    target = o.mu * (y - r.x) + gamma * (z - r.x) / tau
                    r.counts["nsubproblem"] += 1
                    mixture, weights, gap, nit = gradient_hull(
                        J, target, tol=o.subproblem_tol, max_iter=o.subproblem_max_iter
                    )
                    r.counts["subproblem_iterations"] += nit
                    zn = (gamma * z + o.mu * tau * y - tau * mixture) / (gamma + o.mu * tau)
                    xn = (r.x + tau * zn) / (1 + tau)
                else:
                    # max_i[J_i(y)(u-x)+g_i(u)-g_i(x)] + ||u-centre||²/(2h)
                    offsets = J @ (centre - r.x) - r.g(r.x)
                    model = r.subproblem(centre, J, offsets, h)
                    xn, weights, gap = model.x, model.weights, model.gap
                    zn = ((1 + tau) * xn - r.x) / tau
                ev = r.values(xn)
                gamma = (gamma + o.mu * tau) / (1 + tau)
                z = zn
                r.accept(
                    xn,
                    ev,
                    h,
                    tau=tau,
                    gamma=gamma,
                    z=z,
                    y=y,
                    objective_weights=weights,
                    direction_gap=gap,
                    smoothness=L,
                )
        except RUN_ERRORS as error:
            r.failed(error)
        return r.result(recurrence="AMG-QP", composite_extension=not zero, z=z, gamma=gamma)


class NesterovAcceleratedProximalGradient(NesterovAcceleratedGradient):
    """Composite shifted-minimax proximal extension of the full AMG-QP clock."""

    name = "nesterov_proximal"
    proximal = True
