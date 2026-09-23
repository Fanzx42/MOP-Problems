"""Convex minimax subproblems, solved through their simplex dual with gap checks."""

from dataclasses import dataclass

import numpy as np


class SubproblemError(RuntimeError):
    """The numerical inner solve did not meet the requested primal-dual gap."""


@dataclass(frozen=True)
class SubproblemResult:
    x: np.ndarray
    weights: np.ndarray
    gap: float
    primal: float
    iterations: int
    prox_evaluations: int


def minimax_step(
    y, J, offsets, penalty, step, lower, upper, *, tol=1e-12, max_iter=500, on_prox=None
):
    """min_z max_i[J_i(z-y)+g_i(z)+offset_i] + ||z-y||^2/(2*step).

    The prox of sum_i lambda_i*g_i (including the box) defines the dual. Its
    envelope derivative is the vector of affine-plus-penalty model values.
    """
    m = len(J)
    calls = 0

    def oracle(w):
        nonlocal calls
        if on_prox is not None:
            on_prox()
        z = np.asarray(penalty.prox(y - step * (w @ J), step, w, lower, upper), dtype=float)
        calls += 1
        if z.shape != y.shape or not np.isfinite(z).all() or np.any(z < lower) or np.any(z > upper):
            raise SubproblemError("prox returned a non-finite, malformed or out-of-box point")
        g = np.asarray(penalty.values(z), dtype=float)
        if g.shape != (m,) or not np.isfinite(g).all():
            raise SubproblemError("penalty.values must return one finite value per objective")
        delta = z - y
        model = J @ delta + g + offsets
        quadratic = float(delta @ delta / (2 * step))
        dual = float(w @ model + quadratic)
        primal = float(np.max(model) + quadratic)
        gap = max(0.0, float(np.max(model) - w @ model))
        return z, model, dual, primal, gap

    def finish(w, data, iteration):
        return SubproblemResult(data[0], w, data[4], data[3], iteration, calls)

    def acceptable(data):
        return data[4] <= tol * max(1.0, abs(data[2]), abs(data[3]))

    if m == 2:
        # Concave differentiable dual on [0,1]; endpoint KKT or bracketed root.
        left, right = 0.0, 1.0
        for w in (np.array([0.0, 1.0]), np.array([1.0, 0.0])):
            data = oracle(w)
            if acceptable(data):
                return finish(w, data, 0)
        for iteration in range(1, max_iter + 1):
            middle = (left + right) / 2
            w = np.array([middle, 1 - middle])
            data = oracle(w)
            if acceptable(data):
                return finish(w, data, iteration)
            if data[1][0] - data[1][1] > 0:
                left = middle
            else:
                right = middle
        raise SubproblemError(f"Two-objective dual failed: gap={data[4]:.3g}")

    try:
        from scipy.optimize import minimize
    except ImportError as error:
        raise ImportError("Install mop-problems[algorithms] for 3+ objective solvers") from error
    scale = max(1.0, step * float(np.max(np.sum(J**2, axis=1))), float(np.max(np.abs(offsets))))

    def fun(w):
        data = oracle(w)
        return -data[2] / scale, -data[1] / scale

    result = minimize(
        fun,
        np.full(m, 1 / m),
        jac=True,
        method="SLSQP",
        bounds=[(0, 1)] * m,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(m)},
        options={"ftol": min(tol / scale, 1e-14), "maxiter": max_iter},
    )
    w = np.maximum(result.x, 0)
    if not np.isfinite(w).all() or w.sum() == 0:
        raise SubproblemError("Simplex solver returned invalid weights")
    w /= w.sum()
    data = oracle(w)
    if acceptable(data):
        return finish(w, data, int(result.nit))
    # Pairwise ascent polishing maintains the simplex and uses exact line-search
    # roots. Certification depends on the gap, never on SLSQP's success flag.
    for iteration in range(1, max_iter + 1):
        model = data[1]
        receiver = int(np.argmax(model))
        donor = int(np.argmin(np.where(w > 0, model, np.inf)))
        direction = np.zeros(m)
        direction[receiver] += 1
        direction[donor] -= 1
        left, right = 0.0, w[donor]
        if hasattr(penalty, "dual_curvature"):
            # A local Newton step accelerates narrow quadratic pieces of the
            # dual; a safeguarded line search and gap still certify the result.
            active = np.flatnonzero((w > 1e-14) | (np.arange(m) == receiver))
            H = penalty.dual_curvature(data[0], J, step, lower, upper)
            block = H[np.ix_(active, active)]
            block = block + 1e-14 * max(1, np.linalg.norm(block)) * np.eye(len(active))
            kkt = np.block(
                [[block, np.ones((len(active), 1))], [np.ones((1, len(active))), np.zeros((1, 1))]]
            )
            rhs = np.r_[model[active] - model[active].mean(), 0.0]
            candidate = np.zeros(m)
            candidate[active] = np.linalg.lstsq(kkt, rhs, rcond=1e-15)[0][:-1]
            # Enforce the equality to roundoff before computing a feasible step.
            candidate[active] -= candidate[active].mean()
            negative = candidate < 0
            limit = min(1.0, float(np.min(-w[negative] / candidate[negative], initial=np.inf)))
            if limit > 1e-12 and model @ candidate > 0:
                direction, right = candidate, limit
        trial = w + right * direction
        trial = np.maximum(trial, 0)
        trial /= trial.sum()
        trial_data = oracle(trial)
        if trial_data[1] @ direction < 0:
            for _ in range(50):
                alpha = (left + right) / 2
                trial = w + alpha * direction
                trial_data = oracle(trial)
                if trial_data[1] @ direction > 0:
                    left = alpha
                else:
                    right = alpha
        w, data = trial, trial_data
        if acceptable(data):
            return finish(w, data, int(result.nit) + iteration)
    raise SubproblemError(f"Simplex dual failed certification: gap={data[4]:.3g}")
