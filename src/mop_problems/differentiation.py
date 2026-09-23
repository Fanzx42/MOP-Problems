"""Derivative metadata and explicit, bound-aware numerical validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import Jacobian, _readonly


def derivative_info(problem):
    name = problem.name
    smoothness = "smooth"
    notes = "Smooth formula on the decision box; this does not imply convexity."
    if name in ("ZDT1", "ZDT3", "ZDT4", "UF1", "UF2", "UF3", "UF7"):
        smoothness = "interior"
        notes = "Fractional powers/square roots can have singular derivatives at x[0]=0."
    elif name in ("ZDT6", "DTLZ6"):
        smoothness = "interior"
        notes = "Fractional distance powers can have singular derivatives at zero tails."
    elif name.startswith("WFG"):
        smoothness = "piecewise"
        notes = "Bias, shift and reduction stages include abs/floor/min/max and fractional powers."
    elif name in ("UF4", "UF5", "UF6", "UF9", "C2-DTLZ2"):
        smoothness = "piecewise"
        notes = "Absolute values or min/max branches: no classical derivative at some ties."
    elif name.startswith("SYM-PART"):
        smoothness = "piecewise_discontinuous"
        notes = "Quadratics inside each tile; tile changes can be discontinuous."
    elif name == "TNK":
        smoothness = "interior"
        notes = (
            "Constraint angle is undefined at the origin; its assigned value is not differentiable."
        )
    alpha = getattr(problem, "alpha", getattr(getattr(problem, "_base", None), "alpha", None))
    if alpha is not None and alpha < 1:
        smoothness = "interior"
        notes = "alpha < 1 introduces singular derivatives at zero position variables."
    if not type(problem).__module__.startswith("mop_problems.problems."):
        smoothness, notes = "unknown", "User-defined problem; validate smoothness and derivatives."
    return {
        "method": "forward",
        "objectives": True,
        "inequalities": True,
        "equalities": True,
        "smoothness": smoothness,
        "objective_smoothness": "smooth" if name in ("TNK", "C2-DTLZ2") else smoothness,
        "constraint_smoothness": smoothness
        if problem.n_ieq_constr or problem.n_eq_constr
        else "none",
        "notes": notes,
        "hessian": False,
        "constraint_convention": "raw G <= 0, H = 0; box derivatives are not included",
    }


def finite_difference(problem, X, *, relative_step=None) -> Jacobian:
    """Three-point differences: central inside, second-order one-sided at bounds.

    This is an explicitly approximate oracle. It cannot diagnose differentiability.
    Costs (1 + 2*n_var) vector evaluations per point, including the shared centre.
    """
    step = np.cbrt(np.finfo(float).eps) if relative_step is None else relative_step
    if not np.isscalar(step) or not np.isfinite(step) or step <= 0:
        raise ValueError("relative_step must be a finite positive scalar")
    x = problem._validate_input(X)
    centre = problem.evaluate(x)
    jac = [np.empty((len(x), a.shape[1], problem.n_var)) for a in (centre.F, centre.G, centre.H)]
    for j in range(problem.n_var):
        h = np.minimum(step * np.maximum(1, np.abs(x[:, j])), (problem.xu[j] - problem.xl[j]) / 4)
        forward = x[:, j] - h < problem.xl[j]
        backward = x[:, j] + h > problem.xu[j]
        a, b = x.copy(), x.copy()
        a[:, j] += np.where(backward, -h, h)
        b[:, j] += np.where(forward, 2 * h, np.where(backward, -2 * h, -h))
        ea, eb = problem.evaluate(a), problem.evaluate(b)
        for target, v0, va, vb in zip(
            jac,
            (centre.F, centre.G, centre.H),
            (ea.F, ea.G, ea.H),
            (eb.F, eb.G, eb.H),
            strict=True,
        ):
            target[:, :, j] = np.where(
                (forward | backward)[:, None],
                (-3 * v0 + 4 * va - vb) * np.where(backward, -1, 1)[:, None],
                va - vb,
            ) / (2 * h[:, None])
    if any(not np.isfinite(a).all() for a in jac):
        raise FloatingPointError("Non-finite finite difference")
    return Jacobian(*map(_readonly, jac), method="finite_difference")


@dataclass(frozen=True)
class DerivativeCheck:
    passed: bool
    max_absolute_error: dict[str, float]
    max_scaled_error: dict[str, float]
    n_points: int
    finite_difference_evaluations: int


def check_derivatives(problem, X, *, rtol=1e-4, atol=1e-6, relative_step=None):
    """Compare dF/dG/dH to independent differences; passes when all scaled errors <= 1."""
    for name, value in (("rtol", rtol), ("atol", atol)):
        if not np.isscalar(value) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative scalar")
    if atol == 0 and rtol == 0:
        raise ValueError("at least one error tolerance must be positive")
    x = problem._validate_input(X)
    exact = problem.jacobian(x)
    numerical = finite_difference(problem, x, relative_step=relative_step)
    absolute, scaled = {}, {}
    for key in ("dF", "dG", "dH"):
        a, b = getattr(exact, key), getattr(numerical, key)
        error = np.abs(a - b)
        denominator = atol + rtol * np.maximum(np.abs(a), np.abs(b))
        ratio = np.divide(error, denominator, out=np.zeros_like(error), where=denominator > 0)
        absolute[key] = float(np.max(error, initial=0))
        scaled[key] = float(np.max(ratio, initial=0))
    return DerivativeCheck(
        max(scaled.values()) <= 1, absolute, scaled, len(x), len(x) * (1 + 2 * problem.n_var)
    )
