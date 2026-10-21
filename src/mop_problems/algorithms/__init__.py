"""Reproducible multiobjective first-order baselines; see docs/algorithms.md."""

import numpy as np

from .accelerated import (
    AMGOptions,
    NesterovAcceleratedGradient,
    NesterovAcceleratedProximalGradient,
)
from .augmented import ALMOptions, AugmentedLagrangian
from .conditional import AwayStepFrankWolfe, FrankWolfe, FWOptions, PairwiseFrankWolfe, Polytope
from .conjugate import CGOptions, ConjugateGradient
from .first_order import (
    BBDMO,
    GD,
    MGDA,
    FirstOrderMethod,
    Iteration,
    OptimizeResult,
    ProximalGradient,
    SolverOptions,
    TFYAcceleratedGradient,
    TFYAcceleratedProximalGradient,
)
from .proximal import L1
from .splitting import ADMMOptions, MultiobjectiveADMM, SplitProblem

_METHODS = {
    "mgda": MGDA,
    "bbdmo": BBDMO,
    "proximal_gradient": ProximalGradient,
    "nesterov": NesterovAcceleratedGradient,
    "nesterov_proximal": NesterovAcceleratedProximalGradient,
    "conjugate_gradient": ConjugateGradient,
    "frank_wolfe": FrankWolfe,
    "away_frank_wolfe": AwayStepFrankWolfe,
    "pairwise_frank_wolfe": PairwiseFrankWolfe,
    "alm": AugmentedLagrangian,
    "admm": MultiobjectiveADMM,
    "tfy": TFYAcceleratedGradient,
    "tfy_proximal": TFYAcceleratedProximalGradient,
}
_ALIASES = {
    "gd": "mgda",
    "steepest_descent": "mgda",
    "pg": "proximal_gradient",
    "nag": "nesterov",
    "amg_qp": "nesterov",
    "apg": "nesterov_proximal",
    "amg_proximal": "nesterov_proximal",
    "cg": "conjugate_gradient",
    "fw": "frank_wolfe",
    "away_fw": "away_frank_wolfe",
    "pairwise_fw": "pairwise_frank_wolfe",
}


def list_algorithms():
    return list(_METHODS)


def get_algorithm(name, **options):
    key = name.lower().replace("-", "_")
    key = _ALIASES.get(key, key)
    if key not in _METHODS:
        raise KeyError(f"Unknown algorithm {name!r}; choose {list_algorithms()}")
    return _METHODS[key](**options)


def minimize(problem, x0, *, algorithm="mgda", **options):
    """Convenience entry point; use a configured method's solve() to reuse settings."""
    return get_algorithm(algorithm, **options).solve(problem, x0)


def multistart(problem, algorithm, *, starts=None, n_starts=10, seed=42):
    """Independent runs sharing a configuration; explicit starts enable paired comparisons.

    Returns every result, including failures. Budgets apply PER start and must be
    summed when comparing the collected set to a population algorithm.
    """
    if isinstance(algorithm, str):
        algorithm = get_algorithm(algorithm)
    X = problem.sample(n_starts, seed=seed) if starts is None else problem._validate_input(starts)
    return [algorithm.solve(problem, row.copy()) for row in X]


def nondominated_mask(F):
    """Exact minimization dominance; duplicates retained, feasible filtering is caller's job."""
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[1] < 2 or not np.isfinite(F).all():
        raise ValueError("F must be a finite (N, n_obj>=2) array")
    return np.array(
        [not np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) for f in F], dtype=bool
    )


__all__ = [
    "ADMMOptions",
    "ALMOptions",
    "AMGOptions",
    "AugmentedLagrangian",
    "AwayStepFrankWolfe",
    "BBDMO",
    "CGOptions",
    "ConjugateGradient",
    "FWOptions",
    "FrankWolfe",
    "GD",
    "L1",
    "MGDA",
    "MultiobjectiveADMM",
    "FirstOrderMethod",
    "Iteration",
    "NesterovAcceleratedGradient",
    "NesterovAcceleratedProximalGradient",
    "OptimizeResult",
    "PairwiseFrankWolfe",
    "Polytope",
    "ProximalGradient",
    "SolverOptions",
    "SplitProblem",
    "TFYAcceleratedGradient",
    "TFYAcceleratedProximalGradient",
    "get_algorithm",
    "list_algorithms",
    "minimize",
    "multistart",
    "nondominated_mask",
]
