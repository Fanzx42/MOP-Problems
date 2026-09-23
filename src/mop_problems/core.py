"""Common numerical contract for continuous minimization problems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def integer(value: int, name: str, minimum: int = 1) -> int:
    """Validate dimensions without silently truncating floats or accepting bools."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _readonly(value: ArrayLike) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class Evaluation:
    """Batch results; even a single decision vector retains its batch dimension.

    ``G <= 0`` and ``H == 0`` are feasible constraints. ``CV`` also includes
    violations of decision bounds, and applies the requested equality tolerance.
    Objectives and residuals are never normalized or replaced with penalties.
    """

    F: FloatArray
    G: FloatArray
    H: FloatArray
    CV: FloatArray

    @property
    def feasible(self) -> NDArray[np.bool_]:
        return self.CV == 0.0


class Problem(ABC):
    """Base class for vectorized, real-valued, all-minimization benchmarks."""

    def __init__(
        self,
        name: str,
        n_var: int,
        n_obj: int,
        xl: ArrayLike,
        xu: ArrayLike,
        n_ieq_constr: int = 0,
        n_eq_constr: int = 0,
    ) -> None:
        self.name = name
        self.n_var = integer(n_var, "n_var")
        self.n_obj = integer(n_obj, "n_obj", 2)
        self.n_ieq_constr = integer(n_ieq_constr, "n_ieq_constr", 0)
        self.n_eq_constr = integer(n_eq_constr, "n_eq_constr", 0)
        try:
            self.xl = _readonly(np.broadcast_to(np.asarray(xl, dtype=float), (self.n_var,)))
            self.xu = _readonly(np.broadcast_to(np.asarray(xu, dtype=float), (self.n_var,)))
        except (TypeError, ValueError) as error:
            raise ValueError("bounds must be scalar or have shape (n_var,)") from error
        if not np.all(np.isfinite(self.xl)) or not np.all(np.isfinite(self.xu)):
            raise ValueError("bounds must be finite")
        if np.any(self.xl >= self.xu):
            raise ValueError("each lower bound must be smaller than its upper bound")

    def sample(self, n: int, seed: int | np.random.Generator | None = None) -> FloatArray:
        """Sample the decision box (not necessarily the feasible region)."""
        n = integer(n, "n", 0)
        rng = np.random.default_rng(seed)
        return rng.uniform(self.xl, self.xu, size=(n, self.n_var))

    def evaluate(
        self, X: ArrayLike, *, check_bounds: bool = True, equality_tol: float = 1e-6
    ) -> Evaluation:
        """Evaluate a vector or a batch, without clipping or modifying the input.

        Set ``check_bounds=False`` only for formulas defined outside the box.
        Bound violations still contribute to ``CV`` in that mode.
        """
        if not np.isscalar(equality_tol) or not np.isfinite(equality_tol) or equality_tol < 0:
            raise ValueError("equality_tol must be a finite nonnegative scalar")
        raw = np.asarray(X)
        if np.iscomplexobj(raw):
            raise TypeError("X must contain real values")
        x = np.asarray(raw, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.n_var:
            raise ValueError(f"X must have shape ({self.n_var},) or (N, {self.n_var})")
        if not np.all(np.isfinite(x)):
            raise ValueError("X must contain only finite values")
        below = np.maximum(self.xl - x, 0.0)
        above = np.maximum(x - self.xu, 0.0)
        bound_cv = np.sum(below + above, axis=1)
        if check_bounds and np.any(bound_cv > 0):
            raise ValueError(f"X is outside the decision bounds of {self.name}")
        F, G, H = self._evaluate(x)
        arrays = []
        for label, value, width in (
            ("F", F, self.n_obj),
            ("G", G, self.n_ieq_constr),
            ("H", H, self.n_eq_constr),
        ):
            if value is None:
                if width:
                    raise RuntimeError(f"{self.name} did not return {label}")
                value = np.empty((len(x), 0))
            array = _readonly(value)
            if array.shape != (len(x), width):
                raise RuntimeError(f"{self.name}: {label} must have shape {(len(x), width)}")
            if not np.all(np.isfinite(array)):
                raise FloatingPointError(f"{self.name} produced non-finite {label}")
            arrays.append(array)
        f, g, h = arrays
        cv = bound_cv + np.maximum(g, 0).sum(axis=1)
        cv += np.maximum(np.abs(h) - equality_tol, 0).sum(axis=1)
        return Evaluation(f, g, h, _readonly(cv))

    @abstractmethod
    def _evaluate(self, X: FloatArray) -> tuple[FloatArray, FloatArray | None, FloatArray | None]:
        """Return objectives and raw residual matrices for an already validated batch."""

    def pareto_front(self, n_points: int = 100, *, seed: int = 0) -> FloatArray:
        """Return reference points only when a supported analytical front is known."""
        raise NotImplementedError(f"No analytical Pareto-front sampler for {self.name}")

    def __repr__(self) -> str:
        return (
            f"{self.name}(n_var={self.n_var}, n_obj={self.n_obj}, "
            f"n_ieq_constr={self.n_ieq_constr}, n_eq_constr={self.n_eq_constr})"
        )
