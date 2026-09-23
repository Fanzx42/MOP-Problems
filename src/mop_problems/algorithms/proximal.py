"""Separable convex terms for vector proximal-gradient subproblems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Zero:
    """Zero penalty; the proximal map still enforces the supplied decision box."""

    n_obj: int

    def values(self, x):
        return np.zeros(self.n_obj)

    def prox(self, v, step, weights, lower, upper):
        return np.clip(v, lower, upper)

    def dual_curvature(self, z, J, step, lower, upper):
        free = (z > lower) & (z < upper)
        return step * J[:, free] @ J[:, free].T


@dataclass(frozen=True)
class L1:
    """g_i(x) = sum_j coefficients[i,j] * abs(x[j] - centres[i,j]).

    Scalars broadcast to every objective/variable. A one-dimensional coefficient
    or centre array has one entry per objective; use (m,n) for coordinate weights.
    """

    coefficients: object = 1.0
    centres: object = 0.0

    def bind(self, n_obj, n_var):
        arrays = []
        for name, value in (("coefficients", self.coefficients), ("centres", self.centres)):
            raw = np.asarray(value)
            if np.iscomplexobj(raw):
                raise ValueError(f"{name} must be real")
            array = np.asarray(raw, dtype=float)
            if array.ndim == 1:
                array = array[:, None]
            try:
                array = np.broadcast_to(array, (n_obj, n_var)).copy()
            except ValueError as error:
                raise ValueError(f"{name} must broadcast to (n_obj, n_var)") from error
            if not np.isfinite(array).all() or (name == "coefficients" and np.any(array < 0)):
                raise ValueError("L1 coefficients must be nonnegative; all parameters finite")
            array.setflags(write=False)
            arrays.append(array)
        return _BoundL1(*arrays)


@dataclass(frozen=True)
class _BoundL1:
    coefficients: np.ndarray
    centres: np.ndarray

    def values(self, x):
        return np.sum(self.coefficients * np.abs(x - self.centres), axis=1)

    def dual_curvature(self, z, J, step, lower, upper):
        # On each smooth piece, dz/dlambda = -step*(J + grad g)^T.
        free = (z > lower) & (z < upper)
        free &= np.all(np.abs(z - self.centres) > 32 * np.finfo(float).eps, axis=0)
        A = J + self.coefficients * np.sign(z - self.centres)
        return step * A[:, free] @ A[:, free].T

    def prox(self, v, step, weights, lower, upper):
        # Exact coordinatewise prox of the WEIGHTED SUM of objective penalties.
        # Sorting breakpoints also handles different centres, duplicates and zeros.
        v, weights = np.asarray(v, dtype=float), np.asarray(weights, dtype=float)
        order = np.argsort(self.centres, axis=0, kind="stable")
        centres = np.take_along_axis(self.centres, order, axis=0)
        charges = np.take_along_axis(step * weights[:, None] * self.coefficients, order, axis=0)
        cumulative = np.cumsum(charges, axis=0)
        total = cumulative[-1]
        right_derivative = centres - v + 2 * cumulative - total
        crosses = right_derivative >= 0
        first = np.where(crosses.any(axis=0), crosses.argmax(axis=0), len(centres))
        previous = np.vstack((np.zeros_like(total), cumulative))
        root = v + total - 2 * previous[first, np.arange(len(v))]
        stops = np.vstack((centres, np.full_like(total, np.inf)))
        return np.clip(np.minimum(root, stops[first, np.arange(len(v))]), lower, upper)
