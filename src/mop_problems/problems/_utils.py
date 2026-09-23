"""Shared mathematical shapes and explicit piecewise-boundary bookkeeping."""

import numpy as np

from .._autodiff import Jet


def positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite positive scalar")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return float(value)


def mark_boundary(value, *residuals):
    """Mark evaluated branches at specified switch surfaces; keep ordinary arrays intact."""
    if isinstance(value, Jet):
        regular = value.r.copy()
        for residual in residuals:
            raw = residual.v if isinstance(residual, Jet) else np.asarray(residual)
            mask = np.abs(raw) <= 32 * np.finfo(float).eps
            if mask.ndim < value.ndim:
                mask = mask[..., None]
            regular &= ~mask
        return Jet(value.v, value.d, regular)
    return value


def linear_shape(x):
    """M objective simplex shape, from M-1 position variables."""
    return np.column_stack(
        [
            np.prod(x[:, : x.shape[1] - i], axis=1) * (1 - x[:, x.shape[1] - i] if i else 1)
            for i in range(x.shape[1] + 1)
        ]
    )


def rastrigin(x):
    return np.sum(x * x - 10 * np.cos(2 * np.pi * x) + 10, axis=1)
