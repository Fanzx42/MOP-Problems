"""DTLZ1--7, using n_var = n_obj + k - 1 and unit decision bounds."""

import numpy as np

from mop_problems.core import FloatArray, Problem, integer


def _spherical(angles: FloatArray) -> FloatArray:
    """Map M-1 angles to the nonnegative part of an M-dimensional unit sphere."""
    n, m_minus_one = angles.shape
    values = np.empty((n, m_minus_one + 1))
    for i in range(m_minus_one + 1):
        prefix = m_minus_one - i
        values[:, i] = np.prod(np.cos(angles[:, :prefix]), axis=1)
        if i:
            values[:, i] *= np.sin(angles[:, prefix])
    return values


class _DTLZ(Problem):
    number: int

    def __init__(self, n_var: int | None = None, n_obj: int = 3):
        n_obj = integer(n_obj, "n_obj", 2)
        default_k = 5 if self.number == 1 else 20 if self.number == 7 else 10
        if n_var is None:
            n_var = n_obj + default_k - 1
        n_var = integer(n_var, "n_var", n_obj)
        super().__init__(f"DTLZ{self.number}", n_var, n_obj, 0.0, 1.0)

    def _evaluate(self, X):
        positions = X[:, : self.n_obj - 1]
        tail = X[:, self.n_obj - 1 :]
        if self.number in (1, 3):
            shifted = tail - 0.5
            g = 100 * (tail.shape[1] + np.sum(shifted**2 - np.cos(20 * np.pi * shifted), axis=1))
        elif self.number == 6:
            g = np.sum(tail**0.1, axis=1)
        elif self.number == 7:
            g = 1 + 9 * np.mean(tail, axis=1)
        else:
            g = np.sum((tail - 0.5) ** 2, axis=1)
        if self.number == 1:
            F = np.empty((len(X), self.n_obj))
            for i in range(self.n_obj):
                prefix = self.n_obj - i - 1
                F[:, i] = 0.5 * (1 + g) * np.prod(positions[:, :prefix], axis=1)
                if i:
                    F[:, i] *= 1 - positions[:, prefix]
        elif self.number == 7:
            final = (1 + g) * self.n_obj - np.sum(
                positions * (1 + np.sin(3 * np.pi * positions)), axis=1
            )
            F = np.column_stack((positions, final))
        else:
            angles = positions * (np.pi / 2)
            if self.number == 4:
                angles = positions**self.alpha * (np.pi / 2)
            elif self.number in (5, 6):
                angles[:, 1:] = (
                    np.pi * (1 + 2 * g[:, None] * positions[:, 1:]) / (4 * (1 + g[:, None]))
                )
            F = (1 + g[:, None]) * _spherical(angles)
        return F, None, None

    def pareto_front(self, n_points: int = 100, *, seed: int = 0) -> FloatArray:
        """Reference samples; high-dimensional sampling is seeded, not a uniform grid.

        DTLZ7's disconnected front is intentionally left unsupported instead of
        returning points from the dominated part of its g-minimal surface.
        """
        n_points = integer(n_points, "n_points", 2)
        if self.number == 7 or (self.number in (5, 6) and self.n_obj > 3):
            return super().pareto_front(n_points, seed=seed)
        if self.number in (5, 6):
            angles = np.full((n_points, self.n_obj - 1), np.pi / 4)
            angles[:, 0] = np.linspace(0, np.pi / 2, n_points)
            return _spherical(angles)
        if self.n_obj == 2:
            u = np.linspace(0, 1, n_points)
            if self.number == 1:
                return 0.5 * np.column_stack((u, 1 - u))
            return _spherical((u * np.pi / 2)[:, None])
        rng = np.random.default_rng(seed)
        if self.number == 1:
            F = rng.exponential(size=(n_points, self.n_obj))
            F = 0.5 * F / F.sum(axis=1, keepdims=True)
        else:
            F = np.abs(rng.normal(size=(n_points, self.n_obj)))
            F /= np.linalg.norm(F, axis=1, keepdims=True)
        count = min(n_points, self.n_obj)
        F[:count] = np.eye(self.n_obj)[:count] * (0.5 if self.number == 1 else 1.0)
        return F


class DTLZ1(_DTLZ):
    """Linear Pareto front and multimodal distance function; default k=5."""

    number = 1


class DTLZ2(_DTLZ):
    """Spherical Pareto front; default k=10."""

    number = 2


class DTLZ3(_DTLZ):
    """Spherical front with a multimodal distance function; default k=10."""

    number = 3


class DTLZ4(_DTLZ):
    """Biased spherical front; alpha=100 and k=10 by default."""

    number = 4

    def __init__(self, n_var: int | None = None, n_obj: int = 3, alpha: float = 100.0):
        if not np.isscalar(alpha) or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a finite positive scalar")
        self.alpha = float(alpha)
        super().__init__(n_var, n_obj)


class DTLZ5(_DTLZ):
    """Degenerate spherical front; default k=10."""

    number = 5


class DTLZ6(_DTLZ):
    """Degenerate front with fractional-power distance; default k=10."""

    number = 6


class DTLZ7(_DTLZ):
    """Disconnected front; default k=20."""

    number = 7
