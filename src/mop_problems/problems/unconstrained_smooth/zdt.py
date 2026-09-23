"""Continuous ZDT problems (Zitzler, Deb and Thiele, 2000).

ZDT5 uses binary variables and is intentionally outside this collection.
"""

import numpy as np

from mop_problems.core import FloatArray, Problem, integer


class _ZDT(Problem):
    number: int

    def __init__(self, n_var: int | None = None):
        if n_var is None:
            n_var = 10 if self.number in (4, 6) else 30
        n_var = integer(n_var, "n_var", 2)
        lower = np.zeros(n_var)
        upper = np.ones(n_var)
        if self.number == 4:
            lower[1:], upper[1:] = -5.0, 5.0
        super().__init__(f"ZDT{self.number}", n_var, 2, lower, upper)

    def _evaluate(self, X):
        f1 = X[:, 0]
        tail = X[:, 1:]
        if self.number == 4:
            g = 1 + 10 * (self.n_var - 1) + np.sum(tail**2 - 10 * np.cos(4 * np.pi * tail), axis=1)
        elif self.number == 6:
            f1 = 1 - np.exp(-4 * f1) * np.sin(6 * np.pi * f1) ** 6
            g = 1 + 9 * np.mean(tail, axis=1) ** 0.25
        else:
            g = 1 + 9 * np.mean(tail, axis=1)
        ratio = f1 / g
        if self.number in (2, 6):
            h = 1 - ratio**2
        else:
            h = 1 - np.sqrt(ratio)
            if self.number == 3:
                h -= ratio * np.sin(10 * np.pi * f1)
        return np.column_stack((f1, g * h)), None, None

    def pareto_front(self, n_points: int = 100, *, seed: int = 0) -> FloatArray:
        """Sample the front; ZDT3 endpoints are numerical to about 1e-8."""
        n_points = integer(n_points, "n_points", 2)
        if self.number == 3:
            intervals = (
                (0.0, 0.0830015349),
                (0.1822287280, 0.2577623634),
                (0.4093136748, 0.4538821041),
                (0.6183967944, 0.6525117038),
                (0.8233317983, 0.8518328654),
            )
            counts = [n_points // 5 + (i < n_points % 5) for i in range(5)]
            # Exclude each later segment's open lower endpoint.
            f1 = np.concatenate(
                [
                    np.linspace(lo, hi, count) if i == 0 else np.linspace(lo, hi, count + 1)[1:]
                    for i, ((lo, hi), count) in enumerate(zip(intervals, counts, strict=True))
                ]
            )
        elif self.number == 6:
            x_min = np.arctan(9 * np.pi) / (6 * np.pi)
            f_min = 1 - np.exp(-4 * x_min) * np.sin(6 * np.pi * x_min) ** 6
            f1 = np.linspace(f_min, 1, n_points)
        else:
            f1 = np.linspace(0, 1, n_points)
        if self.number in (2, 6):
            f2 = 1 - f1**2
        else:
            f2 = 1 - np.sqrt(f1)
            if self.number == 3:
                f2 -= f1 * np.sin(10 * np.pi * f1)
        return np.column_stack((f1, f2))


class ZDT1(_ZDT):
    """Convex Pareto front, 30 variables by default."""

    number = 1


class ZDT2(_ZDT):
    """Concave Pareto front, 30 variables by default."""

    number = 2


class ZDT3(_ZDT):
    """Five disconnected Pareto-front segments."""

    number = 3


class ZDT4(_ZDT):
    """Multimodal distance function, mixed decision bounds."""

    number = 4


class ZDT6(_ZDT):
    """Biased objective distribution and a nonuniform decision-to-front map."""

    number = 6
