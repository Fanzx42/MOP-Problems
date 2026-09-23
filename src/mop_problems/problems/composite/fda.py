"""FDA snapshots with explicit benchmark time; Farina, Deb and Amato (2004).

FDA2 uses the corrected exponent in the pinned PlatEMO definition, explicitly
identified in the problem card. Evaluation never advances time implicitly.
"""

from copy import copy

import numpy as np

from ...core import Problem, integer
from ..unconstrained_smooth.dtlz import _spherical


def benchmark_time(step, frequency=10, severity=10):
    """t = floor(step/frequency)/severity; step is a generation/environment counter."""
    return (
        integer(step, "step", 0) // integer(frequency, "frequency") / integer(severity, "severity")
    )


def _time(value):
    if (
        isinstance(value, (bool, np.bool_))
        or not np.isscalar(value)
        or not np.isfinite(value)
        or value < 0
    ):
        raise ValueError("time must be a finite nonnegative scalar")
    return float(value)


class _FDA(Problem):
    number: int

    def __init__(self, n_var=None, n_obj=None, time=0.0):
        self._time = _time(time)
        m = (2 if self.number <= 3 else 3) if n_obj is None else integer(n_obj, "n_obj", 2)
        if self.number <= 3 and m != 2:
            raise ValueError("FDA1-3 have exactly two objectives")
        n = (
            (11 if self.number == 2 else 10 if self.number <= 3 else m + 9)
            if n_var is None
            else integer(n_var, "n_var", max(m, 3 if self.number == 2 else 2))
        )
        if self.number == 2 and n % 2 != 1:
            raise ValueError("FDA2 requires odd n_var to split the tail into equal groups")
        lo = np.zeros(n) if self.number >= 4 else np.r_[0.0, -np.ones(n - 1)]
        super().__init__(f"FDA{self.number}", n, m, lo, 1)

    @property
    def time(self):
        return self._time

    def at_time(self, time):
        result = copy(self)
        result._time = _time(time)
        return result

    def _evaluate(self, X):
        i = self.number
        s = np.sin(np.pi * self.time / 2)
        if i <= 3:
            a = X[:, 0]
            if i == 1:
                g = 1 + np.sum((X[:, 1:] - s) ** 2, axis=1)
                b = g - np.sqrt(a * g)
            elif i == 2:
                middle = (self.n_var + 1) // 2
                g = 1 + np.sum(X[:, 1:middle] ** 2, axis=1)
                h = 0.75 + 0.7 * s
                exponent = h + np.sum((X[:, middle:] - h) ** 2, axis=1)
                b = g * (1 - (a / g) ** exponent)
            else:
                a = a ** (10 ** (2 * s))
                g = 1 + abs(s) + np.sum((X[:, 1:] - abs(s)) ** 2, axis=1)
                b = g - np.sqrt(a * g)
            F = np.column_stack((a, b))
        else:
            g = np.sum((X[:, self.n_obj - 1 :] - abs(s)) ** 2, axis=1)
            pos = X[:, : self.n_obj - 1]
            if i == 5:
                g = g + abs(s)
                pos = pos ** (1 + 100 * s**4)
            F = (1 + g[:, None]) * _spherical(pos * np.pi / 2)
        return F, None, None


class FDA1(_FDA):
    number = 1


class FDA2(_FDA):
    number = 2


class FDA3(_FDA):
    number = 3


class FDA4(_FDA):
    number = 4


class FDA5(_FDA):
    number = 5
