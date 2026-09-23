"""CEC 2009 CF1-10 (Zhang et al., CES-487); raw G <= 0 residuals."""

import numpy as np

from ...core import Problem, integer
from .._utils import mark_boundary
from ..unconstrained_smooth.dtlz import _spherical


def _compressed(t):
    # Equivalent to -t/(1+exp(4*abs(t))), without exponential overflow.
    q = np.exp(-4 * np.abs(t))
    return -t * q / (1 + q)


class _CF(Problem):
    number: int

    def __init__(self, n_var=10):
        i = self.number
        n = integer(n_var, "n_var", 5 if i >= 8 else 4 if i in (6, 7) else 3)
        m = 3 if i >= 8 else 2
        lo = np.zeros(n)
        hi = np.ones(n)
        if i != 1:
            extent = 1 if i == 2 else 4 if i == 8 else 2
            lo[m - 1 :], hi[m - 1 :] = -extent, extent
        super().__init__(f"CF{i}", n, m, lo, hi, 2 if i in (6, 7) else 1)

    def _evaluate(self, X):
        i, n = self.number, self.n_var
        x = X[:, 0]
        j = np.arange(2, n + 1)
        angle = 6 * np.pi * x[:, None] + j * np.pi / n
        if i >= 8:
            j = np.arange(3, n + 1)
            y = X[:, 2:] - 2 * X[:, 1, None] * np.sin(2 * np.pi * x[:, None] + j * np.pi / n)
            h = 4 * y * y - np.cos(8 * np.pi * y) + 1 if i == 10 else y * y
            F = _spherical(X[:, :2] * np.pi / 2) + 2 * np.column_stack(
                [np.mean(h[:, j % 3 == r], axis=1) for r in (1, 2, 0)]
            )
            a, b, c = F[:, 0], F[:, 1], F[:, 2]
            denominator = 1 - c * c
            oscillation = np.sin(2 * np.pi * ((a * a - b * b) / denominator + 1))
            wave = 4 * np.abs(oscillation) if i == 8 else (3 if i == 9 else 1) * oscillation
            G = (1 - (a * a + b * b) / denominator + wave)[:, None]
            return F, G, None
        if i == 1:
            y = X[:, 1:] - x[:, None] ** (0.5 * (1 + 3 * (j - 2) / (n - 2)))
        elif i == 2:
            y = X[:, 1:] - np.where(j % 2 == 1, np.sin(angle), np.cos(angle))
        elif i in (3, 4):
            y = X[:, 1:] - np.sin(angle)
        else:
            y = X[:, 1:] - (1 if i == 7 else 0.8 * x[:, None]) * np.where(
                j % 2 == 1, np.cos(angle), np.sin(angle)
            )
        odd, even = j % 2 == 1, j % 2 == 0
        if i <= 2:
            a = x + 2 * np.mean(y[:, odd] ** 2, axis=1)
            b = 1 - (x if i == 1 else np.sqrt(x)) + 2 * np.mean(y[:, even] ** 2, axis=1)
        elif i == 3:

            def penalty(mask):
                z = y[:, mask]
                return (
                    2
                    / z.shape[1]
                    * (
                        4 * np.sum(z * z, axis=1)
                        - 2 * np.prod(np.cos(20 * np.pi * z / np.sqrt(j[mask])), axis=1)
                        + 2
                    )
                )

            a, b = x + penalty(odd), 1 - x * x + penalty(even)
        else:
            h = 2 * y * y - np.cos(4 * np.pi * y) + 1 if i in (5, 7) else y * y
            if i in (4, 5):
                threshold = 1.5 * (1 - np.sqrt(0.5))
                first = np.where(y[:, 0] < threshold, np.abs(y[:, 0]), 0.125 + (y[:, 0] - 1) ** 2)
                h = np.column_stack((mark_boundary(first, y[:, 0] - threshold), h[:, 1:]))
            elif i == 7:
                h[:, [0, 2]] = y[:, [0, 2]] ** 2
            a = x + np.sum(h[:, odd], axis=1)
            b = ((1 - x) ** 2 if i >= 6 else 1 - x) + np.sum(h[:, even], axis=1)
        if i == 1:
            G = (1 - a - b + np.abs(np.sin(10 * np.pi * (a - b + 1))))[:, None]
        elif i == 2:
            G = _compressed(b + np.sqrt(a) - np.sin(2 * np.pi * (np.sqrt(a) - b + 1)) - 1)[:, None]
        elif i == 3:
            G = (1 - b - a * a + np.sin(2 * np.pi * (a * a - b + 1)))[:, None]
        elif i in (4, 5):
            t = y[:, 0] - 0.5 * x + 0.25
            G = (_compressed(t) if i == 4 else -t)[:, None]
        else:
            u = 0.5 * (1 - x) - (1 - x) ** 2
            v = 0.25 * np.sqrt(1 - x) - 0.5 * (1 - x)
            G = np.column_stack(
                (
                    -y[:, 0] + np.sign(u) * np.sqrt(np.abs(u)),
                    -y[:, 2] + np.sign(v) * np.sqrt(np.abs(v)),
                )
            )
        return np.column_stack((a, b)), G, None


class CF1(_CF):
    number = 1


class CF2(_CF):
    number = 2


class CF3(_CF):
    number = 3


class CF4(_CF):
    number = 4


class CF5(_CF):
    number = 5


class CF6(_CF):
    number = 6


class CF7(_CF):
    number = 7


class CF8(_CF):
    number = 8


class CF9(_CF):
    number = 9


class CF10(_CF):
    number = 10
