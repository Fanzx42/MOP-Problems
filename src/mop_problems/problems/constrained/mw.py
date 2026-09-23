"""MW1-14, Ma and Wang (2019), parameterization checked against pymoo 0.6.2."""

import numpy as np

from ...core import Problem, integer
from .._utils import linear_shape
from ..unconstrained_smooth.dtlz import _spherical


class _MW(Problem):
    number: int

    def __init__(self, n_var=None, n_obj=None):
        i = self.number
        m = (3 if i in (4, 8, 14) else 2) if n_obj is None else integer(n_obj, "n_obj", 2)
        if i not in (4, 8, 14) and m != 2:
            raise ValueError("Only MW4, MW8 and MW14 have scalable objectives")
        n = max(15, m + 1) if n_var is None else integer(n_var, "n_var", m + 1)
        upper = 1.1 if i == 6 else np.sqrt(2) if i == 11 else 1.5 if i in (13, 14) else 1.0
        count = (1, 1, 2, 1, 3, 1, 2, 1, 1, 3, 4, 2, 2, 1)[i - 1]
        super().__init__(f"MW{i}", n, m, 0, upper, count)

    def _evaluate(self, X):
        i, n, m = self.number, self.n_var, self.n_obj
        j = np.arange(m - 1, n)
        tail = X[:, m - 1 :]
        if i in (1, 4, 5, 9, 12):
            z = tail ** (n - m) - 0.5 - j / (2 * n)
            g = 1 + np.sum(-np.expm1(-10 * z * z), axis=1)
        elif i in (2, 6, 8, 10, 13):
            z = -np.expm1(-10 * (tail - j / n) ** 2)
            g = 1 + np.sum(0.1 / n * z * z + 1.5 - 1.5 * np.cos(2 * np.pi * z), axis=1)
        else:
            g = 1 + 2 * np.sum((tail + (X[:, m - 2 : -1] - 0.5) ** 2 - 1) ** 2, axis=1)
        x = X[:, 0]
        if i == 4:
            F = g[:, None] * linear_shape(1 - X[:, : m - 1])
            t = F[:, -1] - np.sum(F[:, :-1], axis=1)
            G = [np.sum(F, axis=1) - 1 - 0.4 * np.sin(2.5 * np.pi * t) ** 8]
        elif i == 8:
            F = g[:, None] * _spherical(X[:, : m - 1] * np.pi / 2)
            # arcsin(F_M/||F||) = pi*x_1/2 throughout the decision box.
            G = [np.sum(F * F, axis=1) - (1.25 - 0.5 * np.sin(3 * np.pi * x) ** 2) ** 2]
        elif i == 14:
            pos = X[:, : m - 1]
            wave = 1.5 * np.sin(1.1 * np.pi * pos**2)
            last = g * np.mean(6 - np.exp(pos) - wave, axis=1)
            F = np.column_stack((pos, last))
            G = [last - np.mean(5.1 - pos - 0.5 * pos**2 - wave, axis=1)]
        else:
            a = x if i <= 3 else g * x**n if i == 10 else g * x
            if i <= 3:
                b = g - (0.85 if i == 1 else 1) * a
            elif i in (5, 6, 7, 11):
                radius = 1.21 if i == 6 else 2 if i == 11 else 1
                b = g * np.sqrt(np.maximum(0, radius - x * x))
            elif i == 9:
                b = g * (1 - x**0.6)
            elif i == 10:
                b = g * (1 - x ** (2 * n))
            elif i == 12:
                b = g * (0.85 - 0.8 * x - 0.08 * np.abs(np.sin(3.2 * np.pi * x)))
            else:
                b = g * (5 - np.exp(x) - 0.5 * np.abs(np.sin(3 * np.pi * x)))
            F = np.column_stack((a, b))
            t = np.sqrt(2) * (b - a)
            r = a * a + b * b
            if i in (1, 2):
                G = [a + b - 1 - 0.5 * np.sin((2 if i == 1 else 3) * np.pi * t) ** 8]
            elif i == 3:
                G = [
                    a + b - 1.05 - 0.45 * np.sin(0.75 * np.pi * t) ** 6,
                    0.85 - a - b + 0.3 * np.sin(0.75 * np.pi * t) ** 2,
                ]
            elif i == 5:
                angle = np.arctan2(b, a)
                theta = 0.5 * np.pi - 2 * np.abs(angle - 0.25 * np.pi)
                G = [
                    r - (1.7 - 0.2 * np.sin(2 * angle)) ** 2,
                    (1 + 0.5 * np.sin(6 * theta**3)) ** 2 - r,
                    (1 - 0.45 * np.sin(6 * theta**3)) ** 2 - r,
                ]
            elif i == 6:
                wave = np.cos(6 * np.arctan2(b, a) ** 4) ** 10
                G = [(a / (1 + 0.15 * wave)) ** 2 + (b / (1 + 0.75 * wave)) ** 2 - 1]
            elif i == 7:
                wave = np.sin(4 * np.arctan2(b, a))
                G = [r - (1.2 + 0.4 * wave**16) ** 2, (1.15 - 0.2 * wave**8) ** 2 - r]
            elif i == 9:
                G = [
                    np.minimum(
                        (1 - 0.64 * a * a - b) * (1 - 0.36 * a * a - b),
                        (1.35**2 - (a + 0.35) ** 2 - b) * (1.15**2 - (a + 0.15) ** 2 - b),
                    )
                ]
            elif i == 10:
                G = [
                    -(2 - 4 * a * a - b) * (2 - 8 * a * a - b),
                    (2 - 2 * a * a - b) * (2 - 16 * a * a - b),
                    (1 - a * a - b) * (1.2 - 1.2 * a * a - b),
                ]
            elif i == 11:
                G = [
                    -(3 - a * a - b) * (3 - 2 * a * a - b),
                    (3 - 0.625 * a * a - b) * (3 - 7 * a * a - b),
                    -(1.62 - 0.18 * a * a - b) * (1.125 - 0.125 * a * a - b),
                    (2.07 - 0.23 * a * a - b) * (0.63 - 0.07 * a * a - b),
                ]
            elif i == 12:
                G = [
                    -(1 - 0.625 * a - b + 0.08 * np.sin(2 * np.pi * (b - a / 1.6)))
                    * (1.4 - 0.875 * a - b + 0.08 * np.sin(2 * np.pi * (b / 1.4 - a / 1.6))),
                    (1 - 0.8 * a - b + 0.08 * np.sin(2 * np.pi * (b - a / 1.5)))
                    * (1.8 - 1.125 * a - b + 0.08 * np.sin(2 * np.pi * (b / 1.8 - a / 1.6))),
                ]
            else:
                wave = 0.5 * np.sin(3 * np.pi * a)
                G = [
                    -(4 - a - 0.5 * a * a - wave - b) * (4 - 0.7 * a - wave - b),
                    (5 - np.exp(a) - wave - b) * (4 - 0.4 * a - wave - b),
                ]
        return F, np.column_stack(G), None


class MW1(_MW):
    number = 1


class MW2(_MW):
    number = 2


class MW3(_MW):
    number = 3


class MW4(_MW):
    number = 4


class MW5(_MW):
    number = 5


class MW6(_MW):
    number = 6


class MW7(_MW):
    number = 7


class MW8(_MW):
    number = 8


class MW9(_MW):
    number = 9


class MW10(_MW):
    number = 10


class MW11(_MW):
    number = 11


class MW12(_MW):
    number = 12


class MW13(_MW):
    number = 13


class MW14(_MW):
    number = 14
