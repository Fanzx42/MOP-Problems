"""LIR-CMOP1-14, Fan et al. (2019); pinned PlatEMO formula variant."""

import numpy as np

from ...core import Problem, integer
from ..unconstrained_smooth.dtlz import _spherical


def _ellipse(a, b, p, q, major, minor):
    return (
        0.1
        - ((a - p + b - q) / np.sqrt(2) / major) ** 2
        - ((-a + p + b - q) / np.sqrt(2) / minor) ** 2
    )


class _LIRCMOP(Problem):
    number: int

    def __init__(self, n_var=30):
        count = 3 if self.number in (3, 4, 7, 8, 14) else 2
        super().__init__(
            f"LIRCMOP{self.number}",
            integer(n_var, "n_var", 3),
            3 if self.number >= 13 else 2,
            0,
            1,
            count,
        )

    def _evaluate(self, X):
        i = self.number
        x = X[:, 0]
        if i >= 13:
            g = 1.7057 + 10 * np.sum((X[:, 2:] - 0.5) ** 2, axis=1)
            F = g[:, None] * _spherical(X[:, :2] * np.pi / 2)
            r = g * g
            G = [(r - 9) * (4 - r), (r - 3.61) * (3.24 - r)]
            if i == 14:
                G.append((r - 3.0625) * (2.56 - r))
            return F, np.column_stack(G), None
        j = np.arange(2, self.n_var + 1)
        if i == 1:
            target = np.where(
                j % 2 == 1, np.sin(np.pi * x[:, None] / 2), np.cos(np.pi * x[:, None] / 2)
            )
        elif i <= 4:
            target = x[:, None]
        else:
            angle = 0.5 * np.pi * x[:, None] * j / self.n_var
            target = np.where(j % 2 == 1, np.sin(angle), np.cos(angle))
        z = (X[:, 1:] - target) ** 2
        g1, g2 = np.sum(z[:, j % 2 == 1], axis=1), np.sum(z[:, j % 2 == 0], axis=1)
        shape = 1 - x * x if i in (1, 3, 6, 8, 9, 12) else 1 - np.sqrt(x)
        if i <= 4:
            a, b = x + g1, shape + g2
            G = [(0.5 - g1) * (0.51 - g1), (0.5 - g2) * (0.51 - g2)]
            if i in (3, 4):
                G.append(0.5 - np.sin(20 * np.pi * x))
        else:
            a, b = (
                (x + 10 * g1 + 0.7057, shape + 10 * g2 + 0.7057)
                if i <= 8
                else (1.7057 * x * (1 + 10 * g1), 1.7057 * shape * (1 + 10 * g2))
            )
            if i == 5:
                params = [(1.6, 1.6, 2, 4), (2.5, 2.5, 2, 8)]
            elif i == 6:
                params = [(1.8, 1.8, 2, 8), (2.8, 2.8, 2, 8)]
            elif i <= 8:
                params = [(1.2, 1.2, 2, 6), (2.25, 2.25, 2.5, 12), (3.5, 3.5, 2.5, 10)]
            else:
                params = [
                    [(1.4, 1.4, 1.5, 6)],
                    [(1.1, 1.2, 2, 4)],
                    [(1.2, 1.2, 1.5, 5)],
                    [(1.6, 1.6, 1.5, 6)],
                ][i - 9]
            G = [_ellipse(a, b, *p) for p in params]
            if i >= 9:
                G.append(
                    [2.0, 1.0, 2.1, 2.5][i - 9]
                    - (a + b) / np.sqrt(2)
                    + np.sin(4 * np.pi * (a - b) / np.sqrt(2))
                )
        return np.column_stack((a, b)), np.column_stack(G), None


class LIRCMOP1(_LIRCMOP):
    number = 1


class LIRCMOP2(_LIRCMOP):
    number = 2


class LIRCMOP3(_LIRCMOP):
    number = 3


class LIRCMOP4(_LIRCMOP):
    number = 4


class LIRCMOP5(_LIRCMOP):
    number = 5


class LIRCMOP6(_LIRCMOP):
    number = 6


class LIRCMOP7(_LIRCMOP):
    number = 7


class LIRCMOP8(_LIRCMOP):
    number = 8


class LIRCMOP9(_LIRCMOP):
    number = 9


class LIRCMOP10(_LIRCMOP):
    number = 10


class LIRCMOP11(_LIRCMOP):
    number = 11


class LIRCMOP12(_LIRCMOP):
    number = 12


class LIRCMOP13(_LIRCMOP):
    number = 13


class LIRCMOP14(_LIRCMOP):
    number = 14
