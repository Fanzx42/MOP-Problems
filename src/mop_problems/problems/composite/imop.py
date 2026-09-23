"""IMOP1-8: Tian, Cheng, Zhang, Li and Jin (2019)."""

import numpy as np

from ...core import Problem, integer
from .._utils import mark_boundary, positive
from ..unconstrained_smooth.dtlz import _spherical


class _IMOP(Problem):
    number: int

    def __init__(self, n_var=10, k=5, a1=0.05, a2=10.0):
        self.k = integer(k, "k", 2)
        self.a1, self.a2 = positive(a1, "a1"), positive(a2, "a2")
        n = integer(n_var, "n_var", self.k + 1)
        super().__init__(f"IMOP{self.number}", n, 2 if self.number <= 3 else 3, 0, 1)

    def _evaluate(self, X):
        i = self.number
        y = np.mean(X[:, : self.k] if i <= 4 else X[:, : self.k : 2], axis=1) ** self.a1
        g = np.sum((X[:, self.k :] - 0.5) ** 2, axis=1)
        if i <= 2:
            power = 8 if i == 1 else 0.5
            F = (
                np.column_stack((np.cos(np.pi * y / 2) ** power, np.sin(np.pi * y / 2) ** power))
                + g[:, None]
            )
        elif i == 3:
            F = np.column_stack((1 + np.cos(10 * np.pi * y) / 5 - y, y)) + g[:, None]
        elif i == 4:
            F = (1 + g[:, None]) * np.column_stack((y, y + np.sin(10 * np.pi * y) / 10, 1 - y))
        else:
            z = np.mean(X[:, 1 : self.k : 2], axis=1) ** self.a2
            if i == 5:
                theta = np.pi * np.ceil(8 * y) / 4
                f1 = 0.4 * np.cos(theta) + 0.1 * z * np.cos(16 * np.pi * y)
                f2 = 0.4 * np.sin(theta) + 0.1 * z * np.sin(16 * np.pi * y)
                F = np.column_stack((f1, f2, 0.5 - f1 - f2)) + g[:, None]
            elif i == 6:
                # ceil(max(0,r)) is a step at r=0; outside that switch it is constant.
                r = np.minimum(np.sin(3 * np.pi * y) ** 2, np.sin(3 * np.pi * z) ** 2) - 0.05
                jump = np.where(r > 0, 1.0, 0.0)
                F = (
                    np.column_stack(((1 + g) * y, (1 + g) * z, (0.5 + g) * (2 - y - z)))
                    + jump[:, None]
                )
                F = mark_boundary(F, r)
            elif i == 7:
                F = (1 + g[:, None]) * _spherical(np.column_stack((y, z)) * np.pi / 2)
                r = np.minimum(
                    np.minimum(np.abs(F[:, 0] - F[:, 1]), np.abs(F[:, 1] - F[:, 2])),
                    np.abs(F[:, 2] - F[:, 0]),
                )
                F = F + 10 * np.maximum(0, r - 0.1)[:, None]
            else:
                f3 = (
                    3 * (1 + g)
                    - y * (1 + np.sin(19 * np.pi * y))
                    - z * (1 + np.sin(19 * np.pi * z))
                )
                F = np.column_stack((y, z, f3))
        return F, None, None


class IMOP1(_IMOP):
    number = 1


class IMOP2(_IMOP):
    number = 2


class IMOP3(_IMOP):
    number = 3


class IMOP4(_IMOP):
    number = 4


class IMOP5(_IMOP):
    number = 5


class IMOP6(_IMOP):
    number = 6


class IMOP7(_IMOP):
    number = 7


class IMOP8(_IMOP):
    number = 8
