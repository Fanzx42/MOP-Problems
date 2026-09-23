"""NCTP1-18, Li, Wang, Yang and Cai (CEC 2016), original three-by-six numbering."""

import numpy as np

from ...core import Problem, integer
from .ctp import rotated_constraint


class _NCTP(Problem):
    number: int

    def __init__(self, n_var=30):
        n = integer(n_var, "n_var", 3)
        super().__init__(f"NCTP{self.number}", n, 2, 0, 5, 2 if (self.number - 1) % 6 < 3 else 1)

    def _evaluate(self, X):
        group, member = divmod(self.number - 1, 6)
        kind = member % 3
        x = X[:, 0]
        g = np.sum(100 * (X[:, 2:] - X[:, 1:-1] ** 2) ** 2 + (1 - X[:, 1:-1]) ** 2, axis=1)
        shift = -0.5 if group == 0 else 1.0 if group == 1 else (2.0, 2.5, 4.0)[kind]
        # Cancel g*(1-sqrt(x)/g) analytically, including its continuous g=0 limit.
        y = g - np.sqrt(x)
        F = np.column_stack((x, y + shift))
        parameters = (-0.2, (0.2, 0.75, 2.0)[kind], 10, 1, (0.5, 0.5, 6.0)[kind], 1)
        G = [rotated_constraint(F[:, 0], F[:, 1], parameters)]
        if member < 3:
            limit = ((4.5, 4.5, 6.5), (3.0, 3.0, 5.0), (2.0, 1.5, 2.0))[group][kind]
            G.append(y + 0.73 * x - limit)
        return F, np.column_stack(G), None


class NCTP1(_NCTP):
    number = 1


class NCTP2(_NCTP):
    number = 2


class NCTP3(_NCTP):
    number = 3


class NCTP4(_NCTP):
    number = 4


class NCTP5(_NCTP):
    number = 5


class NCTP6(_NCTP):
    number = 6


class NCTP7(_NCTP):
    number = 7


class NCTP8(_NCTP):
    number = 8


class NCTP9(_NCTP):
    number = 9


class NCTP10(_NCTP):
    number = 10


class NCTP11(_NCTP):
    number = 11


class NCTP12(_NCTP):
    number = 12


class NCTP13(_NCTP):
    number = 13


class NCTP14(_NCTP):
    number = 14


class NCTP15(_NCTP):
    number = 15


class NCTP16(_NCTP):
    number = 16


class NCTP17(_NCTP):
    number = 17


class NCTP18(_NCTP):
    number = 18
