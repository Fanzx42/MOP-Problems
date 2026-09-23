"""Difficulty-adjustable DAS-CMOP1-9, Fan et al. (2020).

The feasibility constraint at zeta=0 is -g<=0 (no finite upper bound);
this avoids the arbitrary 1e30 sentinel used in some reference software.
"""

import numpy as np

from ...core import Problem, integer
from ..unconstrained_smooth.dtlz import _spherical

DIFFICULTIES = (
    (0.25, 0, 0),
    (0, 0.25, 0),
    (0, 0, 0.25),
    (0.25, 0.25, 0.25),
    (0.5, 0, 0),
    (0, 0.5, 0),
    (0, 0, 0.5),
    (0.5, 0.5, 0.5),
    (0.75, 0, 0),
    (0, 0.75, 0),
    (0, 0, 0.75),
    (0.75, 0.75, 0.75),
    (0, 1, 0),
    (0.5, 1, 0),
    (0, 1, 0.5),
    (0.5, 1, 0.5),
)


class _DASCMOP(Problem):
    number: int

    def __init__(self, n_var=30, difficulty=4):
        if isinstance(difficulty, (int, np.integer)) and not isinstance(
            difficulty, (bool, np.bool_)
        ):
            index = integer(difficulty, "difficulty")
            if index > len(DIFFICULTIES):
                raise ValueError("difficulty index must be in 1..16")
            difficulty = DIFFICULTIES[index - 1]
        triplet = np.asarray(difficulty, dtype=float)
        if (
            triplet.shape != (3,)
            or not np.isfinite(triplet).all()
            or np.any((triplet < 0) | (triplet > 1))
        ):
            raise ValueError("difficulty must be a 1..16 index or (eta,zeta,gamma) in [0,1]^3")
        self.difficulty = tuple(triplet)
        m = 2 if self.number <= 6 else 3
        super().__init__(
            f"DASCMOP{self.number}", integer(n_var, "n_var", m), m, 0, 1, 11 if m == 2 else 7
        )

    def _evaluate(self, X):
        i, m = self.number, self.n_obj
        x = X[:, 0]
        tail = X[:, m - 1 :]
        if i <= 3:
            g = np.sum((tail - np.sin(np.pi * x[:, None] / 2)) ** 2, axis=1)
        elif i <= 8:
            z = tail - 0.5
            g = np.sum(z * z + 1 - np.cos(20 * np.pi * z), axis=1)
        else:
            j = np.arange(m, self.n_var + 1)
            g = np.sum(
                (tail - np.cos(0.25 * np.pi * j / self.n_var * (X[:, 0, None] + X[:, 1, None])))
                ** 2,
                axis=1,
            )
        if m == 2:
            shape = 1 - x * x if i in (1, 4) else 1 - np.sqrt(x)
            if i in (3, 6):
                shape += 0.5 * np.abs(np.sin(5 * np.pi * x))
            F = np.column_stack((x + g, shape + g))
        elif i == 7:
            F = np.column_stack((x * X[:, 1], X[:, 1] * (1 - x), 1 - X[:, 1])) + g[:, None]
        else:
            F = _spherical(X[:, :2] * np.pi / 2) + g[:, None]
        eta, zeta, gamma = self.difficulty
        G = [2 * eta - 1 - np.sin(20 * np.pi * x)]
        if m == 3:
            G.append(2 * eta - 1 - np.cos(20 * np.pi * X[:, 1]))
        if zeta == 0:
            G.append(-g)
        elif zeta == 1:
            G.append(np.abs(0.5 - g) - 1e-4)
        else:
            G.append((g - (0.5 - np.log(zeta))) * (g - 0.5))
        if m == 2:
            for p, q in zip(
                (0, 1, 0, 1, 2, 0, 1, 2, 3),
                (1.5, 0.5, 2.5, 1.5, 0.5, 3.5, 2.5, 1.5, 0.5),
                strict=True,
            ):
                u, v = F[:, 0] - p, F[:, 1] - q
                G.append(0.5 * gamma - (u + v) ** 2 / 0.6 - (v - u) ** 2 / 2.4)
        else:
            for centre in [*np.eye(3), np.full(3, 1 / np.sqrt(3))]:
                G.append((0.5 * gamma) ** 2 - np.sum((F - centre) ** 2, axis=1))
        return F, np.column_stack(G), None


class DASCMOP1(_DASCMOP):
    number = 1


class DASCMOP2(_DASCMOP):
    number = 2


class DASCMOP3(_DASCMOP):
    number = 3


class DASCMOP4(_DASCMOP):
    number = 4


class DASCMOP5(_DASCMOP):
    number = 5


class DASCMOP6(_DASCMOP):
    number = 6


class DASCMOP7(_DASCMOP):
    number = 7


class DASCMOP8(_DASCMOP):
    number = 8


class DASCMOP9(_DASCMOP):
    number = 9
