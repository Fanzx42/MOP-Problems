"""DAS-CMaOP1--9: WFG objectives and Fan et al. (2020), equation (8)."""

import numpy as np

from ...core import Problem, integer
from ..composite import wfg
from .das_cmop import DIFFICULTIES


class _DASCMaOP(Problem):
    number: int

    def __init__(self, n_var=30, n_obj=3, k=None, difficulty=4):
        m = integer(n_obj, "n_obj", 2)
        k = 2 * (m - 1) if k is None else integer(k, "k")
        self._base = getattr(wfg, f"WFG{self.number}")(n_var=n_var, n_obj=m, k=k)
        self.k = k
        if isinstance(difficulty, (int, np.integer)) and not isinstance(
            difficulty, (bool, np.bool_)
        ):
            if not 1 <= difficulty <= 12:
                raise ValueError("difficulty index must be in 1..12")
            triple = DIFFICULTIES[difficulty - 1]
        else:
            triple = np.asarray(difficulty, dtype=float)
            if (
                triple.shape != (3,)
                or not np.isfinite(triple).all()
                or np.any((triple < 0) | (triple > 1))
            ):
                raise ValueError("difficulty must be an index or three values in [0, 1]")
        self.difficulty = tuple(float(v) for v in triple)
        super().__init__(
            f"DASCMaOP{self.number}",
            self._base.n_var,
            m,
            self._base.xl,
            self._base.xu,
            n_ieq_constr=2 * m + 1,
        )

    def _evaluate(self, X):
        eta, zeta, gamma = self.difficulty
        pos, distance = self._base._transform(X)
        f = self._base._evaluate(X)[0]
        zero = distance[:, 0] * 0
        c = [
            eta - (np.sin if j % 2 == 0 else np.cos)(20 * np.pi * pos[:, j]) if eta else zero
            for j in range(self.n_obj - 1)
        ]
        d = distance[:, 0]
        c.append((d - 0.5) * (d - 0.5 - 10 ** (-2 * zeta)) if zeta else zero)
        q = f / (2 * np.arange(1, self.n_obj + 1))
        for j in range(self.n_obj):
            c.append(
                (0.5 * gamma) ** 2 - ((q - np.eye(self.n_obj)[j]) ** 2).sum(axis=1)
                if gamma
                else zero
            )
        c.append(
            (0.5 * gamma) ** 2 - ((q - 1 / np.sqrt(self.n_obj)) ** 2).sum(axis=1) if gamma else zero
        )
        return f, np.column_stack(c), None


class DASCMaOP1(_DASCMaOP):
    number = 1


class DASCMaOP2(_DASCMaOP):
    number = 2


class DASCMaOP3(_DASCMaOP):
    number = 3


class DASCMaOP4(_DASCMaOP):
    number = 4


class DASCMaOP5(_DASCMaOP):
    number = 5


class DASCMaOP6(_DASCMaOP):
    number = 6


class DASCMaOP7(_DASCMaOP):
    number = 7


class DASCMaOP8(_DASCMaOP):
    number = 8


class DASCMaOP9(_DASCMaOP):
    number = 9
