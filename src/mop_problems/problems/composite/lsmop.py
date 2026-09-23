"""LSMOP1-9, Cheng, Jin and Olhofer (2017); explicit logistic grouping."""

import numpy as np

from ...core import Problem, integer
from .._utils import linear_shape, rastrigin
from ..unconstrained_smooth.dtlz import _spherical


def _component(z, kind):
    if kind == "sphere":
        return np.sum(z * z, axis=1)
    if kind == "rastrigin":
        return rastrigin(z)
    if kind == "rosenbrock":
        return np.sum(100 * (z[:, :-1] ** 2 - z[:, 1:]) ** 2 + (z[:, :-1] - 1) ** 2, axis=1)
    if kind == "schwefel":
        return np.max(np.abs(z), axis=1)
    if kind == "griewank":
        return (
            np.sum(z * z, axis=1) / 4000
            - np.prod(np.cos(z / np.sqrt(np.arange(1, z.shape[1] + 1))), axis=1)
            + 1
        )
    return (
        20
        - 20 * np.exp(-0.2 * np.sqrt(np.mean(z * z, axis=1)))
        - np.exp(np.mean(np.cos(2 * np.pi * z), axis=1))
        + np.e
    )


class _LSMOP(Problem):
    number: int

    def __init__(self, n_var=None, n_obj=3, nk=5):
        m = integer(n_obj, "n_obj", 2)
        n = 100 * m if n_var is None else integer(n_var, "n_var", m)
        self.nk = integer(nk, "nk")
        c = [0.342]
        for _ in range(1, m):
            c.append(3.8 * c[-1] * (1 - c[-1]))
        self.sublen = np.floor(np.asarray(c) / sum(c) * (n - m + 1) / self.nk).astype(int)
        if np.any(self.sublen < 2):
            raise ValueError(
                "n_var too small: every logistic subgroup must contain at least two variables"
            )
        self.offsets = np.r_[0, np.cumsum(self.sublen * self.nk)]
        super().__init__(
            f"LSMOP{self.number}", n, m, 0, np.r_[np.ones(m - 1), np.full(n - m + 1, 10.0)]
        )

    def _groups(self, X):
        n, m, i = self.n_var, self.n_obj, self.number
        j = np.arange(m, n + 1) / n
        coefficient = 1 + j if i <= 4 else 1 + np.cos(np.pi * j / 2)
        tail = coefficient * X[:, m - 1 :] - 10 * X[:, 0, None]
        kinds = [
            ("sphere", "sphere"),
            ("griewank", "schwefel"),
            ("rastrigin", "rosenbrock"),
            ("ackley", "griewank"),
            ("sphere", "sphere"),
            ("rosenbrock", "schwefel"),
            ("ackley", "rosenbrock"),
            ("griewank", "sphere"),
            ("sphere", "ackley"),
        ][i - 1]
        columns = []
        for objective in range(m):
            width = self.sublen[objective]
            blocks = [
                tail[
                    :,
                    self.offsets[objective] + k * width : self.offsets[objective] + (k + 1) * width,
                ]
                for k in range(self.nk)
            ]
            columns.append(
                sum(_component(block, kinds[objective % 2]) for block in blocks) / (self.nk * width)
            )
        return np.column_stack(columns)

    def _evaluate(self, X):
        G = self._groups(X)
        pos = X[:, : self.n_obj - 1]
        if self.number <= 4:
            F = (1 + G) * linear_shape(pos)
        elif self.number <= 8:
            following = np.column_stack((G[:, 1:], np.zeros(len(X))))
            F = (1 + G + following) * _spherical(pos * np.pi / 2)
        else:
            distance = 2 + np.sum(G, axis=1)
            F = np.column_stack(
                (pos, self.n_obj * distance - np.sum(pos * (1 + np.sin(3 * np.pi * pos)), axis=1))
            )
        return F, None, None


class LSMOP1(_LSMOP):
    number = 1


class LSMOP2(_LSMOP):
    number = 2


class LSMOP3(_LSMOP):
    number = 3


class LSMOP4(_LSMOP):
    number = 4


class LSMOP5(_LSMOP):
    number = 5


class LSMOP6(_LSMOP):
    number = 6


class LSMOP7(_LSMOP):
    number = 7


class LSMOP8(_LSMOP):
    number = 8


class LSMOP9(_LSMOP):
    number = 9
