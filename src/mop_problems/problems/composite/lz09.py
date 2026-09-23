"""LZ09_F1-9, Li and Zhang (2009), jMetal unit-box parameterization."""

import numpy as np

from ...core import Problem, integer
from ..unconstrained_smooth.dtlz import _spherical


class _LZ09(Problem):
    number: int

    def __init__(self, n_var=None):
        i = self.number
        n = (
            (10 if i in (1, 6, 7, 8) else 30)
            if n_var is None
            else integer(n_var, "n_var", 5 if i == 6 else 3)
        )
        super().__init__(f"LZ09_F{i}", n, 3 if i == 6 else 2, 0, 1)

    def _evaluate(self, X):
        i, n = self.number, self.n_var
        x = X[:, 0]
        if i == 6:
            j = np.arange(3, n + 1)
            residual = 4 * (X[:, 2:] - 0.5) - 2 * X[:, 1, None] * np.sin(
                2 * np.pi * x[:, None] + j * np.pi / n
            )
            penalty = 2 * np.column_stack(
                [np.mean(residual[:, j % 3 == r] ** 2, axis=1) for r in (2, 1, 0)]
            )
            return _spherical(X[:, :2] * np.pi / 2) + penalty, None, None
        j = np.arange(2, n + 1)
        theta = 6 * np.pi * x[:, None] + j * np.pi / n
        if i in (1, 7, 8):
            target = x[:, None] ** ((n + 3 * j - 8) / (2 * (n - 2)))
        elif i in (2, 9):
            target = np.sin(theta)
        else:
            radius = (
                0.3 * x[:, None] * (x[:, None] * np.cos(4 * theta) + 2)
                if i == 5
                else 0.8 * x[:, None]
            )
            target = radius * np.where(
                j % 2 == 1, np.cos(theta / 3 if i == 4 else theta), np.sin(theta)
            )
        residual = 2 * (X[:, 1:] - 0.5) - target

        def penalty(mask):
            z = residual[:, mask]
            if i == 7:
                return 2 * np.mean(4 * z * z - np.cos(8 * np.pi * z) + 1, axis=1)
            if i == 8:
                return (
                    2
                    / z.shape[1]
                    * (
                        4 * np.sum(z * z, axis=1)
                        - 2
                        * np.prod(
                            np.cos(20 * np.pi * z / np.sqrt(np.arange(1, z.shape[1] + 1))), axis=1
                        )
                        + 2
                    )
                )
            return 2 * np.mean(z * z, axis=1)

        return (
            np.column_stack(
                (
                    x + penalty(j % 2 == 0),
                    1 - (x * x if i == 9 else np.sqrt(x)) + penalty(j % 2 == 1),
                )
            ),
            None,
            None,
        )


class LZ09_F1(_LZ09):
    number = 1


class LZ09_F2(_LZ09):
    number = 2


class LZ09_F3(_LZ09):
    number = 3


class LZ09_F4(_LZ09):
    number = 4


class LZ09_F5(_LZ09):
    number = 5


class LZ09_F6(_LZ09):
    number = 6


class LZ09_F7(_LZ09):
    number = 7


class LZ09_F8(_LZ09):
    number = 8


class LZ09_F9(_LZ09):
    number = 9
