"""Li et al. (2019) DC-DTLZ; DTLZ3 distance coefficient is 10, not 100."""

import numpy as np

from ...core import Problem, integer
from .._utils import linear_shape
from ..unconstrained_smooth.dtlz import _spherical


class _DCDTLZ(Problem):
    variant: int
    base: int

    def __init__(self, n_var=None, n_obj=3):
        m = integer(n_obj, "n_obj", 2)
        n = m + (4 if self.base == 1 else 9) if n_var is None else integer(n_var, "n_var", m)
        count = self.variant if self.variant < 3 else m
        super().__init__(f"DC{self.variant}-DTLZ{self.base}", n, m, 0, 1, count)

    def _evaluate(self, X):
        z = X[:, self.n_obj - 1 :] - 0.5
        g = (100 if self.base == 1 else 10) * np.sum(z * z + 1 - np.cos(20 * np.pi * z), axis=1)
        pos = X[:, : self.n_obj - 1]
        shape = 0.5 * linear_shape(pos) if self.base == 1 else _spherical(pos * np.pi / 2)
        F = (1 + g[:, None]) * shape
        if self.variant == 1:
            G = (0.5 - np.cos(3 * np.pi * X[:, 0]))[:, None]
        elif self.variant == 2:
            G = np.column_stack((0.5 - np.cos(3 * np.pi * g), 0.5 - np.exp(-g)))
        else:
            G = np.column_stack((0.5 - np.cos(3 * np.pi * pos), 0.5 - np.cos(3 * np.pi * g)))
        return F, G, None


class DC1DTLZ1(_DCDTLZ):
    variant, base = 1, 1


class DC1DTLZ3(_DCDTLZ):
    variant, base = 1, 3


class DC2DTLZ1(_DCDTLZ):
    variant, base = 2, 1


class DC2DTLZ3(_DCDTLZ):
    variant, base = 2, 3


class DC3DTLZ1(_DCDTLZ):
    variant, base = 3, 1


class DC3DTLZ3(_DCDTLZ):
    variant, base = 3, 3
