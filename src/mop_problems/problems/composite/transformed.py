"""Inverted/scaled DTLZ definitions of Jain and Deb (2014), Parts II/I."""

import numpy as np

from ...core import Problem
from .._utils import positive
from ..unconstrained_smooth.dtlz import DTLZ1, DTLZ2


class _TransformedDTLZ(Problem):
    number: int
    inverted: bool

    def __init__(self, n_var=None, n_obj=3, scale=2.0):
        self.scale = positive(scale, "scale")
        if self.inverted and self.scale != 2:
            raise ValueError("scale is only configurable for SDTLZ")
        self._base = (DTLZ1 if self.number == 1 else DTLZ2)(n_var, n_obj)
        super().__init__(
            f"{'I' if self.inverted else 'S'}DTLZ{self.number}",
            self._base.n_var,
            self._base.n_obj,
            0,
            1,
        )

    def _evaluate(self, X):
        f = self._base._evaluate(X)[0]
        if not self.inverted:
            return f * self.scale ** np.arange(self.n_obj), None, None
        if self.number == 1:
            radius = np.sum(f, axis=1)
        else:
            radius = 1 + np.sum((X[:, self.n_obj - 1 :] - 0.5) ** 2, axis=1)
        return radius[:, None] - f, None, None

    def pareto_front(self, n_points=100, *, seed=0):
        f = self._base.pareto_front(n_points, seed=seed)
        if self.inverted:
            return (0.5 if self.number == 1 else 1) - f
        return f * self.scale ** np.arange(self.n_obj)


class IDTLZ1(_TransformedDTLZ):
    number, inverted = 1, True


class IDTLZ2(_TransformedDTLZ):
    number, inverted = 2, True


class SDTLZ1(_TransformedDTLZ):
    number, inverted = 1, False


class SDTLZ2(_TransformedDTLZ):
    number, inverted = 2, False
