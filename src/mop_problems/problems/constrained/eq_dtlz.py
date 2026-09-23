"""Cuate et al. (2020) Eq-DTLZ/IDTLZ, exact H=0 spherical intersections.

The published supplement provides p=1 (r=.4) and p=2 (r=.5) instances.
The reference's abs(H)-1e-4 is a feasibility tolerance, not the equality itself.
"""

import numpy as np

from ...core import Problem, integer
from ..composite.transformed import IDTLZ1, IDTLZ2
from ..unconstrained_smooth.dtlz import DTLZ1, DTLZ2, DTLZ3, DTLZ4


class _EqDTLZ(Problem):
    number: int
    inverted = False

    def __init__(self, n_var=None, n_obj=3, n_equalities=1):
        count = integer(n_equalities, "n_equalities")
        if count not in (1, 2):
            raise ValueError("The published supplement supports n_equalities=1 or 2")
        m = integer(n_obj, "n_obj", count + 1)
        base_cls = (
            (IDTLZ1, IDTLZ2)[self.number - 1]
            if self.inverted
            else (DTLZ1, DTLZ2, DTLZ3, DTLZ4)[self.number - 1]
        )
        self._base = base_cls(n_var=n_var, n_obj=m)
        self.radius = 0.4 if count == 1 else 0.5
        super().__init__(
            f"Eq-{'I' if self.inverted else ''}DTLZ{self.number}",
            self._base.n_var,
            m,
            0,
            1,
            n_eq_constr=count,
        )

    def _evaluate(self, X):
        F = self._base._evaluate(X)[0]
        z = X[:, : self.n_obj - 1] - 0.5
        first = np.sum(z * z, axis=1) - self.radius**2
        H = [first]
        if self.n_eq_constr == 2:
            H.append(first - 2 * self.radius * z[:, -1] + self.radius**2)
        return F, None, np.column_stack(H)


class EqDTLZ1(_EqDTLZ):
    number = 1


class EqDTLZ2(_EqDTLZ):
    number = 2


class EqDTLZ3(_EqDTLZ):
    number = 3


class EqDTLZ4(_EqDTLZ):
    number = 4


class EqIDTLZ1(_EqDTLZ):
    number, inverted = 1, True


class EqIDTLZ2(_EqDTLZ):
    number, inverted = 2, True
