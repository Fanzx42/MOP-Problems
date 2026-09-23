"""Five constrained DTLZ variants with explicit raw G <= 0 residuals."""

import numpy as np

from ...core import Problem, integer
from ..unconstrained_smooth.dtlz import DTLZ1, DTLZ2, DTLZ3, DTLZ4


class _CDTLZ(Problem):
    base_class: type
    constraint: str
    canonical_name: str

    def __init__(self, n_var: int | None = None, n_obj: int = 3, **kwargs):
        n_obj = integer(n_obj, "n_obj", 2)
        n_var = n_obj + 9 if n_var is None else integer(n_var, "n_var", n_obj)
        self._base = self.base_class(n_var=n_var, n_obj=n_obj, **kwargs)
        count = n_obj if self.constraint.startswith("c3") else 1
        super().__init__(self.canonical_name, n_var, n_obj, 0, 1, n_ieq_constr=count)

    def _evaluate(self, X):
        F, _, _ = self._base._evaluate(X)
        if self.constraint == "c1_linear":
            G = (2 * F[:, :-1].sum(axis=1) + F[:, -1] / 0.6 - 1)[:, None]
        elif self.constraint == "c1_spherical":
            squared = (F**2).sum(axis=1)
            G = (-(squared - 16) * (squared - self.r**2))[:, None]
        elif self.constraint == "c2":
            axes = (F**2).sum(axis=1) - 2 * F.max(axis=1) + 1 - self.r**2
            centre = ((F - 1 / np.sqrt(self.n_obj)) ** 2).sum(axis=1) - self.r**2
            G = np.minimum(axes, centre)[:, None]
        elif self.constraint == "c3_linear":
            G = 1 - F.sum(axis=1, keepdims=True) - F
        else:
            G = 1 - (F**2).sum(axis=1, keepdims=True) + 0.75 * F**2
        return F, G, None


def _radius(r):
    if not np.isscalar(r) or not np.isfinite(r) or r <= 0:
        raise ValueError("r must be a finite positive scalar")
    return float(r)


class C1DTLZ1(_CDTLZ):
    """DTLZ1 under one linear constraint in objective space."""

    base_class, constraint, canonical_name = DTLZ1, "c1_linear", "C1-DTLZ1"


class C1DTLZ3(_CDTLZ):
    """DTLZ3 with an infeasible spherical shell."""

    base_class, constraint, canonical_name = DTLZ3, "c1_spherical", "C1-DTLZ3"

    def __init__(self, n_var: int | None = None, n_obj: int = 3, r: float | None = None):
        n_obj = integer(n_obj, "n_obj", 2)
        default = 9 if n_obj < 5 else 12.5 if n_obj <= 12 else 15
        self.r = _radius(default if r is None else r)
        super().__init__(n_var, n_obj)


class C2DTLZ2(_CDTLZ):
    """DTLZ2 restricted to a union of balls in objective space."""

    base_class, constraint, canonical_name = DTLZ2, "c2", "C2-DTLZ2"

    def __init__(self, n_var: int | None = None, n_obj: int = 3, r: float | None = None):
        n_obj = integer(n_obj, "n_obj", 2)
        default = 0.2 if n_obj == 2 else 0.4 if n_obj == 3 else 0.5
        self.r = _radius(default if r is None else r)
        super().__init__(n_var, n_obj)


class C3DTLZ1(_CDTLZ):
    """DTLZ1 with one linear constraint per objective."""

    base_class, constraint, canonical_name = DTLZ1, "c3_linear", "C3-DTLZ1"


class C3DTLZ4(_CDTLZ):
    """Biased DTLZ4 with one ellipsoidal constraint per objective."""

    base_class, constraint, canonical_name = DTLZ4, "c3_spherical", "C3-DTLZ4"

    def __init__(self, n_var: int | None = None, n_obj: int = 3, alpha: float = 100):
        super().__init__(n_var, n_obj, alpha=alpha)
