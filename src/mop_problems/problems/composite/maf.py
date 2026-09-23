"""CEC 2017 MaF1-15, Cheng et al. (2017).

MaF9 rejects the original invalid polygon regions instead of randomly repairing
an evaluated input. Its sample() draws reproducibly from the valid domain.
"""

import numpy as np

from ...core import Problem, integer
from .._utils import linear_shape
from ..unconstrained_smooth.dtlz import _spherical
from .lsmop import LSMOP3, LSMOP8
from .wfg import WFG1, WFG2, WFG9


def _inside_polygon(x, vertices):
    inside = np.zeros(len(x), dtype=bool)
    on_boundary = inside.copy()
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
        edge = b - a
        residual = x - a
        cross = residual[:, 0] * edge[1] - residual[:, 1] * edge[0]
        dot = np.sum(residual * edge, axis=1)
        on_boundary |= (
            (np.abs(cross) < 1e-12) & (dot >= -1e-12) & (dot <= np.sum(edge * edge) + 1e-12)
        )
        if edge[1] != 0:
            intersects = ((a[1] > x[:, 1]) != (b[1] > x[:, 1])) & (
                x[:, 0] < a[0] + (x[:, 1] - a[1]) * edge[0] / edge[1]
            )
            inside ^= intersects
    return inside | on_boundary


class _MaF(Problem):
    number: int

    def __init__(self, n_var=None, n_obj=None):
        i = self.number
        m = (
            (10 if i in (8, 9) else 3)
            if n_obj is None
            else integer(n_obj, "n_obj", 3 if i in (8, 9, 13) else 2)
        )
        default_n = (
            2
            if i in (8, 9)
            else 5
            if i == 13
            else 20 * m
            if i >= 14
            else m + 19
            if i == 7
            else m + 9
        )
        n = (
            default_n
            if n_var is None
            else integer(n_var, "n_var", 2 if i in (8, 9) else 5 if i == 13 else m)
        )
        lo, hi = np.zeros(n), np.ones(n)
        if i in (8, 9):
            if n != 2:
                raise ValueError("MaF8/9 have exactly two decision variables")
            theta = np.pi / 2 - 2 * np.pi * np.arange(1, m + 1) / m
            self.vertices = np.column_stack((np.cos(theta), np.sin(theta)))
            self._polygons = []
            for width in range(1, int(np.ceil(m / 2 - 2)) + 1):
                for start in range(m):
                    end = start + width
                    points = self.vertices[np.arange(start, end + 1) % m]
                    a, b, c, d = self.vertices[np.array([start - 1, start, end, end + 1]) % m]
                    t = np.linalg.solve(np.column_stack((b - a, -(d - c))), c - a)[0]
                    centre = a + t * (b - a)
                    self._polygons.append(np.concatenate((points, 2 * centre - points)))
            lo[:], hi[:] = -10000, 10000
        elif i in (10, 11, 12):
            self._base = (WFG1, WFG2, WFG9)[i - 10](n_var=n, n_obj=m, k=m - 1)
            hi = self._base.xu
        elif i == 13:
            lo[2:], hi[2:] = -2, 2
        elif i >= 14:
            self._base = (LSMOP3 if i == 14 else LSMOP8)(n_var=n, n_obj=m, nk=2)
            hi = self._base.xu
        super().__init__(f"MaF{i}", n, m, lo, hi)

    def domain_mask(self, X):
        x = super()._validate_input(X, check_bounds=False)
        if self.number != 9:
            return np.ones(len(x), dtype=bool)
        forbidden = np.zeros(len(x), dtype=bool)
        for polygon in self._polygons:
            forbidden |= _inside_polygon(x, polygon)
        return ~forbidden | _inside_polygon(x, self.vertices)

    def _validate_input(self, X, *, check_bounds=True):
        x = super()._validate_input(X, check_bounds=check_bounds)
        if self.number == 9 and not self.domain_mask(x).all():
            raise ValueError(
                "MaF9: input in an excluded polygon; inspect domain_mask() or sample()"
            )
        return x

    def sample(self, n, seed=None):
        rng = np.random.default_rng(seed)
        x = super().sample(n, rng)
        if self.number == 9:
            invalid = ~self.domain_mask(x)
            while invalid.any():
                x[invalid] = super().sample(int(invalid.sum()), rng)
                invalid = ~self.domain_mask(x)
        return x

    def _evaluate(self, X):
        i, m = self.number, self.n_obj
        if i in (10, 11, 12, 14):
            return self._base._evaluate(X)
        if i == 15:
            G = self._base._groups(X)
            factors = 1 + G + np.column_stack((G[:, 1:], np.zeros(len(X))))
            return factors * (1 - _spherical(X[:, : m - 1] * np.pi / 2)), None, None
        if i == 8:
            F = np.column_stack([np.sqrt(np.sum((X - v) ** 2, axis=1)) for v in self.vertices])
        elif i == 9:
            F = np.column_stack(
                [
                    np.abs((X[:, 0] - a[0]) * (b[1] - a[1]) - (X[:, 1] - a[1]) * (b[0] - a[0]))
                    / np.linalg.norm(b - a)
                    for a, b in zip(self.vertices, np.roll(self.vertices, -1, axis=0), strict=True)
                ]
            )
        elif i == 13:
            j = np.arange(3, self.n_var + 1)
            y = X[:, 2:] - 2 * X[:, 1, None] * np.sin(
                2 * np.pi * X[:, 0, None] + j * np.pi / self.n_var
            )
            shape = _spherical(X[:, :2] * np.pi / 2)[:, ::-1]
            F3 = shape + 2 * np.column_stack(
                [np.mean(y[:, j % 3 == r] ** 2, axis=1) for r in (1, 2, 0)]
            )
            if m == 3:
                F = F3
            else:
                last = (
                    F3[:, 0] ** 2
                    + F3[:, 1] ** 10
                    + F3[:, 2] ** 10
                    + 2 * np.mean(y[:, 1:] ** 2, axis=1)
                )
                F = np.column_stack((F3, *[last for _ in range(m - 3)]))
        else:
            pos, tail = X[:, : m - 1], X[:, m - 1 :]
            z = tail - 0.5
            g = np.sum(z * z, axis=1)
            if i in (3, 4):
                g = 100 * np.sum(z * z + 1 - np.cos(20 * np.pi * z), axis=1)
            if i == 1:
                F = (1 + g[:, None]) * (1 - linear_shape(pos))
            elif i == 2:
                width = tail.shape[1] // m
                groups = [
                    np.sum(
                        (tail[:, j * width : (j + 1) * width if j < m - 1 else None] / 2 - 0.25)
                        ** 2,
                        axis=1,
                    )
                    for j in range(m)
                ]
                F = (1 + np.column_stack(groups)) * _spherical((pos / 2 + 0.25) * np.pi / 2)
            elif i == 3:
                F = ((1 + g[:, None]) * _spherical(pos * np.pi / 2)) ** np.r_[np.full(m - 1, 4), 2]
            elif i == 4:
                F = (
                    (1 + g[:, None])
                    * (1 - _spherical(pos * np.pi / 2))
                    * 2.0 ** np.arange(1, m + 1)
                )
            elif i == 5:
                F = (1 + g[:, None]) * _spherical(pos**100 * np.pi / 2) * 2.0 ** np.arange(m, 0, -1)
            elif i == 6:
                theta = np.column_stack(
                    (pos[:, 0], (1 + 2 * g[:, None] * pos[:, 1:]) / (2 + 2 * g[:, None]))
                )
                F = (1 + 100 * g[:, None]) * _spherical(theta * np.pi / 2)
            else:
                radius = 2 + 9 * np.mean(tail, axis=1)
                F = np.column_stack(
                    (pos, m * radius - np.sum(pos * (1 + np.sin(3 * np.pi * pos)), axis=1))
                )
        return F, None, None


class MaF1(_MaF):
    number = 1


class MaF2(_MaF):
    number = 2


class MaF3(_MaF):
    number = 3


class MaF4(_MaF):
    number = 4


class MaF5(_MaF):
    number = 5


class MaF6(_MaF):
    number = 6


class MaF7(_MaF):
    number = 7


class MaF8(_MaF):
    number = 8


class MaF9(_MaF):
    number = 9


class MaF10(_MaF):
    number = 10


class MaF11(_MaF):
    number = 11


class MaF12(_MaF):
    number = 12


class MaF13(_MaF):
    number = 13


class MaF14(_MaF):
    number = 14


class MaF15(_MaF):
    number = 15
