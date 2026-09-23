"""LSCM1--12 of Qiao et al. (2024), exposed as LSCMOP1--12.

The author's aggregate constraint and numerical cutoffs are retained. This is
a piecewise, partly discontinuous benchmark, not a smooth constrained reformulation.
"""

import numpy as np

from ...core import Problem, integer
from .._utils import linear_shape, mark_boundary
from ..unconstrained_smooth.dtlz import _spherical
from .doc import DOC4, DOC7, DOC8, DOC9

_CONFIG = (
    ([5, 4, 6, 6, 3, 5, 5, 5, 5, 2], [2] * 10, 1, 2, (1, 2), 1, 1),
    ([5, 4, 6, 6, 3, 5, 5, 5, 5, 2], [6] * 10, 1, 1, (1, 1), 3, 1),
    ([5, 4, 6, 6, 3, 5, 5, 5, 5, 2], [18] * 10, 1, 1, (1, 3), 2, 2),
    ([5, 4, 3, 5, 4, 3, 5, 4, 3, 5], [6, 24] * 5, 2, 2, (4, 3), 3, 1),
    ([5, 4, 3, 5, 4, 3, 5, 4, 3, 5], [2, 1] * 5, 1, 1, (2, 3), 1, 3),
    ([5, 3, 5, 4, 3, 5, 4, 3, 5, 2], [11, 12] * 5, 2, 1, (1, 5), 1, 1),
    ([3, 6, 5, 6, 3, 6, 5, 6, 3, 6], [3, 24, 6, 3, 24, 6, 3, 24, 6, 3], 1, 1, (4, 5), 1, 2),
    ([4, 4, 5, 6, 3, 5, 5, 5, 5, 2], [18, 24, 14, 18, 24, 14, 18, 24, 14, 18], 2, 2, (1, 1), 2, 1),
    ([4, 5] * 5, [6, 11, 24, 6, 11, 24, 6, 11, 24, 6], 1, 1, (1, 4), 2, 2),
    ([4, 5] * 5, [3] * 10, 2, 2, (1, 2), 1, 1),
    ([2, 3] * 5, [6, 10] * 5, 1, 2, (2, 3), 2, 2),
    ([11, 4, 3, 11, 4, 3, 11, 4, 3, 11], [6, 9, 10, 6, 6, 9, 10, 6, 6, 9], 1, 1, (1, 1), 3, 1),
)
_CEC = {
    1: ([0] * 13, [1] * 9 + [100] * 3 + [1], -15),
    2: ([0] * 20, [10] * 20, -0.803619),
    3: ([0] * 10, [1] * 10, -1),
    6: ([13, 0], [100, 100], -6961.81388),
    9: ([-10] * 7, [10] * 7, 680.6300573),
    10: ([100, 1000, 1000, 10, 10, 10, 10, 10], [10000] * 3 + [1000] * 5, 7049.2480),
    11: ([-1, -1], [1, 1], 0.75),
    12: ([0] * 3, [10] * 3, -1),
    14: ([0] * 10, [10] * 10, -47.7648884595),
    18: ([-10] * 8 + [0], [10] * 8 + [20], -0.8660254038),
    24: ([0, 0], [3, 4], -5.5080132716),
}


def _cec(p, number):
    x = [None] + [p[:, j] for j in range(p.shape[1])]
    if number == 1:
        f = 5 * np.sum(p[:, :4] - p[:, :4] ** 2, axis=1) - p[:, 4:].sum(axis=1)
        g = [
            2 * x[1] + 2 * x[2] + x[10] + x[11] - 10,
            2 * x[1] + 2 * x[3] + x[10] + x[12] - 10,
            2 * x[2] + 2 * x[3] + x[11] + x[12] - 10,
            -8 * x[1] + x[10],
            -8 * x[2] + x[11],
            -8 * x[3] + x[12],
            -2 * x[4] - x[5] + x[10],
            -2 * x[6] - x[7] + x[11],
            -2 * x[8] - x[9] + x[12],
        ]
    elif number == 2:
        f = -np.abs((np.cos(p) ** 4).sum(axis=1) - 2 * (np.cos(p) ** 2).prod(axis=1)) / np.sqrt(
            1e-30 + (p * p * np.arange(1, p.shape[1] + 1)).sum(axis=1)
        )
        g = [0.75 - p.prod(axis=1), p.sum(axis=1) - 7.5 * p.shape[1]]
    elif number == 3:
        f = -1e5 * p.prod(axis=1)
        g = [np.abs((p * p).sum(axis=1) - 1) - 1e-4]
    elif number == 6:
        f = (x[1] - 10) ** 3 + (x[2] - 20) ** 3
        g = [100 - (x[1] - 5) ** 2 - (x[2] - 5) ** 2, (x[1] - 6) ** 2 + (x[2] - 5) ** 2 - 82.81]
    elif number in (9, 10, 14, 18):
        cls, head, offset, start = {
            9: (DOC4, 1, 1 - 680.6300573745, 2),
            10: (DOC8, 2, 1 - 7049.2480205286, 1),
            14: (DOC7, 1, 48.7648884595, 3),
            18: (DOC9, 2, 1.8660254038, 1),
        }[number]
        F, G, _ = cls()._evaluate(np.column_stack([*[p[:, 0] * 0 for _ in range(head)], p]))
        f = F[:, 0 if number == 18 else -1] - offset
        g = [G[:, j] for j in range(start, G.shape[1])]
    elif number == 11:
        f = x[1] ** 2 + (x[2] - 1) ** 2
        g = [np.abs(x[2] - x[1] ** 2) - 1e-4]
    elif number == 12:
        f = -1 + np.sum((p - 5) ** 2, axis=1) / 100
        nearest = np.clip(np.floor(p + 0.5), 1, 9)
        g = [np.sum((p - nearest) ** 2, axis=1) - 0.0625]
    elif number == 24:
        f = -x[1] - x[2]
        g = [
            -2 * x[1] ** 4 + 8 * x[1] ** 3 - 8 * x[1] ** 2 + x[2] - 2,
            -4 * x[1] ** 4 + 32 * x[1] ** 3 - 88 * x[1] ** 2 + 96 * x[1] + x[2] - 36,
        ]
    else:
        raise ValueError("Unsupported CEC2006 component")
    residual = np.column_stack(g)
    violation = np.maximum(residual, 0).sum(axis=1)
    violation = mark_boundary(np.where(violation < 1e-6, 0, violation), violation - 1e-6)
    shifted = f - _CEC[number][2]
    shifted = mark_boundary(
        np.where(np.abs(shifted) <= 0.001, 0, shifted), shifted - 0.001, shifted + 0.001
    )
    return shifted, violation


def _distance(p, kind):
    z = 10 * p
    if kind == 1:
        f = (z * z).sum(axis=1)
    elif kind == 2:
        f = np.abs(z).max(axis=1)
    elif kind == 3:
        f = (z * z - 10 * np.cos(2 * np.pi * z) + 10).sum(axis=1)
    elif kind == 4:
        f = (
            (z * z).sum(axis=1) / 4000
            - np.cos(z / np.sqrt(np.arange(1, z.shape[1] + 1))).prod(axis=1)
            + 1
        )
    else:
        f = (
            20
            - 20 * np.exp(-0.2 * np.sqrt((z * z).mean(axis=1)))
            - np.exp(np.cos(2 * np.pi * z).mean(axis=1))
            + np.e
        )
    return mark_boundary(np.where(f < 1e-8, 0, f), f - 1e-8)


class _LSCMOP(Problem):
    number: int

    def __init__(self, n_var=100):
        n = integer(n_var, "n_var", 100)
        base = int(np.ceil(n / 1000)) * 100
        if n % base:
            raise ValueError(f"n_var must be a multiple of {base}; no implicit dimension rounding")
        m = 2 if self.number <= 9 else 3
        self._config = _CONFIG[self.number - 1]
        nk, cec, *_ = self._config
        self._groups = []
        c = [0.342]
        for _ in range(m - 1):
            c.append(3.8 * c[-1] * (1 - c[-1]))
        c = np.array(c) / sum(c)
        for segment in range(n // base):
            number = cec[segment]
            width = len(_CEC[number][0])
            first = segment * base + (m if segment == 0 else 0)
            available = base - width - (m if segment == 0 else 0)
            lengths = np.floor(c * available / nk[segment]).astype(int)
            if np.any(lengths < 1):
                raise ValueError("n_var gives an empty distance block")
            blocks = []
            cursor = first + width
            for length in lengths:
                blocks.append(
                    [
                        slice(cursor + j * length, cursor + (j + 1) * length)
                        for j in range(nk[segment])
                    ]
                )
                cursor += nk[segment] * length
            self._groups.append((number, slice(first, first + width), blocks, lengths, cursor))
        super().__init__(f"LSCMOP{self.number}", n, m, 0, 1, n_ieq_constr=1)

    def _evaluate(self, X):
        _, _, cct, dct, dist, pf, qform = self._config
        columns = [X[:, 0] * 0 for _ in range(self.n_obj)]
        total = X[:, 0] * 0
        for seg, (number, cs, blocks, lengths, end) in enumerate(self._groups):
            control = X[:, 0 if seg == 0 else self._groups[seg - 1][-1] - 1]
            pp = X[:, cs] + control[:, None]
            remainder = pp - 0.5 * np.floor(pp / 0.5)
            linked = 1 - 2 * remainder if cct == 1 else np.cos(np.pi * remainder)
            lo, hi, _ = _CEC[number]
            f, cv = _cec(np.array(lo) + (np.array(hi) - lo) * linked, number)
            total += cv
            start = cs.stop
            j = np.arange(1, end - start + 1) / (end - start)
            coefficient = 1 + (j if dct == 1 else np.cos(np.pi * j / 2))
            for obj, bset in enumerate(blocks):
                value = control * 0
                for sl in bset:
                    z = (
                        coefficient[sl.start - start : sl.stop - start] * X[:, sl]
                        - control[:, None]
                    )
                    value += _distance(z, dist[obj % 2])
                columns[obj] += f / self.n_obj + value / lengths[obj] / len(bset)
        g = np.column_stack(columns)
        pos = X[:, : self.n_obj - 1]
        if pf == 3:
            F = np.column_stack(
                (
                    pos,
                    (2 + g[:, -1]) * self.n_obj - (pos * (1 + np.sin(3 * np.pi * pos))).sum(axis=1),
                )
            )
        else:
            if qform == 2:
                g = g + np.column_stack((g[:, 1:], g[:, 0] * 0))
            elif qform == 3:
                g = g.sum(axis=1, keepdims=True)
            shape = linear_shape(pos) if pf == 1 else _spherical(pos * np.pi / 2)
            F = (1 + g) * shape
        return F, total[:, None], None


class LSCMOP1(_LSCMOP):
    number = 1


class LSCMOP2(_LSCMOP):
    number = 2


class LSCMOP3(_LSCMOP):
    number = 3


class LSCMOP4(_LSCMOP):
    number = 4


class LSCMOP5(_LSCMOP):
    number = 5


class LSCMOP6(_LSCMOP):
    number = 6


class LSCMOP7(_LSCMOP):
    number = 7


class LSCMOP8(_LSCMOP):
    number = 8


class LSCMOP9(_LSCMOP):
    number = 9


class LSCMOP10(_LSCMOP):
    number = 10


class LSCMOP11(_LSCMOP):
    number = 11


class LSCMOP12(_LSCMOP):
    number = 12
