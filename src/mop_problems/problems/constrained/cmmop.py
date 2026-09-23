"""CMMOP1--14, Ming et al. (2023), author CMMOP-code archive."""

import numpy as np

from ...core import Problem
from .._utils import mark_boundary


class _CMMOP(Problem):
    number: int

    def __init__(self):
        i = self.number
        bounds = {
            2: ([0, 0], [1, 2]),
            3: ([1, -1], [3, 3]),
            4: ([-1, 0], [1, 2]),
            5: ([-np.pi, 0], [np.pi, 9]),
            6: ([0, 0], [1, 1.5]),
            7: ([1, -1], [3, 2]),
            11: ([1, -1, -1], [3, 1, 1]),
            12: ([-1, 0, 0], [1, 2, 2]),
        }
        lo, hi = bounds.get(i, ([1, -1], [3, 1]))
        ng = {5: 2, 8: 3, 9: 4, 10: 2}.get(i, 1)
        super().__init__(f"CMMOP{i}", len(lo), 2, lo, hi, n_ieq_constr=ng)

    def _evaluate(self, X):
        i, x, y = self.number, X[:, 0], X[:, 1]
        a = x if i in (2, 6) else np.abs(x) if i in (4, 5, 12) else np.abs(x - 2)
        switches = []
        if i in (2, 6):
            keep = y <= 1 if i == 2 else ((y <= 0.5) | ((y < 1) & (x > 0.25)))
            z = y - np.where(keep, 0, 1 if i == 2 else 0.5) - np.sqrt(x)
            b = (1 - np.sqrt(x) if i == 2 else 1.21 - x * x) + 2 * (
                4 * z * z - 2 * np.cos(20 * np.pi * z / np.sqrt(2)) + 2
            )
            switches = [y - 1] if i == 2 else [y - 0.5, y - 1, x - 0.25]
        elif i in (4, 12):
            tail = X[:, 1:]
            z = tail - np.where(tail < 1, 0, 1) - np.sin(np.pi * a[:, None])
            b = (1.21 - x * x if i == 4 else 1 - a) + 2 * np.sum(z * z, axis=1)
            switches = [tail[:, j] - 1 for j in range(tail.shape[1])]
        elif i == 5:
            z = y - np.where(y <= 4, 0, 4) - np.sin(a) - a
            b = np.abs(np.cos(a)) + 2 * z * z
            a = np.sin(a)
            switches = [y - 4]
        elif i == 8:
            z = y - (0.3 * a * a * np.cos(24 * np.pi * a + 4 * np.pi) + 0.6 * a) * np.sin(
                6 * np.pi * a + np.pi
            )
            b = np.sqrt(1 - a * a) + z * z
        else:
            offset = y * 0
            if i == 3:
                offset = np.where(y <= 1, 0, 2)
                switches = [y - 1]
            if i == 7:
                stripes = x <= 7 / 6
                for lo, hi in [
                    (8 / 6, 9 / 6),
                    (10 / 6, 11 / 6),
                    (13 / 6, 14 / 6),
                    (15 / 6, 16 / 6),
                ]:
                    stripes |= (x > lo) & (x <= hi)
                stripes |= x > 17 / 6
                keep = (y <= 0) | ((y <= 1) & stripes)
                offset = np.where(keep, 0, 1)
                switches = [y, y - 1, *[x - k / 6 for k in (7, 8, 9, 10, 11, 13, 14, 15, 16, 17)]]
            z = y - offset - np.sin(6 * np.pi * a + np.pi)
            b = (1 - np.sqrt(a) if i in (3, 7) else 1 - a) + 2 * z * z
            if i == 11:
                b = 1 - a + z * z + (X[:, 2] - np.sin(6 * np.pi * a + np.pi)) ** 2
            elif i == 13:
                b += 2 * z * z
            elif i == 14:
                b -= np.expm1(-10 * z * z)
        F = mark_boundary(np.column_stack((a, b)), *switches)
        a, b = F[:, 0], F[:, 1]
        s = a + b
        ell = np.sqrt(2) * (b - a)
        if i in (4, 6):
            wave = np.cos(6 * np.arctan2(b, a) ** (4 if i == 4 else 5)) ** 10
            c = [(a / (1 + 0.15 * wave)) ** 2 + (b / (1 + 0.75 * wave)) ** 2 - 1]
        elif i in (5, 8):
            theta = np.arctan2(b, a)
            angle = np.pi / 2 - 2 * np.abs(theta - np.pi / 4)
            norm = a * a + b * b
            c = [norm - (1.7 - 0.2 * np.sin(2 * theta)) ** 2]
            if i == 8:
                c.append(a * 0)  # Empty second column in the author's MATLAB definition.
            c.append((1 - (0.5 if i == 5 else 0.7) * np.sin(6 * angle**3)) ** 2 - norm)
        elif i == 9:
            in_second = (s >= 1.5) & (s <= 2)
            in_first = (s >= 1) & (s <= 1.1)
            c = [
                np.where(in_second, 0, 1 - s),
                np.where(in_second, 0, s - 1.1),
                np.where(in_first | in_second, 0, 1.5 - s),
                np.where(in_first | in_second, 0, s - 2),
            ]
        elif i == 10:
            c = [
                s - 1.05 - 0.45 * np.sin(0.75 * np.pi * ell) ** 6,
                0.85 - s + 0.3 * np.sin(0.75 * np.pi * ell) ** 2,
            ]
        elif i in (2, 3, 7):
            c = [
                np.sqrt(a)
                + b
                - 1
                - ({2: 0.3, 3: 0.5, 7: 0.3}[i])
                * np.sin((3 if i == 2 else 2) * np.pi * ell) ** (12 if i == 7 else 8)
            ]
        else:
            c = [s - 1 - 0.6 * np.sin(2 * np.pi * ell) ** 8]
        G = np.column_stack(c)
        if i == 9:
            G = mark_boundary(G, *[s - v for v in (1, 1.1, 1.5, 2)])
        return F, G, None


class CMMOP1(_CMMOP):
    number = 1


class CMMOP2(_CMMOP):
    number = 2


class CMMOP3(_CMMOP):
    number = 3


class CMMOP4(_CMMOP):
    number = 4


class CMMOP5(_CMMOP):
    number = 5


class CMMOP6(_CMMOP):
    number = 6


class CMMOP7(_CMMOP):
    number = 7


class CMMOP8(_CMMOP):
    number = 8


class CMMOP9(_CMMOP):
    number = 9


class CMMOP10(_CMMOP):
    number = 10


class CMMOP11(_CMMOP):
    number = 11


class CMMOP12(_CMMOP):
    number = 12


class CMMOP13(_CMMOP):
    number = 13


class CMMOP14(_CMMOP):
    number = 14
