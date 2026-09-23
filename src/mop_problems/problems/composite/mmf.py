"""Original MMF1-8, Yue, Qu and Liang (2018); later suffixed variants excluded."""

import numpy as np

from ...core import Problem
from .._utils import mark_boundary


class _MMF(Problem):
    number: int

    def __init__(self):
        bounds = [
            (1, -1, 3, 1),
            (0, 0, 1, 2),
            (0, 0, 1, 1.5),
            (-1, 0, 1, 2),
            (1, -1, 3, 3),
            (1, -1, 3, 2),
            (1, -1, 3, 1),
            (-np.pi, 0, np.pi, 9),
        ]
        a, b, c, d = bounds[self.number - 1]
        super().__init__(f"MMF{self.number}", 2, 2, [a, b], [c, d])

    def _evaluate(self, X):
        x, y = X[:, 0], X[:, 1]
        i = self.number
        boundaries = []
        if i in (2, 3):
            f1 = x
            if i == 2:
                shift = np.where(y <= 1, 0.0, 1.0)
                boundaries = [y - 1]
            else:
                shift = np.where((y <= 0.5) | ((y < 1) & (x > 0.25)), 0.0, 0.5)
                boundaries = [y - 0.5, y - 1, x - 0.25]
            d = y - shift - np.sqrt(x)
            f2 = 1 - np.sqrt(x) + 2 * (4 * d * d - 2 * np.cos(20 * np.pi * d / np.sqrt(2)) + 2)
        elif i == 4:
            f1 = np.abs(x)
            d = y - np.where(y < 1, 0.0, 1.0) - np.sin(np.pi * f1)
            f2 = 1 - x * x + 2 * d * d
            boundaries = [y - 1]
        elif i == 8:
            a = np.abs(x)
            f1 = np.sin(a)
            d = y - np.where(y <= 4, 0.0, 4.0) - np.sin(a) - a
            # |cos(a)| is algebraically sqrt(1-sin(a)^2), with better rounding.
            f2 = np.abs(np.cos(a)) + 2 * d * d
            boundaries = [y - 4]
        else:
            f1 = np.abs(x - 2)
            shift = 0
            if i == 5:
                shift = np.where(y <= 1, 0.0, 2.0)
                boundaries = [y - 1]
            elif i == 6:
                selected = (
                    (x <= 7 / 6)
                    | ((x > 8 / 6) & (x <= 9 / 6))
                    | ((x > 10 / 6) & (x <= 11 / 6))
                    | ((x > 13 / 6) & (x <= 14 / 6))
                    | ((x > 15 / 6) & (x <= 16 / 6))
                    | (x > 17 / 6)
                )
                shift = np.where((y <= 0) | ((y <= 1) & selected), 0.0, 1.0)
                boundaries = [y, y - 1, *[x - b / 6 for b in (7, 8, 9, 10, 11, 13, 14, 15, 16, 17)]]
            target = np.sin(6 * np.pi * f1 + np.pi)
            if i == 7:
                target *= 0.3 * f1 * f1 * np.cos(24 * np.pi * f1 + 4 * np.pi) + 0.6 * f1
            f2 = 1 - np.sqrt(f1) + (1 if i == 7 else 2) * (y - shift - target) ** 2
        return mark_boundary(np.column_stack((f1, f2)), *boundaries), None, None


class MMF1(_MMF):
    number = 1


class MMF2(_MMF):
    number = 2


class MMF3(_MMF):
    number = 3


class MMF4(_MMF):
    number = 4


class MMF5(_MMF):
    number = 5


class MMF6(_MMF):
    number = 6


class MMF7(_MMF):
    number = 7


class MMF8(_MMF):
    number = 8
