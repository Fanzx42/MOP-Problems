"""ZCAT1--20 (Zapotecas-Martinez et al., 2023), levels 1--6.

Equations and indexing follow the jMetalPy reference revision in the catalogue.
In particular, the g-function angle denominator is n minus the Pareto-set dimension.
"""

import numpy as np

from ...core import Problem, integer
from .._utils import mark_boundary


def _degenerate(y, number):
    t = y[:, 0]
    if number == 19:
        return (t <= 0.2) | ((t >= 0.4) & (t <= 0.6))
    return ((t >= 0.1) & (t <= 0.4)) | ((t >= 0.6) & (t <= 0.9))


def _shape(y, m, number):
    p, t = y[:, : m - 1], y[:, 0]
    s, c = np.sin(p * np.pi / 2), np.cos(p * np.pi / 2)
    one = t * 0 + 1
    f = []
    if number in (1, 2, 8, 12):
        for i in range(m - 1):
            k = m - 1 - i
            if number == 1:
                v = np.prod(s[:, :k], axis=1) * (c[:, k] if i else one)
            elif number == 2:
                v = np.prod(1 - c[:, :k], axis=1) * (1 - s[:, k] if i else one)
            elif number == 8:
                v = 1 - np.prod(1 - s[:, :k], axis=1) * (1 - c[:, k] if i else one)
            else:
                v = 1 - np.prod(1 - p[:, :k], axis=1) * (p[:, k] if i else one)
            f.append(v)
    elif number in (3, 9, 11, 13):
        for i in range(m - 1):
            k = m - 1 - i
            v = (
                np.sum(s[:, :k] if number in (9, 13) else p[:, :k], axis=1)
                + ((c[:, k] if number in (9, 13) else 1 - p[:, k]) if i else 0)
            ) / (k + bool(i))
            f.append(1 - v if number == 13 else v)
    elif number in (14, 15, 16):
        for i in range(m - 1):
            if number == 15:
                v = t ** (1 + i / (4 * m))
            elif i == m - 2 and m > 2:
                v = 0.5 * (1 + np.sin((6 if number == 14 else 10) * t * np.pi / 2 - np.pi / 2))
            else:
                v = s[:, 0] ** ((2 if number == 14 else 1) + i / max(1, m - 2))
            f.append(v)
    else:
        q = p
        if number in (17, 18):
            flag = (
                np.all(p <= 0.5, axis=1)
                if number == 17
                else (np.all(p <= 0.4, axis=1) | np.all(p >= 0.6, axis=1))
            )
            q = np.where(flag[:, None], t[:, None], p)
        elif number in (19, 20):
            q = np.where(_degenerate(y, number)[:, None], t[:, None], p)
        f = [q[:, j] for j in range(m - 1)]
    if number in (1, 2):
        last = 1 - s[:, 0]
    elif number in (8, 9, 14):
        last = c[:, 0]
    elif number in (3,):
        last = 1 - t
    elif number == 4:
        last = 1 - p.mean(axis=1)
    elif number in (5, 17):
        last = np.expm1(8 * (1 - (q if number == 17 else p).mean(axis=1))) / np.expm1(8)
    elif number == 6:
        mu = p.mean(axis=1)
        last = (1 / (1 + np.exp(80 * mu - 40)) - 0.05 * mu - 1 / (1 + np.exp(40)) + 0.05) / (
            1 / (1 + np.exp(-40)) - 1 / (1 + np.exp(40)) + 0.05
        )
    elif number in (7, 18, 20):
        last = ((0.5 - (q if number in (18, 20) else p)) ** 5).mean(axis=1) / (2 * 0.5**5) + 0.5
    elif number == 10:
        last = (50 - 1 / (1 - p.mean(axis=1) + 0.02)) / (50 - 1 / 1.02)
    elif number == 19:
        mu = q.mean(axis=1)
        last = 1 - mu - np.cos(10 * np.pi * mu + np.pi / 2) / (10 * np.pi)
    else:
        k = 4 if number == 11 else 5 if number == 16 else 3
        last = (np.cos((2 * k - 1) * t * np.pi) + 2 * t + 4 * k * (1 - t) - 1) / (4 * k)
        if number == 13:
            last = 1 - last
    result = np.column_stack([*f, last])
    if number >= 17:
        bounds = {17: (0.5,), 18: (0.4, 0.6), 19: (0.2, 0.4, 0.6), 20: (0.1, 0.4, 0.6, 0.9)}[number]
        switches = [p[:, j] - b for j in range(m - 1 if number < 19 else 1) for b in bounds]
        result = mark_boundary(result, *switches)
    return result


def _g(p, width, kind):
    theta = 2 * np.pi * np.arange(width) / width
    y, angle = p[:, :, None], theta[None, None, :]
    mu = p.mean(axis=1)[:, None]
    if kind == 0:
        return np.zeros((p.shape[0], width)) + 0.221
    if kind == 1:
        return np.sin(1.5 * np.pi * y + angle).mean(axis=1) / 2 + 0.5
    if kind == 2:
        return (y**2 * np.sin(4.5 * np.pi * y + angle)).mean(axis=1) / 2 + 0.5
    if kind == 3:
        return (np.cos(np.pi * y + angle) ** 2).mean(axis=1)
    if kind == 4:
        return mu * np.cos(4 * np.pi * mu + theta) / 2 + 0.5
    if kind == 5:
        return (np.sin(2 * np.pi * y - 1 + angle) ** 3).mean(axis=1) / 2 + 0.5
    if kind == 6:
        return (
            -10 * np.exp(-0.4 * np.sqrt((p * p).mean(axis=1)))[:, None]
            - np.exp((np.cos(11 * np.pi * y + angle) ** 3).mean(axis=1))
            + 10
            + np.e
        ) / (-10 * np.exp(-0.4) - np.exp(-1) + 10 + np.e)
    if kind == 7:
        return (mu + np.exp(np.sin(7 * np.pi * mu - np.pi / 2 + theta)) - np.exp(-1)) / (
            1 + np.e - np.exp(-1)
        )
    if kind == 8:
        return np.abs(np.sin(2.5 * np.pi * (y - 0.5) + angle)).mean(axis=1)
    if kind == 9:
        return mu / 2 - np.abs(np.sin(2.5 * np.pi * y - np.pi / 2 + angle)).mean(axis=1) / 2 + 0.5
    return np.sin((4 * y - 2) * np.pi + angle).mean(axis=1) ** 3 / 2 + 0.5


def _z(w, level):
    if level == 1:
        return 10 * (w * w).mean(axis=1)
    if level == 2:
        return 10 * np.abs(w).max(axis=1)
    z3 = 10 * (w * w - np.cos(9 * np.pi * w) + 1).mean(axis=1) / 3
    if level == 3:
        return z3
    if level == 5:
        return -0.7 * z3 + 10 * (np.abs(w) ** 0.002).mean(axis=1)
    z4 = (
        10
        * (
            np.exp(np.sqrt(np.abs(w).max(axis=1)))
            - np.exp((0.5 * (np.cos(9 * np.pi * w) + 1)).mean(axis=1))
            - 1
            + np.e
        )
        / (2 * np.e - 2)
    )
    if level == 4:
        return z4
    return -0.7 * z4 + 10 * np.abs(w).mean(axis=1) ** 0.002


class _ZCAT(Problem):
    number: int

    def __init__(
        self,
        n_var=None,
        n_obj=3,
        *,
        complicated_pareto_set=False,
        level=1,
        bias=False,
        imbalance=False,
    ):
        m = integer(n_obj, "n_obj", 2)
        n = 2 * m if n_var is None else integer(n_var, "n_var", m)
        level = integer(level, "level", 1)
        if level > 6:
            raise ValueError("level must be in 1..6")
        for name, value in [
            ("complicated_pareto_set", complicated_pareto_set),
            ("bias", bias),
            ("imbalance", imbalance),
        ]:
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean")
        self.complicated_pareto_set, self.level = bool(complicated_pareto_set), level
        self.bias, self.imbalance = bool(bias), bool(imbalance)
        super().__init__(
            f"ZCAT{self.number}", n, m, -np.arange(1, n + 1) / 2, np.arange(1, n + 1) / 2
        )

    def _distance(self, y, dimension):
        kind = (
            (4, 5, 2, 7, 9, 4, 5, 2, 7, 9, 3, 10, 1, 6, 8, 10, 1, 8, 6, 3)[self.number - 1]
            if self.complicated_pareto_set
            else 0
        )
        w = y[:, dimension:] - _g(y[:, :dimension], self.n_var - dimension, kind)
        if self.bias:
            w = np.abs(w) ** 0.05
        result = []
        for j in range(self.n_obj):
            block = w[:, j :: self.n_obj] if j < w.shape[1] else w[:, :1]
            level = (1 if j % 2 == 0 else 4) if self.imbalance else self.level
            result.append(_z(block, level))
        return np.column_stack(result)

    def _evaluate(self, X):
        y = (X - self.xl) / (self.xu - self.xl)
        dim = 1 if self.number in (14, 15, 16) else self.n_obj - 1
        b = self._distance(y, dim)
        if self.number in (19, 20):
            b = np.where(_degenerate(y, self.number)[:, None], self._distance(y, 1), b)
        return (
            (_shape(y, self.n_obj, self.number) + b) * np.arange(1, self.n_obj + 1) ** 2,
            None,
            None,
        )


class ZCAT1(_ZCAT):
    number = 1


class ZCAT2(_ZCAT):
    number = 2


class ZCAT3(_ZCAT):
    number = 3


class ZCAT4(_ZCAT):
    number = 4


class ZCAT5(_ZCAT):
    number = 5


class ZCAT6(_ZCAT):
    number = 6


class ZCAT7(_ZCAT):
    number = 7


class ZCAT8(_ZCAT):
    number = 8


class ZCAT9(_ZCAT):
    number = 9


class ZCAT10(_ZCAT):
    number = 10


class ZCAT11(_ZCAT):
    number = 11


class ZCAT12(_ZCAT):
    number = 12


class ZCAT13(_ZCAT):
    number = 13


class ZCAT14(_ZCAT):
    number = 14


class ZCAT15(_ZCAT):
    number = 15


class ZCAT16(_ZCAT):
    number = 16


class ZCAT17(_ZCAT):
    number = 17


class ZCAT18(_ZCAT):
    number = 18


class ZCAT19(_ZCAT):
    number = 19


class ZCAT20(_ZCAT):
    number = 20
