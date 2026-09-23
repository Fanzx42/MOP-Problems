"""ZXH-CF1-16, Zhou, Xu and Huang (2021); distinct from CEC 2009 CF."""

import numpy as np

from ...core import Problem, integer


class _ZXHCF(Problem):
    number: int

    def __init__(self, n_var=None, n_obj=3, k=None):
        i = self.number
        m = integer(n_obj, "n_obj", 2)
        n = m + 10 if n_var is None else integer(n_var, "n_var", m + 2)
        default_k = m - 1 if m <= 3 else m // 2 if m <= 8 else 3
        self.k = default_k if k is None else integer(k, "k", 1)
        if self.k >= m:
            raise ValueError("k must be less than n_obj")
        if i <= 4 and k is not None:
            raise ValueError("k only applies to ZXH_CF5-16")
        count = (1 if i in (1, 5, 9, 13) else 2) + (self.k if i >= 5 else 0)
        super().__init__(f"ZXH_CF{i}", n, m, 1e-10, 1 - 1e-10, count)

    def _evaluate(self, X):
        i, m = self.number, self.n_obj
        squared = np.column_stack([np.sum(X[:, j:m] ** 2, axis=1) for j in range(m)])
        theta = 2 / np.pi * np.arctan2(np.sqrt(squared[:, 1:]), X[:, : m - 1])
        z = X[:, m:] - 0.2
        if i in (1, 6, 11, 16):
            h = np.sum(z * z, axis=1)
        elif i in (2, 5, 12, 15):
            h = np.sum(100 * (z[:, :-1] ** 2 - z[:, 1:]) ** 2 + z[:, :-1] ** 2, axis=1)
        elif i in (3, 8, 9, 14):
            h = (
                20
                - 20 * np.exp(-0.2 * np.sqrt(np.mean(z * z, axis=1)))
                + np.e
                - np.exp(np.mean(np.cos(2 * np.pi * z), axis=1))
            )
        else:
            h = 5 * (
                np.sum(z * z, axis=1)
                - np.prod(np.cos(10 * np.pi * z / np.sqrt(np.arange(1, z.shape[1] + 1))), axis=1)
                + 1
            )
        if i in (1, 7, 12, 14):
            shape = np.column_stack(
                [
                    np.prod(theta[:, :j], axis=1) * (1 - theta[:, j] if j < m - 1 else 1)
                    for j in range(m)
                ]
            )
        else:
            s, c = np.sin(np.pi * theta / 2), np.cos(np.pi * theta / 2)
            shape = np.column_stack(
                [np.prod(s[:, :j], axis=1) * (c[:, j] if j < m - 1 else 1) for j in range(m)]
            )
            if i not in (2, 8, 11, 13):
                shape = 1 - shape
            if i in (4, 6, 9, 15):
                shape[:, 0] = theta[:, 0] - np.cos(4 * np.pi * theta[:, 0] + np.pi / 2) / (
                    4 * np.pi
                )
        F = (1 + (1 - squared[:, 0]) ** 2 + h)[:, None] * shape
        radius = squared[:, 0] + h
        G = [radius - 1]
        if i not in (1, 5, 9, 13):
            lower = {
                2: 0.25,
                3: 0.5,
                4: 0.75,
                6: 0.25,
                7: 0.5,
                8: 0.75,
                10: 0.25,
                11: 0.5,
                12: 0.75,
                14: 0.25,
                15: 0.25,
                16: 0.25,
            }[i]
            G.append(lower - radius)
        if i >= 5:
            t = theta[:, : self.k]
            if i <= 8:
                angular = np.maximum(0.25 - t, t - 0.75)
            elif i <= 12:
                angular = np.minimum(t - 0.25, 0.75 - t)
            else:
                angular = np.minimum(np.minimum(t - 0.1, 0.8 - t), np.maximum(0.4 - t, t - 0.7))
            G.append(angular)
        return F, np.column_stack(G), None


class ZXH_CF1(_ZXHCF):
    number = 1


class ZXH_CF2(_ZXHCF):
    number = 2


class ZXH_CF3(_ZXHCF):
    number = 3


class ZXH_CF4(_ZXHCF):
    number = 4


class ZXH_CF5(_ZXHCF):
    number = 5


class ZXH_CF6(_ZXHCF):
    number = 6


class ZXH_CF7(_ZXHCF):
    number = 7


class ZXH_CF8(_ZXHCF):
    number = 8


class ZXH_CF9(_ZXHCF):
    number = 9


class ZXH_CF10(_ZXHCF):
    number = 10


class ZXH_CF11(_ZXHCF):
    number = 11


class ZXH_CF12(_ZXHCF):
    number = 12


class ZXH_CF13(_ZXHCF):
    number = 13


class ZXH_CF14(_ZXHCF):
    number = 14


class ZXH_CF15(_ZXHCF):
    number = 15


class ZXH_CF16(_ZXHCF):
    number = 16
