"""JY1--8 with explicit time, Jiang and Yang (2017).

Unit-box inputs use the original author's 2016 JY.h mapping z=2*x-1 for
distance variables. JY8 repairs the documented missing pow call in that file.
"""

import numpy as np

from ...core import Problem, integer


class _JY(Problem):
    number: int

    def __init__(self, n_var=10, *, time=0):
        if (
            isinstance(time, (bool, np.bool_))
            or not np.isscalar(time)
            or not np.isfinite(time)
            or time < 0
        ):
            raise ValueError("time must be a finite nonnegative scalar")
        self._time = float(time)
        super().__init__(f"JY{self.number}", integer(n_var, "n_var", 2), 2, 0, 1)

    @property
    def time(self):
        return self._time

    def at_time(self, time):
        return type(self)(n_var=self.n_var, time=time)

    def _evaluate(self, X):
        i, t = self.number, self.time
        shift = np.sin(np.pi * t / 2)
        a, w = 0.05, 6
        if i in (2, 3):
            w = np.floor(6 * np.sin(np.pi * (t - 1) / 2))
        elif i == 4:
            w = 10 ** (1 + abs(shift))
        elif i == 5:
            a, w = 0.3 * np.sin(np.pi * (t - 1) / 2), 1
        elif i in (6, 7):
            a, w = 0.1, 3
        pos, tail = X[:, 0], 2 * X[:, 1:] - 1
        z = tail - shift
        if i == 3:
            frequency = np.floor(100 * shift * shift)
            y = np.abs(pos * np.sin((2 * frequency + 1) * np.pi * pos))
            z = tail**2 - np.column_stack((y, tail[:, :-1]))
            g = (z * z).sum(axis=1)
        elif i in (5, 8):
            g = (tail * tail).sum(axis=1)
        elif i == 6:
            g = (4 * z * z - np.cos(2 * np.floor(10 * abs(shift)) * np.pi * z) + 1).sum(axis=1)
        elif i == 7:
            g = (z * z - 10 * np.cos(2 * np.pi * z) + 10).sum(axis=1)
        else:
            g = (z * z).sum(axis=1)
        ripple = a * np.sin(w * np.pi * pos)
        left, right = pos + ripple, 1 - pos + ripple
        if i == 7:
            power = 0.2 + 2.8 * abs(shift)
            left, right = left**power, right**power
        elif i == 8:
            beta = 10 - 9.8 * abs(shift)
            left, right = left ** (2 / beta), right**beta
        return (1 + g[:, None]) * np.column_stack((left, right)), None, None


class JY1(_JY):
    number = 1


class JY2(_JY):
    number = 2


class JY3(_JY):
    number = 3


class JY4(_JY):
    number = 4


class JY5(_JY):
    number = 5


class JY6(_JY):
    number = 6


class JY7(_JY):
    number = 7


class JY8(_JY):
    number = 8
