"""Deb et al. (2001) CTP1-8, with raw paper residuals (no division by sine)."""

import numpy as np

from ...core import Problem, integer
from .._utils import rastrigin

# theta/pi, amplitude, frequency, inner power, outer power, displacement.
CTP_PARAMETERS = {
    2: (-0.2, 0.2, 10, 1, 6, 1),
    3: (-0.2, 0.1, 10, 1, 0.5, 1),
    4: (-0.2, 0.75, 10, 1, 0.5, 1),
    5: (-0.2, 0.1, 10, 2, 0.5, 1),
    6: (0.1, 40, 0.5, 1, 2, -2),
    7: (-0.05, 40, 5, 1, 6, 0),
}


def rotated_constraint(f1, f2, parameters):
    angle, a, b, c, d, e = parameters
    angle *= np.pi
    longitudinal = (f2 - e) * np.cos(angle) - f1 * np.sin(angle)
    transverse = (f2 - e) * np.sin(angle) + f1 * np.cos(angle)
    return a * np.abs(np.sin(b * np.pi * transverse**c)) ** d - longitudinal


class _CTP(Problem):
    number: int

    def __init__(self, n_var=2, distance="linear", n_constraints=None):
        n = integer(n_var, "n_var", 2)
        if distance not in ("linear", "multimodal"):
            raise ValueError("distance must be 'linear' or 'multimodal'")
        self.distance = distance
        count = 2 if self.number in (1, 8) else 1
        if n_constraints is not None:
            if self.number != 1:
                raise ValueError("n_constraints is configurable only for CTP1")
            count = integer(n_constraints, "n_constraints")
        lo, hi = np.zeros(n), np.ones(n)
        if distance == "multimodal":
            lo[1:], hi[1:] = -5.12, 5.12
        if self.number in (6, 8):
            hi[1:] = 20
        super().__init__(f"CTP{self.number}", n, 2, lo, hi, count)
        if self.number == 1:
            a, b = 1.0, 1.0
            self._ab = []
            for j in range(1, count + 1):
                x = j / (count + 1)
                ordinate = a * np.exp(-b * x)
                a = (a + ordinate) / 2
                b = -np.log(ordinate / a) / x
                self._ab.append((a, b))

    def _evaluate(self, X):
        a = X[:, 0]
        g = 1 + (np.sum(X[:, 1:], axis=1) if self.distance == "linear" else rastrigin(X[:, 1:]))
        b = g * np.exp(-a / g) if self.number == 1 else g - np.sqrt(a * g)
        if self.number == 1:
            constraints = [amplitude * np.exp(-decay * a) - b for amplitude, decay in self._ab]
        else:
            params = (
                [CTP_PARAMETERS[self.number]]
                if self.number != 8
                else [CTP_PARAMETERS[6], (-0.05, 40, 2, 1, 6, 0)]
            )
            constraints = [rotated_constraint(a, b, p) for p in params]
        return np.column_stack((a, b)), np.column_stack(constraints), None


class CTP1(_CTP):
    number = 1


class CTP2(_CTP):
    number = 2


class CTP3(_CTP):
    number = 3


class CTP4(_CTP):
    number = 4


class CTP5(_CTP):
    number = 5


class CTP6(_CTP):
    number = 6


class CTP7(_CTP):
    number = 7


class CTP8(_CTP):
    number = 8
