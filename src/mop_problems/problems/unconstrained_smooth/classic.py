"""Schaffer N.1, Fonseca--Fleming and Poloni minimization benchmarks."""

import numpy as np

from mop_problems.core import FloatArray, Problem, integer


class SCH(Problem):
    """Schaffer N.1 with the conventional [-1000, 1000] decision interval."""

    def __init__(self):
        super().__init__("SCH", 1, 2, -1000.0, 1000.0)

    def _evaluate(self, X):
        return np.column_stack((X[:, 0] ** 2, (X[:, 0] - 2) ** 2)), None, None

    def pareto_front(self, n_points: int = 100, *, seed: int = 0) -> FloatArray:
        x = np.linspace(0, 2, integer(n_points, "n_points", 2))
        return self.evaluate(x[:, None]).F.copy()


class FON(Problem):
    """Fonseca--Fleming, three decision variables by default."""

    def __init__(self, n_var: int = 3):
        n_var = integer(n_var, "n_var", 2)
        super().__init__("FON", n_var, 2, -4.0, 4.0)

    def _evaluate(self, X):
        offset = 1 / np.sqrt(self.n_var)
        f1 = -np.expm1(-np.sum((X - offset) ** 2, axis=1))
        f2 = -np.expm1(-np.sum((X + offset) ** 2, axis=1))
        return np.column_stack((f1, f2)), None, None

    def pareto_front(self, n_points: int = 100, *, seed: int = 0) -> FloatArray:
        limit = 1 / np.sqrt(self.n_var)
        x = np.linspace(-limit, limit, integer(n_points, "n_points", 2))
        return self.evaluate(np.repeat(x[:, None], self.n_var, axis=1)).F.copy()


class POL(Problem):
    """Poloni's two-variable problem, with both objectives minimized."""

    def __init__(self):
        super().__init__("POL", 2, 2, -np.pi, np.pi)

    def _evaluate(self, X):
        x, y = X.T
        a1 = 0.5 * np.sin(1) - 2 * np.cos(1) + np.sin(2) - 1.5 * np.cos(2)
        a2 = 1.5 * np.sin(1) - np.cos(1) + 2 * np.sin(2) - 0.5 * np.cos(2)
        b1 = 0.5 * np.sin(x) - 2 * np.cos(x) + np.sin(y) - 1.5 * np.cos(y)
        b2 = 1.5 * np.sin(x) - np.cos(x) + 2 * np.sin(y) - 0.5 * np.cos(y)
        F = np.column_stack((1 + (a1 - b1) ** 2 + (a2 - b2) ** 2, (x + 3) ** 2 + (y + 1) ** 2))
        return F, None, None
