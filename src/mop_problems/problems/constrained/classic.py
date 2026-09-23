"""Classic constrained problems, returning unscaled residuals with G <= 0."""

import numpy as np

from mop_problems.core import Problem


class BNH(Problem):
    """Binh--Korn; two objectives and two circular inequality constraints."""

    def __init__(self):
        super().__init__("BNH", 2, 2, [0, 0], [5, 3], n_ieq_constr=2)

    def _evaluate(self, X):
        x, y = X.T
        F = np.column_stack((4 * (x**2 + y**2), (x - 5) ** 2 + (y - 5) ** 2))
        G = np.column_stack(((x - 5) ** 2 + y**2 - 25, 7.7 - (x - 8) ** 2 - (y + 3) ** 2))
        return F, G, None


class TNK(Problem):
    """Tanaka problem; atan2(x1, x2) avoids division at the box boundary.

    At the origin, which is infeasible, the angle is conventionally set to zero.
    The second residual is not multiplied by two (unlike pymoo).
    """

    def __init__(self):
        super().__init__("TNK", 2, 2, 0.0, np.pi, n_ieq_constr=2)

    def _evaluate(self, X):
        x, y = X.T
        theta = np.arctan2(x, y)
        G = np.column_stack(
            (
                1 + 0.1 * np.cos(16 * theta) - x**2 - y**2,
                (x - 0.5) ** 2 + (y - 0.5) ** 2 - 0.5,
            )
        )
        return X.copy(), G, None


class OSY(Problem):
    """Osyczka--Kundu; six variables, six inequalities, unscaled residuals."""

    def __init__(self):
        super().__init__("OSY", 6, 2, [0, 0, 1, 0, 1, 0], [10, 10, 5, 6, 5, 10], 6)

    def _evaluate(self, X):
        a, b, c, d, e, f = X.T
        first = -(25 * (a - 2) ** 2 + (b - 2) ** 2 + (c - 1) ** 2 + (d - 4) ** 2 + (e - 1) ** 2)
        F = np.column_stack((first, np.sum(X**2, axis=1)))
        G = np.column_stack(
            (
                2 - a - b,
                a + b - 6,
                b - a - 2,
                a - 3 * b - 2,
                (c - 3) ** 2 + d - 4,
                4 - (e - 3) ** 2 - f,
            )
        )
        return F, G, None


class SRN(Problem):
    """Srinivas--Deb; two variables and two inequality constraints."""

    def __init__(self):
        super().__init__("SRN", 2, 2, -20.0, 20.0, n_ieq_constr=2)

    def _evaluate(self, X):
        x, y = X.T
        F = np.column_stack((2 + (x - 2) ** 2 + (y - 1) ** 2, 9 * x - (y - 1) ** 2))
        G = np.column_stack((x**2 + y**2 - 225, x - 3 * y + 10))
        return F, G, None
