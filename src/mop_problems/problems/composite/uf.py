"""CEC 2009 UF1--10 (CES-487), with one-based index sets from the report."""

import numpy as np

from ...core import Problem, integer


class _UF(Problem):
    number: int

    def __init__(self, n_var: int = 30):
        n_obj = 2 if self.number <= 7 else 3
        n_var = integer(n_var, "n_var", 3 if n_obj == 2 else 5)
        radius = 2 if self.number == 4 or n_obj == 3 else 1
        lower, upper = np.full(n_var, -float(radius)), np.full(n_var, float(radius))
        lower[: n_obj - 1], upper[: n_obj - 1] = 0, 1
        if self.number == 3:
            lower[:] = 0
        super().__init__(f"UF{self.number}", n_var, n_obj, lower, upper)

    def _evaluate(self, X):
        first = X[:, 0]
        if self.n_obj == 3:
            second = X[:, 1]
            j = np.arange(3, self.n_var + 1)
            y = X[:, 2:] - 2 * second[:, None] * np.sin(
                2 * np.pi * first[:, None] + np.pi * j / self.n_var
            )
            penalty = 4 * y**2 - np.cos(8 * np.pi * y) + 1 if self.number == 10 else y**2
            if self.number == 9:
                bump = np.maximum(0, 1.1 * (1 - 4 * (2 * first - 1) ** 2))
                front = np.column_stack(
                    (
                        (bump + 2 * first) * second / 2,
                        (bump - 2 * first + 2) * second / 2,
                        1 - second,
                    )
                )
            else:
                front = np.column_stack(
                    (
                        np.cos(np.pi * first / 2) * np.cos(np.pi * second / 2),
                        np.cos(np.pi * first / 2) * np.sin(np.pi * second / 2),
                        np.sin(np.pi * first / 2),
                    )
                )
            for i, remainder in enumerate((1, 2, 0)):
                front[:, i] += 2 * penalty[:, j % 3 == remainder].mean(axis=1)
            return front, None, None

        j = np.arange(2, self.n_var + 1)
        phase = 6 * np.pi * first[:, None] + j * np.pi / self.n_var
        if self.number == 2:
            amplitude = (
                0.3
                * first[:, None] ** 2
                * np.cos(24 * np.pi * first[:, None] + 4 * j * np.pi / self.n_var)
                + 0.6 * first[:, None]
            )
            link = amplitude * np.where(j % 2 == 1, np.cos(phase), np.sin(phase))
        elif self.number == 3:
            link = first[:, None] ** (0.5 * (1 + 3 * (j - 2) / (self.n_var - 2)))
        else:
            link = np.sin(phase)
        y = X[:, 1:] - link
        if self.number <= 3:
            front = np.column_stack((first, 1 - np.sqrt(first)))
        elif self.number == 4:
            front = np.column_stack((first, 1 - first**2))
        elif self.number == 7:
            front = np.column_stack((first**0.2, 1 - first**0.2))
        else:
            front = np.column_stack((first, 1 - first))
            bump = (
                0.15 * np.abs(np.sin(20 * np.pi * first))
                if self.number == 5
                else np.maximum(0, 0.7 * np.sin(4 * np.pi * first))
            )
            front += bump[:, None]
        for i, odd in enumerate((True, False)):
            mask = (j % 2 == 1) == odd
            residual = y[:, mask]
            if self.number in (3, 6):
                product = np.cos(20 * np.pi * residual / np.sqrt(j[mask])).prod(axis=1)
                cost = 4 * (residual**2).sum(axis=1) - 2 * product + 2
                front[:, i] += 2 * cost / mask.sum()
            else:
                if self.number == 4:
                    values = np.abs(residual) / (1 + np.exp(2 * np.abs(residual)))
                elif self.number == 5:
                    values = 2 * residual**2 - np.cos(4 * np.pi * residual) + 1
                else:
                    values = residual**2
                front[:, i] += 2 * values.mean(axis=1)
        return front, None, None

    def pareto_front(self, n_points: int = 100, *, seed: int = 0):
        if self.number in (5, 6, 9):
            return super().pareto_front(n_points, seed=seed)
        n_points = integer(n_points, "n_points", 2)
        if self.n_obj == 3:
            front = np.abs(np.random.default_rng(seed).normal(size=(n_points, 3)))
            front /= np.linalg.norm(front, axis=1, keepdims=True)
            front[: min(3, n_points)] = np.eye(3)[: min(3, n_points)]
            return front
        u = np.linspace(0, 1, n_points)
        second = 1 - np.sqrt(u) if self.number <= 3 else 1 - u**2 if self.number == 4 else 1 - u
        return np.column_stack((u, second))


class UF1(_UF):
    """Sinusoidal linkage with a convex front."""

    number = 1


class UF2(_UF):
    """Oscillating amplitude and parity-dependent linkage."""

    number = 2


class UF3(_UF):
    """Power linkage and a product-based multimodal penalty."""

    number = 3


class UF4(_UF):
    """Bounded nonquadratic penalty and a concave front."""

    number = 4


class UF5(_UF):
    """Discrete front with N=10 and epsilon=0.1."""

    number = 5


class UF6(_UF):
    """Disconnected front with N=2 and epsilon=0.1."""

    number = 6


class UF7(_UF):
    """Biased position parameter and a linear front."""

    number = 7


class UF8(_UF):
    """Three objectives with sinusoidal linkage and a spherical front."""

    number = 8


class UF9(_UF):
    """Three objectives and a disconnected front; epsilon=0.1."""

    number = 9


class UF10(_UF):
    """Three objectives with a multimodal residual penalty."""

    number = 10
