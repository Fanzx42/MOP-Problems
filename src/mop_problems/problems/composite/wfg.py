"""WFG1--9, using the transformations in Huband et al. (2006).

The implementation evaluates normalized decision variables through separate bias,
shift, reduction and shape stages. Only roundoff near zero or one is corrected.
"""

import numpy as np

from ...core import Problem, integer


def _unit(y):
    y = np.where((y < 0) & (y >= -1e-10), 0.0, y)
    return np.where((y > 1) & (y <= 1 + 1e-10), 1.0, y)


def _linear(y):
    return _unit(np.abs(y - 0.35) / np.abs(np.floor(0.35 - y) + 0.35))


def _flat(y):
    left = np.minimum(0, np.floor(y - 0.75)) * 0.8 * (0.75 - y) / 0.75
    right = np.minimum(0, np.floor(0.85 - y)) * 0.2 * (y - 0.85) / 0.15
    return _unit(0.8 + left - right)


def _deceptive(y):
    a, b, c = 0.35, 0.001, 0.05
    left = np.floor(y - a + b) * (1 - c + (a - b) / b) / (a - b)
    right = np.floor(a + b - y) * (1 - c + (1 - a - b) / b) / (1 - a - b)
    return _unit(1 + (np.abs(y - a) - b) * (left + right + 1 / b))


def _multimodal(y, ruggedness):
    q = np.abs(y - 0.35) / (2 * (np.floor(0.35 - y) + 0.35))
    return _unit((1 + np.cos(122 * np.pi * (0.5 - q)) + 4 * ruggedness * q**2) / (ruggedness + 2))


def _parameter_bias(y, u):
    a = 0.98 / 49.98
    exponent = 0.02 + 49.98 * (a - (1 - 2 * u) * np.abs(np.floor(0.5 - u) + a))
    return _unit(y**exponent)


def _nonseparable(y, degree):
    width = y.shape[1]
    numerator = y.sum(axis=1)
    for offset in range(1, degree):
        numerator += np.abs(y - np.roll(y, -offset, axis=1)).sum(axis=1)
    half = np.ceil(degree / 2)
    denominator = width * half * (1 + 2 * degree - 2 * half) / degree
    return _unit(numerator / denominator)


def _shape(x, kind):
    n_obj = x.shape[1] + 1
    columns = []
    for index in range(n_obj):
        prefix = n_obj - index - 1
        if kind == "concave":
            value = np.sin(np.pi * x[:, :prefix] / 2).prod(axis=1)
            if index:
                value *= np.cos(np.pi * x[:, prefix] / 2)
        elif kind == "convex":
            value = (1 - np.cos(np.pi * x[:, :prefix] / 2)).prod(axis=1)
            if index:
                value *= 1 - np.sin(np.pi * x[:, prefix] / 2)
        else:
            value = x[:, :prefix].prod(axis=1)
            if index:
                value *= 1 - x[:, prefix]
        columns.append(value)
    return _unit(np.column_stack(columns))


class _WFG(Problem):
    number: int

    def __init__(self, n_var: int | None = None, n_obj: int = 3, k: int | None = None):
        n_obj = integer(n_obj, "n_obj", 2)
        k = (4 if n_obj == 2 else 2 * (n_obj - 1)) if k is None else integer(k, "k")
        if k % (n_obj - 1):
            raise ValueError("k must be divisible by n_obj - 1")
        n_var = k + 20 if n_var is None else integer(n_var, "n_var", k + 1)
        if self.number in (2, 3) and (n_var - k) % 2:
            raise ValueError("WFG2 and WFG3 require an even number of distance variables")
        self.k = k
        self.l = n_var - k
        super().__init__(f"WFG{self.number}", n_var, n_obj, 0, 2 * np.arange(1, n_var + 1))

    def _reduce(self, y, *, weighted=False, nonseparable=False):
        width = self.k // (self.n_obj - 1)
        groups = [(i * width, (i + 1) * width) for i in range(self.n_obj - 1)]
        groups.append((self.k, y.shape[1]))
        columns = []
        for start, stop in groups:
            block = y[:, start:stop]
            if nonseparable:
                reduced = _nonseparable(block, stop - start)
            elif weighted:
                weights = np.arange(start + 1, stop + 1, dtype=float)
                reduced = block @ weights / weights.sum()
            else:
                reduced = block.mean(axis=1)
            columns.append(_unit(reduced))
        return np.column_stack(columns)

    def _transform(self, X):
        y = X / self.xu
        number, k = self.number, self.k
        if number in (7, 8, 9):
            biased = y.copy()
            indices = range(k) if number == 7 else range(k, self.n_var)
            if number == 9:
                indices = range(self.n_var - 1)
            for i in indices:
                context = y[:, :i] if number == 8 else y[:, i + 1 :]
                biased[:, i] = _parameter_bias(y[:, i], context.mean(axis=1))
            y = biased
        if number in (1, 2, 3, 6, 7, 8):
            y[:, k:] = _linear(y[:, k:])
        if number == 1:
            y[:, k:] = _flat(y[:, k:])
            reduced = self._reduce(_unit(y**0.02), weighted=True)
        elif number in (2, 3):
            pairs = [_nonseparable(y[:, i : i + 2], 2) for i in range(k, self.n_var, 2)]
            reduced = self._reduce(np.column_stack([y[:, :k], *pairs]))
        elif number == 4:
            reduced = self._reduce(_multimodal(y, 10))
        elif number == 5:
            reduced = self._reduce(_deceptive(y))
        elif number == 9:
            y[:, :k] = _deceptive(y[:, :k])
            y[:, k:] = _multimodal(y[:, k:], 95)
            reduced = self._reduce(y, nonseparable=True)
        else:
            reduced = self._reduce(y, nonseparable=number == 6)
        distance = reduced[:, -1:]
        degeneracy = np.ones(self.n_obj - 1)
        if number == 3:
            degeneracy[1:] = 0
        positions = np.maximum(distance, degeneracy) * (reduced[:, :-1] - 0.5) + 0.5
        return positions, distance

    def _evaluate(self, X):
        positions, distance = self._transform(X)
        number = self.number
        kind = "convex" if number in (1, 2) else "linear" if number == 3 else "concave"
        h = _shape(positions, kind)
        first = positions[:, 0]
        if number == 1:
            h[:, -1] = _unit(1 - first - np.cos(10 * np.pi * first + np.pi / 2) / (10 * np.pi))
        elif number == 2:
            h[:, -1] = _unit(1 - first * np.cos(5 * np.pi * first) ** 2)
        return distance + 2 * np.arange(1, self.n_obj + 1) * h, None, None


class WFG1(_WFG):
    """Polynomial and flat biases with a mixed convex/concave front."""

    number = 1


class WFG2(_WFG):
    """Pairwise nonseparability and a disconnected front."""

    number = 2


class WFG3(_WFG):
    """Pairwise nonseparability and a degenerate linear front."""

    number = 3


class WFG4(_WFG):
    """Multimodal transformation and a concave front."""

    number = 4


class WFG5(_WFG):
    """Deceptive transformation and a concave front."""

    number = 5


class WFG6(_WFG):
    """Nonseparable reductions and a concave front."""

    number = 6


class WFG7(_WFG):
    """Position parameters biased by their suffixes."""

    number = 7


class WFG8(_WFG):
    """Distance parameters biased by their original prefixes."""

    number = 8


class WFG9(_WFG):
    """Parameter dependence, deception, multimodality and nonseparability."""

    number = 9
