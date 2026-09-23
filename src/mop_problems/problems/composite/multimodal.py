"""Omni-test and the simple/rotated SYM-PART variants."""

import numpy as np

from ...core import Problem, integer


def _tile(coordinate, first_boundary, width):
    from ..._autodiff import Jet

    if isinstance(coordinate, Jet):
        # Clipped outer tiles are locally constant even at redundant ceil
        # breakpoints, and the central tile is smooth at coordinate=0.
        value = _tile(coordinate.v, first_boundary, width)
        at_jump = np.abs(np.abs(coordinate.v) - first_boundary) <= 32 * np.finfo(float).eps
        return Jet(value, np.zeros_like(coordinate.d), coordinate.r & ~at_jump)
    return np.clip(
        np.sign(coordinate) * np.ceil((np.abs(coordinate) - first_boundary) / width), -1, 1
    )


class OmniTest(Problem):
    """Periodic objectives on [0,6]^n, with 3^n equivalent Pareto subsets."""

    def __init__(self, n_var: int = 2):
        super().__init__("Omni-test", integer(n_var, "n_var", 2), 2, 0, 6)

    def _evaluate(self, X):
        return (
            np.column_stack((np.sin(np.pi * X).sum(axis=1), np.cos(np.pi * X).sum(axis=1))),
            None,
            None,
        )

    def pareto_front(self, n_points: int = 100, *, seed: int = 0):
        angle = np.linspace(0, np.pi / 2, integer(n_points, "n_points", 2))
        return -self.n_var * np.column_stack((np.sin(angle), np.cos(angle)))


class SYMPARTRotated(Problem):
    """Nine equivalent Pareto subsets; a=1, b=c=10, box [-100,100]^2.

    The angle is measured counterclockwise in radians in decision space. This
    selects the simple/rotated pymoo 0.6.2 convention, not SYM-PART3 distortion.
    """

    def __init__(self, angle: float = np.pi / 4):
        if not np.isscalar(angle) or not np.isfinite(angle):
            raise ValueError("angle must be a finite scalar")
        self.angle = float(angle)
        cosine, sine = np.cos(self.angle), np.sin(self.angle)
        self._inverse_rotation = np.array([[cosine, sine], [-sine, cosine]])
        super().__init__("SYM-PART-Rotated", 2, 2, -100, 100)

    def _evaluate(self, X):
        coordinates = X @ self._inverse_rotation.T
        horizontal, vertical = coordinates.T
        tile_x = _tile(horizontal, 6, 12)
        tile_y = _tile(vertical, 5, 10)
        u, v = horizontal - 10 * tile_x, vertical - 10 * tile_y
        return np.column_stack(((u + 1) ** 2 + v**2, (u - 1) ** 2 + v**2)), None, None

    def pareto_front(self, n_points: int = 100, *, seed: int = 0):
        u = np.linspace(-1, 1, integer(n_points, "n_points", 2))
        return np.column_stack(((u + 1) ** 2, (u - 1) ** 2))


class SYMPART(SYMPARTRotated):
    """Unrotated SYM-PART, using the same nine-subset convention."""

    def __init__(self):
        super().__init__(angle=0)
        self.name = "SYM-PART"
