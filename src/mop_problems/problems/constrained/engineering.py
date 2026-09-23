"""Continuous engineering benchmark variants with fully specified conventions."""

import numpy as np

from ...core import Problem


class WeldedBeam(Problem):
    """Four-variable, two-objective, four-constraint pymoo 0.6.2 variant.

    Variables are (h, length, t, b). The benchmark uses the published software's
    J coefficient sqrt(2), not the 2*sqrt(2) found in other welded-beam variants.
    See docs/formulas.md before comparing results across engineering libraries.
    """

    def __init__(self):
        super().__init__(
            "Welded-Beam", 4, 2, [0.125, 0.1, 0.1, 0.125], [5, 10, 10, 5], n_ieq_constr=4
        )

    def _evaluate(self, X):
        h, length, t, b = X.T
        cost = 1.10471 * h**2 * length + 0.04811 * t * b * (14 + length)
        deflection = 2.1952 / (b * t**3)
        radius = np.sqrt(length**2 + (h + t) ** 2) / 2
        moment = 6000 * (14 + length / 2)
        polar = np.sqrt(2) * h * length * (length**2 / 12 + (h + t) ** 2 / 4)
        primary = 6000 / (np.sqrt(2) * h * length)
        secondary = moment * radius / polar
        shear = np.sqrt(primary**2 + secondary**2 + primary * secondary * length / radius)
        normal = 504000 / (b * t**2)
        buckling = 64746.022 * (1 - 0.0282346 * t) * t * b**3
        G = np.column_stack((shear - 13600, normal - 30000, h - b, 6000 - buckling))
        return np.column_stack((cost, deflection)), G, None
