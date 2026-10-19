"""Fixed-geometry synthetic Gamma Knife application (optional SciPy/COPT).

This application does not change the standard 343-problem benchmark registry.
The dose maps are synthetic and are not patient data.
"""

from .model import Case, initial, make_case, weight_grid

__all__ = ["Case", "initial", "make_case", "weight_grid"]
