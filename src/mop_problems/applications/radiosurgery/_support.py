"""Simplex operations and lazy optional COPT environment creation."""

import numpy as np


def simplex(v):
    if len(v) == 2:
        t = np.clip((v[0] - v[1] + 1) / 2, 0, 1)
        return np.array([t, 1 - t])
    u = np.sort(v)[::-1]
    t = (np.cumsum(u) - 1) / np.arange(1, len(v) + 1)
    k = np.flatnonzero(u > t)[-1]
    return np.maximum(v - t[k], 0)


def diagonal_simplex(v, diagonal):
    """Exact diagonal-metric simplex projection, including zero coordinates."""
    order = np.argsort(-(diagonal * v))
    sv = si = 0.0
    level = 0.0
    for k, j in enumerate(order):
        sv += v[j]
        si += 1 / diagonal[j]
        level = (sv - 1) / si
        if k == len(v) - 1 or level >= diagonal[order[k + 1]] * v[order[k + 1]]:
            break
    w = np.maximum(v - level / diagonal, 0.0)
    # Only remove floating point mass error on an already positive coordinate.
    w[np.argmax(w)] += 1 - w.sum()
    return w


def create_environment():
    try:
        import coptpy as cp
    except ImportError as error:
        raise RuntimeError(
            'Install the optional radiosurgery dependencies with pip install -e ".[radiosurgery]" '
            "and configure a COPT 8 license to run the solvers."
        ) from error
    try:
        config = cp.EnvrConfig()
        config.set("nobanner", "1")
        return cp.Envr(config)
    except cp.CoptError as error:
        raise RuntimeError(f"COPT environment/license initialization failed: {error}") from error
