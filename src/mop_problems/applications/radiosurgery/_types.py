"""Local convex models for the radiosurgery application; not registry Problem types."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Term:
    kind: str
    matrix: np.ndarray
    offset: np.ndarray
    coefficient: float = 1.0
    shape: tuple | None = None

    def value(self, u):
        v = self.matrix @ u - self.offset
        if self.kind == "quadratic":
            value = 0.5 * (v @ v)
        elif self.kind == "linear":
            value = v.sum()
        elif self.kind == "l1":
            value = np.abs(v).sum()
        elif self.kind == "hinge":
            value = np.maximum(v, 0).sum()
        elif self.kind == "squared_hinge":
            value = 0.5 * (np.maximum(v, 0) @ np.maximum(v, 0))
        elif self.kind == "nuclear":
            value = np.linalg.svd(v.reshape(self.shape), compute_uv=False).sum()
        else:
            raise ValueError(self.kind)
        return float(self.coefficient * value)

    def gradient(self, u):
        v = self.matrix @ u - self.offset
        if self.kind == "quadratic":
            derivative = v
        elif self.kind == "linear":
            derivative = np.ones_like(v)
        elif self.kind == "squared_hinge":
            derivative = np.maximum(v, 0.0)
        else:
            raise ValueError("Nonsmooth term has no single gradient")
        return self.coefficient * (self.matrix.T @ derivative)


@dataclass
class Domain:
    lower: np.ndarray
    upper: np.ndarray
    equalities: np.ndarray
    rhs: np.ndarray
    inequalities: np.ndarray
    limits: np.ndarray

    @classmethod
    def box(cls, n, lower=-np.inf, upper=np.inf):
        return cls(
            np.broadcast_to(lower, (n,)).copy(),
            np.broadcast_to(upper, (n,)).copy(),
            np.zeros((0, n)),
            np.zeros(0),
            np.zeros((0, n)),
            np.zeros(0),
        )

    def violation(self, u):
        parts = [
            np.maximum(self.lower - u, 0),
            np.maximum(u - self.upper, 0),
            np.abs(self.equalities @ u - self.rhs),
            np.maximum(self.inequalities @ u - self.limits, 0),
        ]
        return float(max(np.max(a, initial=0.0) for a in parts))


@dataclass
class Application:
    name: str
    source_id: str
    variant: str
    f: list
    g: list
    A: np.ndarray
    B: np.ndarray
    c: np.ndarray
    X: Domain
    Z: Domain
    nominal_x: np.ndarray
    nominal_z: np.ndarray
    scales: np.ndarray
    metadata: dict = field(default_factory=dict)

    @property
    def m(self):
        return len(self.f)

    @property
    def nx(self):
        return self.A.shape[1]

    @property
    def nz(self):
        return self.B.shape[1]

    @property
    def sdp(self):
        return any(t.kind == "nuclear" for obj in self.f + self.g for t in obj)

    def block_values(self, block, u, normalized=True):
        objectives = self.f if block == "x" else self.g
        v = np.array([sum(t.value(u) for t in obj) for obj in objectives])
        return v / self.scales if normalized else v

    def values(self, x, z, normalized=True):
        return self.block_values("x", x, normalized) + self.block_values("z", z, normalized)

    def residual(self, x, z):
        return self.A @ x + self.B @ z - self.c

    def violation(self, x, z):
        return max(self.X.violation(x), self.Z.violation(z))


def term(kind, matrix, offset=None, coefficient=1.0, shape=None):
    matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
    offset = np.zeros(matrix.shape[0]) if offset is None else np.asarray(offset, dtype=float)
    return Term(kind, matrix, offset, coefficient, shape)
