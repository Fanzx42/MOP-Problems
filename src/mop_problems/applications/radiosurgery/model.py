"""Four clinical criteria and the paper's explicit dual LP (one fixed pass).

The influence matrices below are synthetic Gaussian kernels, not clinical data.
The 8 sectors / 3 collimators and max-sector beam-on-time formulation are kept.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from ._types import Application, Domain, term


@dataclass
class Case:
    problem: Application
    K: np.ndarray
    groups: dict
    thresholds: dict
    C: np.ndarray
    T: np.ndarray
    dual_A: np.ndarray
    dual_c: np.ndarray
    seed: int

    def dual_bounds(self, weight):
        g, w = self.groups, np.asarray(weight)
        upper = np.r_[
            np.full(len(g["TS"]), w[0] / len(g["TS"])),
            np.full(len(g["TI"]), w[0] / len(g["TI"])),
            np.full(len(g["R"]), w[1] / len(g["R"])),
            np.full(len(g["LD"]), w[2] / len(g["LD"])),
            np.full(len(g["O"]), np.inf),
            np.full(self.C.shape[0], np.inf),
        ]
        bound = np.r_[np.zeros(self.K.shape[1]), np.full(self.T.shape[1], w[3])]
        return bound, upper

    def primal_from_dual_multiplier(self, multiplier):
        n = self.K.shape[1]
        return np.asarray(multiplier)[: n + self.T.shape[1]]

    def coupling_multiplier(self, dual):
        g, h = self.groups, self.thresholds
        result = np.zeros(len(self.K))
        cursor = 0
        for group, sign in [("TS", -1), ("TI", -1), ("R", 1), ("LD", 1), ("O", 1)]:
            ids = g[group]
            count = len(ids)
            result[ids] = sign * dual[cursor : cursor + count] / h[group]
            cursor += count
        return result

    def physical(self, x):
        """Feasible recovery for evaluation only; the raw iterate is saved too."""
        n = self.K.shape[1]
        times = np.maximum(np.asarray(x)[:n], 0.0)
        oar = self.K[self.groups["O"]] @ times
        factor = min(1.0, float(np.min(self.thresholds["O"] / np.maximum(oar, 1e-300))))
        times = times * factor
        tau = (self.C @ times).reshape(self.T.shape[1], 8).max(axis=1)
        recovered = np.r_[times, tau]
        return recovered, self.K @ times

    def objectives(self, x):
        """Direct physical evaluation independent of Application.Term.value."""
        n = self.K.shape[1]
        dose = self.K @ np.asarray(x)[:n]
        g, h = self.groups, self.thresholds
        target = sum(np.maximum(1 - dose[g[s]] / h[s], 0).mean() for s in ("TS", "TI"))
        ring = np.maximum(dose[g["R"]] / h["R"] - 1, 0).mean()
        low = np.maximum(dose[g["LD"]] / h["LD"] - 1, 0).mean()
        return np.array([target, ring, low, np.asarray(x)[n:].sum()])

    def lp(self, weight, caps=None, cap_slack=0.0):
        """Independent physical primal epigraph LP, solved by HiGHS dual simplex.

        caps optionally performs a Pareto dominance check. No ADMM matrices or
        Application term expressions are used to build this reference.
        """
        g, h, K = self.groups, self.thresholds, self.K
        n = K.shape[1]
        ni = self.T.shape[1]
        ids = np.r_[g["TS"], g["TI"], g["R"], g["LD"]]
        dim = n + ni + len(ids)
        rows, limits = [], []
        costs = np.zeros((4, dim))
        cursor = n + ni
        for group, objective, sign in [("TS", 0, -1), ("TI", 0, -1), ("R", 1, 1), ("LD", 2, 1)]:
            for index in g[group]:
                row = np.zeros(dim)
                row[:n] = sign * K[index] / h[group]
                row[cursor] = -1.0
                rows.append(row)
                limits.append(float(sign))
                costs[objective, cursor] = 1 / len(g[group])
                cursor += 1
        for _i, row_K in enumerate(K[g["O"]]):
            row = np.zeros(dim)
            row[:n] = row_K
            rows.append(row)
            limits.append(h["O"])
        for c, t in zip(self.C, self.T, strict=False):
            row = np.zeros(dim)
            row[:n] = c
            row[n : n + ni] = -t
            rows.append(row)
            limits.append(0.0)
        costs[3, n : n + ni] = 1.0
        if caps is not None:
            rows.extend(costs)
            limits.extend(np.asarray(caps) + cap_slack)
        answer = linprog(
            np.asarray(weight) @ costs,
            A_ub=np.array(rows),
            b_ub=np.array(limits),
            bounds=(0, None),
            method="highs-ds",
            options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9},
        )
        if not answer.success:
            raise RuntimeError("Physical reference LP: " + answer.message)
        x = answer.x[: n + ni]
        return dict(
            value=float(answer.fun),
            objectives=self.objectives(x),
            x=x,
            iterations=int(answer.nit),
            lp_vector=answer.x,
        )


def make_case(seed=101, isocenters=1):
    rng = np.random.default_rng(seed)

    def sphere(number, radius):
        v = rng.normal(size=(number, 3))
        v /= np.linalg.norm(v, axis=1)[:, None]
        return v * np.broadcast_to(np.asarray(radius), (number,))[:, None]

    points = [
        sphere(4, 0.43),
        sphere(4, rng.uniform(0.05, 0.32, 4)),
        sphere(8, 0.67),
        sphere(12, rng.uniform(0.86, 1.3, 12)),
        np.array([1.05, 0.05, 0.1]) + sphere(4, 0.09),
    ]
    groups = {}
    cursor = 0
    for name, q in zip(("TS", "TI", "R", "LD", "O"), points, strict=False):
        groups[name] = np.arange(cursor, cursor + len(q))
        cursor += len(q)
    points = np.vstack(points)
    centers = rng.normal(scale=0.07, size=(isocenters, 3))
    columns = []
    for center in centers:
        delta = points - center
        for s in range(8):
            angle = 2 * np.pi * s / 8
            direction = np.array([np.cos(angle), np.sin(angle), 0.25])
            directional = delta @ direction
            for width, amplitude in zip((0.32, 0.49, 0.72), (0.65, 0.95, 1.3), strict=False):
                radius2 = np.sum(delta * delta, axis=1) + 0.15 * directional**2
                columns.append(
                    amplitude
                    * np.exp(-radius2 / (2 * width**2))
                    * (1 + 0.025 * rng.normal(size=len(points)))
                )
    K = np.maximum(np.array(columns).T, 0.0)
    C = np.kron(np.eye(isocenters * 8), np.ones((1, 3)))
    T = np.kron(np.eye(isocenters), np.ones((8, 1)))
    n, nd, ni = K.shape[1], len(K), isocenters
    h = dict(TS=1.0, TI=1.0, R=1.0, LD=0.5, O=0.35)
    X = Domain.box(n + ni, 0.0)
    X.inequalities = np.c_[C, -T]
    X.limits = np.zeros(len(C))
    Z = Domain.box(nd, 0.0)
    Z.upper[groups["O"]] = h["O"]
    f = [[], [], [], [term("linear", np.r_[np.zeros(n), np.ones(ni)])]]
    identity = np.eye(nd)
    g = [
        [
            term("hinge", -identity[groups[s]] / h[s], -np.ones(len(groups[s])), 1 / len(groups[s]))
            for s in ("TS", "TI")
        ],
        [
            term(
                "hinge",
                identity[groups["R"]] / h["R"],
                np.ones(len(groups["R"])),
                1 / len(groups["R"]),
            )
        ],
        [
            term(
                "hinge",
                identity[groups["LD"]] / h["LD"],
                np.ones(len(groups["LD"])),
                1 / len(groups["LD"]),
            )
        ],
        [],
    ]
    A = np.c_[K, np.zeros((nd, ni))]
    p = Application(
        "radiosurgery",
        "A05-four-objective",
        "synthetic Gaussian influence; fixed one-pass sample",
        f,
        g,
        A,
        -np.eye(nd),
        np.zeros(nd),
        X,
        Z,
        np.zeros(n + ni),
        np.zeros(nd),
        np.ones(4),
        dict(
            seed=seed,
            isocenters=ni,
            sectors=8,
            collimators=3,
            clinical_data=False,
            two_pass=False,
            thresholds=h,
            points=points.tolist(),
            centers=centers.tolist(),
        ),
    )
    dose_blocks = [K[groups[s]] / h[s] for s in ("TS", "TI", "R", "LD", "O")]
    signs = (1, 1, -1, -1, -1)
    upper_rows = np.concatenate(
        [sign * q.T for sign, q in zip(signs, dose_blocks, strict=True)] + [-C.T], axis=1
    )
    lower_rows = np.c_[np.zeros((ni, nd)), T.T]
    dual_A = np.vstack((upper_rows, lower_rows))
    dual_c = np.r_[
        -np.ones(len(groups["TS"]) + len(groups["TI"])),
        np.ones(len(groups["R"]) + len(groups["LD"]) + len(groups["O"])),
        np.zeros(len(C)),
    ]
    return Case(p, K, groups, h, C, T, dual_A, dual_c, seed)


def initial(case, start):
    n = case.K.shape[1]
    mix = np.array([0.7, 0.2, 0.1]) if start == 0 else np.array([0.15, 0.25, 0.6])
    times = np.tile(mix, n // 3)
    times *= (0.8 if start == 0 else 1.15) / np.mean(case.K[case.groups["TS"]] @ times)
    x, z = case.physical(np.r_[times, np.zeros(case.T.shape[1])])
    return x, z


def weight_grid():
    values = []
    for low in (0.1, 1.0, 10.0):
        for bot in (0.1, 1.0, 10.0):
            w = np.array([1.0, 1.0, low, bot])
            values.append(w / w.sum())
    return np.array(values)
