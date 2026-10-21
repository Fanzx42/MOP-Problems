"""Small deterministic demonstrations of every requested algorithm family.

Run: python examples/unified_algorithms.py
These examples verify usage; they are not a speed ranking or a Pareto-front study.
"""

import numpy as np

from mop_problems import Problem
from mop_problems.algorithms import L1, Polytope, SplitProblem, get_algorithm


class Quadratic(Problem):
    def __init__(self, centres=((1.0, 1.0), (-1.0, -1.0)), *, constrained=False):
        self.centres = np.asarray(centres)
        self.constrained = constrained
        super().__init__(
            "demo-quadratic",
            self.centres.shape[1],
            len(self.centres),
            -10,
            10,
            n_ieq_constr=int(constrained),
            n_eq_constr=int(constrained),
        )

    def _evaluate(self, X):
        F = np.column_stack([0.5 * np.sum((X - c) ** 2, axis=1) for c in self.centres])
        return (F, 2 - X[:, 1:2], X[:, :1] - 1) if self.constrained else (F, None, None)


def run():
    problem = Quadratic()
    cases = [
        ("MGDA fixed", problem, [4, -3], "mgda", {"step_rule": "fixed", "L": 2}),
        ("MGDA Armijo", problem, [4, -3], "mgda", {}),
        ("BBDMO", problem, [4, -3], "bbdmo", {}),
        ("CG PRP+", problem, [4, -3], "cg", {"beta_rule": "PRP+"}),
        (
            "PG fixed",
            problem,
            [4, -3],
            "pg",
            {"step_rule": "fixed", "L": 2, "regularizer": L1([0.1, 0.2])},
        ),
        ("PG search", problem, [4, -3], "pg", {"regularizer": L1([0.1, 0.2])}),
        ("AMG-QP", problem, [4, -3], "amg_qp", {"L": 2}),
        ("AMG proximal", problem, [4, -3], "amg_proximal", {"L": 2, "regularizer": L1([0.1, 0.2])}),
        ("FW diminishing", problem, [4, -3], "fw", {"step_rule": "diminishing"}),
        ("FW adaptive", problem, [4, -3], "fw", {}),
    ]
    poly = Polytope([[0, 0], [1, 0], [0, 1]])
    constrained_domain = Quadratic(((0.8, 0.1), (0.8, 0.1)))
    for name in ("away_fw", "pairwise_fw"):
        cases.append((name, constrained_domain, [0.4, 0.4], name, {"domain": poly}))
    cases.append(("ALM", Quadratic(((0, 0), (0, 0)), constrained=True), [-2, -1], "alm", {}))
    split = SplitProblem(
        Quadratic(((0.0,), (0.0,))), Quadratic(((2.0,), (2.0,))), [[1]], [[-1]], [0]
    )
    cases.append(("ADMM adaptive", split, [0, 2], "admm", {"penalty": 0.01}))
    results = []
    for label, p, x0, name, options in cases:
        result = get_algorithm(name, **options).solve(p, x0)
        results.append(result)
        print(
            f"{label:18s} {result.status:12s} k={result.n_iter:4d} "
            f"residual={result.stationarity_bound!s:22s} x={np.round(result.x, 6)}"
        )
    return results


if __name__ == "__main__":
    run()
