"""Paired multistart comparisons of gradient and genuinely composite baselines.

python examples/gradient_comparison.py --problem JOS1 --starts 12 --seed 42
python examples/gradient_comparison.py --problem DTLZ2 --starts 12 --seed 42
All results, including unsuccessful runs, are exported. See docs/algorithms.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from mop_problems import Problem, __version__, check_derivatives, get_problem
from mop_problems.algorithms import (
    BBDMO,
    L1,
    MGDA,
    NesterovAcceleratedGradient,
    NesterovAcceleratedProximalGradient,
    ProximalGradient,
    multistart,
    nondominated_mask,
)
from mop_problems.core import integer


class JOS1Demo(Problem):
    """Convex quadratic example used in TFY (2023), Experiment 1.

    This example is not an additional registered benchmark. The decision box
    [-4,4]^n is explicitly imposed in this experiment; starts use [-2,4]^n.
    """

    def __init__(self, n_var=10):
        super().__init__("JOS1-demo", n_var, 2, -4, 4)

    def _evaluate(self, X):
        return np.column_stack((np.mean(X**2, axis=1), np.mean((X - 2) ** 2, axis=1))), None, None

    def pareto_front(self, n_points=200, *, seed=0):
        x = np.linspace(0, 2, n_points)
        return np.column_stack((x**2, (x - 2) ** 2))


def _serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _json(path, data):
    path.write_text(
        json.dumps(data, indent=2, default=_serializable, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def run(output, *, problem_name="JOS1", starts=12, seed=42, max_iter=300, plots=True):
    starts, seed = integer(starts, "starts"), integer(seed, "seed", 0)
    max_iter = integer(max_iter, "max_iter", 0)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if problem_name == "JOS1":
        problem = JOS1Demo()
        X0 = rng.uniform(-2, 4, size=(starts, problem.n_var))
        initialization = "uniform [-2,4]^n; declared decision box [-4,4]^n"
    elif problem_name == "FON":
        problem = get_problem("FON")
        X0 = rng.uniform(-1, 1, size=(starts, problem.n_var))
        initialization = (
            "uniform [-1,1]^n; avoids exponentially saturated gradients far from the front"
        )
    elif problem_name == "DTLZ2":
        problem = get_problem("DTLZ2", n_var=8, n_obj=3)
        X0 = problem.sample(starts, seed=rng)
        initialization = "uniform in the full decision box"
    else:
        raise ValueError("problem_name must be JOS1, FON or DTLZ2")
    check = check_derivatives(problem, X0[: min(3, starts)])
    if not check.passed:
        raise RuntimeError(f"Example derivative check failed: {check}")
    common = dict(max_iter=max_iter, max_nfev=5000, max_njev=2000, tol=1e-5)
    smooth_methods = {
        "MGDA-fixed": MGDA(step_rule="fixed", step_size=0.1, **common),
        "MGDA-Armijo": MGDA(**common),
        "BBDMO": BBDMO(**common),
        "PG": ProximalGradient(**common),
        "NAG": NesterovAcceleratedGradient(**common),
        "APG": NesterovAcceleratedProximalGradient(**common),
    }
    # Different objective-specific centres/weights make this a real proximal test.
    reg = L1(
        coefficients=[1 / problem.n_var, 1 / (2 * problem.n_var)],
        centres=[
            0,
            1,
        ],
    )
    if problem.n_obj == 3:
        reg = L1(coefficients=[0.05, 0.1, 0.15], centres=[0.0, 0.3, 0.7])
    composite_methods = {
        "PG-L1": ProximalGradient(regularizer=reg, **common),
        "APG-L1": NesterovAcceleratedProximalGradient(regularizer=reg, **common),
    }
    summary, configurations = [], {}
    for case, methods in (("smooth", smooth_methods), ("composite_l1", composite_methods)):
        directory = output / case
        directory.mkdir(exist_ok=True)
        collected = {}
        for label, method in methods.items():
            results = multistart(problem, method, starts=X0)
            collected[label] = results
            F = np.stack([r.fun for r in results])
            feasible = np.array([r.feasible for r in results])
            nd = np.zeros(starts, dtype=bool)
            nd[feasible] = nondominated_mask(F[feasible])
            np.savez_compressed(
                directory / f"{label}.npz",
                X0=X0,
                X=np.stack([r.x for r in results]),
                F=F,
                smooth_F=np.stack([r.smooth_fun for r in results]),
                feasible=feasible,
                nondominated=nd,
            )
            _json(directory / f"{label}.json", [asdict(r) for r in results])
            configurations[f"{case}/{label}"] = results[0].options
            for i, result in enumerate(results):
                summary.append(
                    {
                        "case": case,
                        "algorithm": label,
                        "start": i,
                        "seed": seed,
                        "status": result.status,
                        "success": result.success,
                        "feasible": result.feasible,
                        "nondominated": bool(nd[i]),
                        "iterations": result.n_iter,
                        "nfev": result.nfev,
                        "njev": result.njev,
                        "nprox": result.nprox,
                        "subproblems": result.nsubproblem,
                        "stationarity": result.stationarity,
                        "stationarity_bound": result.stationarity_bound,
                        "seconds": result.elapsed_seconds,
                    }
                )
        if plots:
            _plot(problem, collected, X0, directory / "comparison.png", composite=case != "smooth")
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    _json(
        output / "manifest.json",
        {
            "mop_problems": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": _version("scipy"),
            "matplotlib": _version("matplotlib"),
            "example_sha256": hashlib.sha256(
                Path(__file__).read_text(encoding="utf-8").encode()
            ).hexdigest(),
            "problem": problem.name,
            "n_var": problem.n_var,
            "n_obj": problem.n_obj,
            "bounds": [problem.xl, problem.xu],
            "initialization": initialization,
            "starts": starts,
            "seed": seed,
            "X0": X0,
            "derivative_check": asdict(check),
            "methods": configurations,
            "composite_penalty": {"type": "L1", **asdict(reg)},
            "accounting": "nfev includes joint value/Jacobian passes; njev is their subset. "
            "All diagnostic subproblems and failed line-search trials are counted. "
            "Budgets are per start; compare sums for set-producing algorithms.",
            "interpretation": "First-order stationarity is not a global Pareto certificate. "
            "L1 changes the optimization problem and its Pareto front.",
        },
    )
    return summary


def _plot(problem, collected, X0, output, *, composite):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 4.7), layout="constrained")
    ax = fig.add_subplot(1, 3, 1, projection="3d" if problem.n_obj == 3 else None)
    if not composite:
        front = problem.pareto_front(200)
        ax.scatter(*front.T, color="0.8", s=7, label="Reference PF")
    for label, results in collected.items():
        F = np.stack([r.fun for r in results if r.feasible])
        if len(F):
            ax.scatter(*F.T, s=24, label=label, alpha=0.7)
    ax.set(xlabel="Objective 1", ylabel="Objective 2", title="All feasible terminal points")
    if problem.n_obj == 3:
        ax.set_zlabel("Objective 3")
    ax.legend(fontsize=7)
    ax = fig.add_subplot(1, 3, 2)
    for label, results in collected.items():
        history = [h for h in results[0].history if h.stationarity_bound is not None]
        ax.semilogy(
            [h.iteration for h in history],
            [max(h.stationarity_bound, 1e-16) for h in history],
            label=label,
        )
    ax.axhline(1e-5, color="0.5", linestyle="--", linewidth=1)
    ax.set(xlabel="Iteration", ylabel="Unit-step residual upper bound", title="Shared start #0")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)
    ax = fig.add_subplot(1, 3, 3)
    positions = np.arange(len(collected))
    ax.bar(
        positions - 0.2,
        [sum(r.nfev for r in runs) for runs in collected.values()],
        width=0.4,
        label="All formula passes",
    )
    ax.bar(
        positions + 0.2,
        [sum(r.njev for r in runs) for runs in collected.values()],
        width=0.4,
        label="AD passes (subset)",
    )
    ax.set_xticks(positions, list(collected), rotation=35, ha="right", fontsize=8)
    ax.set(ylabel="Total calls over all starts", title="Measured computational work")
    ax.legend(fontsize=7)
    kind = "smooth objectives" if not composite else "objectives + distinct L1 penalties"
    fig.suptitle(f"{problem.name}: {kind} | {len(X0)} paired starts", fontsize=13)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=["JOS1", "FON", "DTLZ2"], default="JOS1")
    parser.add_argument("--starts", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("outputs/gradient_comparison"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    summary = run(
        args.output,
        problem_name=args.problem,
        starts=args.starts,
        seed=args.seed,
        max_iter=args.max_iter,
        plots=not args.no_plots,
    )
    for label in dict.fromkeys(row["algorithm"] for row in summary):
        rows = [r for r in summary if r["algorithm"] == label]
        print(
            f"{label}: {sum(r['success'] for r in rows)}/{len(rows)} stationary; "
            f"nfev={sum(r['nfev'] for r in rows)}, njev={sum(r['njev'] for r in rows)}"
        )
    print(f"Saved paired starts, terminal sets, histories and manifest to {args.output.resolve()}")


if __name__ == "__main__":
    main()
