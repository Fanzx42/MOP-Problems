"""Seeded numerical and visual experiments; no optimization algorithm is run.

Usage: python examples/representative_suite.py --samples 2000 --seed 42
Use --no-plots for a NumPy-only run. CSV files retain all X, F, G and CV values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from itertools import product
from pathlib import Path

import numpy as np

from mop_problems import __version__, get_problem, get_problem_info

# Explicit case IDs keep output names portable across operating systems.
CASES = [
    ("wfg1", "WFG1", {"n_obj": 2}),
    ("wfg2", "WFG2", {"n_obj": 2}),
    ("wfg3", "WFG3", {"n_obj": 3}),
    ("wfg9_m3", "WFG9", {"n_obj": 3}),
    ("uf1", "UF1", {}),
    ("uf5", "UF5", {}),
    ("uf6", "UF6", {}),
    ("uf8", "UF8", {}),
    ("uf9", "UF9", {}),
    ("c2_dtlz2", "C2-DTLZ2", {}),
    ("c3_dtlz4", "C3-DTLZ4", {}),
    ("welded_beam", "Welded-Beam", {}),
    ("dtlz2_m5", "DTLZ2", {"n_obj": 5}),
    ("wfg9_m8", "WFG9", {"n_obj": 8}),
]


def nondominated_mask(F):
    """Exact minimization dominance, preserving duplicates; O(N^2) worst case.

    Constraint filtering is the caller's responsibility. This educational
    routine bounds temporary storage by N*M instead of constructing N*N*M.
    """
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[1] < 2 or not np.isfinite(F).all():
        raise ValueError("F must be a finite (N,M) array with M>=2")
    keep = np.ones(len(F), dtype=bool)
    for i in np.argsort(F[:, 0], kind="stable"):
        if keep[i]:
            active = np.flatnonzero(keep)
            other = F[active]
            dominates = np.all(other <= F[i], axis=1) & np.any(other < F[i], axis=1)
            if dominates.any():
                keep[i] = False
            else:
                dominated = np.all(F[i] <= other, axis=1) & np.any(F[i] < other, axis=1)
                keep[active[dominated]] = False
    return keep


def save_csv(path, columns, values):
    np.savetxt(path, values, delimiter=",", header=",".join(columns), comments="", fmt="%.17g")


def evaluate_case(output, case_id, name, parameters, samples, seed):
    p = get_problem(name, **parameters)
    X = p.sample(samples, seed=seed)
    result = p.evaluate(X)
    eligible = np.flatnonzero(result.feasible)
    nd = np.zeros(samples, dtype=bool)
    nd[eligible] = nondominated_mask(result.F[eligible])
    columns = [f"x{i + 1}" for i in range(p.n_var)] + [f"f{i + 1}" for i in range(p.n_obj)]
    columns += [f"g{i + 1}" for i in range(p.n_ieq_constr)] + ["cv", "feasible", "sample_nd"]
    save_csv(
        output / f"{case_id}.csv",
        columns,
        np.column_stack((X, result.F, result.G, result.CV, result.feasible, nd)),
    )
    if get_problem_info(name)["pareto_front"]:
        save_csv(
            output / f"{case_id}_reference.csv",
            [f"f{i + 1}" for i in range(p.n_obj)],
            p.pareto_front(301, seed=seed),
        )
    summary = dict(
        case=case_id,
        problem=name,
        n_var=p.n_var,
        n_obj=p.n_obj,
        n_constraints=p.n_ieq_constr,
        samples=samples,
        seed=seed,
        feasible=int(result.feasible.sum()),
        sample_nd=int(nd.sum()),
        feasible_fraction=float(result.feasible.mean()),
    )
    resolved = dict(
        case=case_id,
        name=name,
        parameters=parameters,
        n_var=p.n_var,
        n_obj=p.n_obj,
        seed=seed,
        lower_bounds=p.xl.tolist(),
        upper_bounds=p.xu.tolist(),
    )
    return summary, resolved, (p, result, nd)


def multimodal_sets(output):
    """Construct all nine known Pareto subsets for the two-variable variants."""
    sets = {}
    for name, case in [
        ("Omni-test", "omni_sets"),
        ("SYM-PART", "sympart_sets"),
        ("SYM-PART-Rotated", "sympart_rotated_sets"),
    ]:
        p = get_problem(name)
        rows = []
        for label, (a, b) in enumerate(product(range(3), repeat=2)):
            if name == "Omni-test":
                t = np.linspace(1, 1.5, 101)
                X = np.column_stack((t + 2 * a, t + 2 * b))
            else:
                t = np.linspace(-1, 1, 101)
                X = np.column_stack((t + 10 * (a - 1), np.full(len(t), 10 * (b - 1))))
                c, s = np.cos(p.angle), np.sin(p.angle)
                X = X @ np.array([[c, s], [-s, c]])
            rows.append(np.column_stack((np.full(len(X), label), X, p.evaluate(X).F)))
        values = np.vstack(rows)
        save_csv(output / f"{case}.csv", ["subset", "x1", "x2", "f1", "f2"], values)
        sets[name] = values
    return sets


def constrained_surface(output):
    p = get_problem("C2-DTLZ2")
    angles = np.linspace(0, 1, 51)
    first, second = np.meshgrid(angles, angles)
    X = np.full((first.size, p.n_var), 0.5)
    X[:, :2] = np.column_stack((first.ravel(), second.ravel()))
    result = p.evaluate(X)
    save_csv(
        output / "c2_dtlz2_g0_surface.csv",
        ["f1", "f2", "f3", "g1", "feasible"],
        np.column_stack((result.F, result.G, result.feasible)),
    )
    return result


def plot_results(output, batches, sets, surface):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    blue, orange, grey = "#2878a5", "#d55e00", "#bac3ca"
    fig = plt.figure(figsize=(13, 8), layout="constrained")
    for i, case in enumerate(["wfg1", "wfg2", "uf1", "uf6", "uf8", "wfg9_m3"], 1):
        p, result, nd = batches[case]
        ax = fig.add_subplot(2, 3, i, projection="3d" if p.n_obj == 3 else None)
        F = result.F
        ax.scatter(*F.T, s=6, color=grey, alpha=0.35, label="Box samples", rasterized=True)
        ax.scatter(*F[nd].T, s=12, color=orange, label="Sample nondominated")
        if get_problem_info(p.name)["pareto_front"]:
            front = p.pareto_front(301, seed=42)
            ax.scatter(*front.T, s=5, color=blue, label="Analytical front")
        ax.set(xlabel="$f_1$", ylabel="$f_2$", title=f"{p.name} | n={p.n_var}, M={p.n_obj}")
        if p.n_obj == 3:
            ax.set_zlabel("$f_3$")
        if i in (1, 3):
            ax.legend(fontsize=7, loc="best")
    fig.suptitle(
        "Representative objective geometry | uniform box sampling, no optimizer", fontsize=14
    )
    fig.savefig(output / "objective_geometry.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    palette = plt.get_cmap("tab10")
    for col, (name, values) in enumerate(sets.items()):
        for label in range(9):
            part = values[values[:, 0] == label]
            axes[0, col].plot(part[:, 1], part[:, 2], color=palette(label), linewidth=2.4)
            axes[1, col].plot(
                part[:, 3], part[:, 4], color=palette(label), linewidth=1.4, alpha=0.7
            )
        axes[0, col].set(title=name, xlabel="$x_1$", ylabel="$x_2$")
        axes[1, col].set(xlabel="$f_1$", ylabel="$f_2$", title="Nine subsets map to the same front")
        for row in (0, 1):
            axes[row, col].set_aspect("equal", adjustable="box")
            axes[row, col].grid(alpha=0.15)
    fig.suptitle("Equivalent Pareto subsets | analytical constructions", fontsize=14)
    fig.savefig(output / "multimodal_sets.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 5), layout="constrained")
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for mask, color, label in [
        (~surface.feasible, grey, "Infeasible g=0 points"),
        (surface.feasible, blue, "Feasible g=0 points"),
    ]:
        ax.scatter(*surface.F[mask].T, color=color, s=5, label=label)
    ax.set(
        xlabel="$f_1$",
        ylabel="$f_2$",
        zlabel="$f_3$",
        title="C2-DTLZ2 | constraint on the unit sphere",
    )
    ax.legend(fontsize=8)
    ax = fig.add_subplot(1, 2, 2)
    _, beam, nd = batches["welded_beam"]
    for mask, color, label in [
        (~beam.feasible, grey, "Infeasible box samples"),
        (beam.feasible, blue, "Feasible box samples"),
        (nd, orange, "Feasible sample nondominated"),
    ]:
        ax.scatter(*beam.F[mask].T, color=color, s=9, alpha=0.7, label=label)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Fabrication cost ($f_1$)",
        ylabel="Deflection ($f_2$, inches)",
        title="Welded-Beam | four-constraint variant",
    )
    ax.legend(fontsize=8)
    fig.savefig(output / "constraint_geometry.png", dpi=160)
    plt.close(fig)


def run(output: Path, samples: int = 2000, seed: int = 42, plots: bool = True):
    if samples < 2 or seed < 0:
        raise ValueError("samples must be >=2 and seed must be nonnegative")
    output.mkdir(parents=True, exist_ok=True)
    summaries, configurations, batches = [], [], {}
    for index, (case, name, parameters) in enumerate(CASES):
        summary, config, batch = evaluate_case(
            output, case, name, parameters, samples, seed + index
        )
        summaries.append(summary)
        configurations.append(config)
        batches[case] = batch
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    sets, surface = multimodal_sets(output), constrained_surface(output)
    if plots:
        plot_results(output, batches, sets, surface)
    manifest = dict(
        package_version=__version__,
        python=platform.python_version(),
        numpy=np.__version__,
        script_sha256=hashlib.sha256(
            Path(__file__).read_text(encoding="utf-8").encode("utf-8")
        ).hexdigest(),
        samples_per_case=samples,
        base_seed=seed,
        rng="numpy.random.default_rng (PCG64)",
        method="uniform decision-box sampling; no optimizer",
        dominance="all objectives minimized; raw G<=0; duplicate objective rows retained",
        warning="Sample nondominated points are not an estimate with certified PF accuracy.",
        cases=configurations,
        analytical_constructions={
            "multimodal": "101 points per subset, all 9 subsets",
            "c2_dtlz2": "51x51 position grid, distance variables=0.5",
        },
    )
    if plots:
        import matplotlib

        manifest["matplotlib"] = matplotlib.__version__
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/representative"))
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.samples < 2 or args.seed < 0:
        parser.error("samples must be >=2 and seed must be nonnegative")
    summaries = run(args.output, args.samples, args.seed, not args.no_plots)
    print(f"Saved {len(summaries)} reproducible sampling cases to {args.output.resolve()}")


if __name__ == "__main__":
    main()
