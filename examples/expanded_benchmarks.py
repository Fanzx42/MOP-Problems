"""Reproducible expanded benchmarks, Jacobians, dynamic snapshots and equalities.

python examples/expanded_benchmarks.py --samples 1000 --seed 42
Use --no-plots for NumPy-only execution. No optimizer is run by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from mop_problems import __version__, get_problem, get_problem_info

CASES = [
    ("IMOP2", {}),
    ("MMF1", {}),
    ("ZCAT1", {"n_obj": 2, "n_var": 10, "level": 3}),
    ("MW2", {}),
    ("NCTP1", {"n_var": 10}),
    ("DASCMOP2", {"difficulty": 4}),
    ("CMMOP4", {}),
    ("LSMOP1", {"n_var": 100, "n_obj": 3}),
    ("RE21", {}),
    ("RCM50", {}),
]


def export(output, name, parameters, x, sampling, records):
    p = get_problem(name, **parameters)
    value = p.evaluate(x)
    labels, arrays = [], []
    for label, array in (("x", x), ("f", value.F), ("g", value.G), ("h", value.H)):
        labels.extend(f"{label}{j + 1}" for j in range(array.shape[1]))
        arrays.append(array)
    labels.extend(("CV", "feasible"))
    arrays.extend((value.CV[:, None], value.feasible[:, None]))
    case_id = name + (f"_t{parameters['time']:g}" if "time" in parameters else "")
    path = output / (case_id + ".csv")
    np.savetxt(
        path,
        np.column_stack(arrays),
        delimiter=",",
        header=",".join(labels),
        comments="",
        fmt="%.17g",
    )
    # Evaluate a single independent regular interior point. At larger dimensions
    # this also avoids the quadratic seed-memory cost of a full batch Jacobian.
    point = p.sample(1, seed=238)
    if name == "NCTP1":
        point[:, 1:] = 1 + 0.01 * (point[:, 1:] - 2.5)
    jac = p.jacobian(point)
    info = get_problem_info(name)
    records.append(
        {
            "case": case_id,
            "problem": name,
            "parameters": parameters,
            "n_var": p.n_var,
            "n_obj": p.n_obj,
            "sampling": sampling,
            "samples": len(x),
            "feasible_count": int(value.feasible.sum()),
            "equality_tolerance": 1e-6,
            "csv": path.name,
            "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "references": info["references"],
            "reference_implementation": info["reference_implementation"],
            "variant_notes": info["notes"],
            "derivative_info": p.derivative_info,
            "gradient_point": point[0].tolist(),
            "dF": jac.dF[0].tolist(),
            "dG": jac.dG[0].tolist(),
            "dH": jac.dH[0].tolist(),
        }
    )
    return value


def run(output, samples=1000, seed=42, plots=True):
    if samples < 2:
        raise ValueError("samples must be at least 2")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records, values = [], {}
    for name, parameters in CASES:
        p = get_problem(name, **parameters)
        x = p.sample(samples, seed=seed)
        sampling = "uniform box"
        if name == "NCTP1":
            x[:, 1:] = 1 + 0.01 * (x[:, 1:] - 2.5)
            sampling = "x1 uniform [0,5]; tails uniform [0.975,1.025] near Rosenbrock valley"
        values[name] = export(output, name, parameters, x, sampling, records)

    dynamic = get_problem("FDA1")
    shared = dynamic.sample(samples, seed=seed)
    snapshots = {}
    for t in (0.0, 0.5, 1.0):
        snapshots[t] = export(
            output,
            "FDA1",
            {"time": t},
            shared,
            "identical decision sample at every fixed time",
            records,
        )

    eq = get_problem("Eq-DTLZ2", n_obj=3, n_equalities=1)
    theta = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    x = np.full((samples, eq.n_var), 0.5)
    x[:, 0] += 0.4 * np.cos(theta)
    x[:, 1] += 0.4 * np.sin(theta)
    equalities = export(
        output,
        "Eq-DTLZ2",
        {"n_obj": 3, "n_equalities": 1},
        x,
        "analytical feasible circle in first two decisions; remaining decisions 0.5",
        records,
    )
    assert equalities.feasible.all()

    if plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        for ax, name in zip(
            axes.flat, ("IMOP2", "MMF1", "ZCAT1", "MW2", "NCTP1", "CMMOP4"), strict=True
        ):
            ev = values[name]
            for mask, color, label in (
                (~ev.feasible, "#cbd5e1", "infeasible sample"),
                (ev.feasible, "#2563eb", "feasible sample"),
            ):
                if np.any(mask):
                    ax.scatter(*ev.F[mask, :2].T, s=8, color=color, alpha=0.65, label=label)
            ax.set(title=name, xlabel="f1 (minimize)", ylabel="f2 (minimize)")
            ax.legend(fontsize=8)
        fig.suptitle("Expanded benchmarks: sampled objectives, not reference Pareto fronts")
        fig.savefig(output / "expanded_samples.png", dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(
            1, 3, figsize=(12, 3.8), sharex=True, sharey=True, constrained_layout=True
        )
        for ax, (t, ev) in zip(axes, snapshots.items(), strict=True):
            ax.scatter(*ev.F.T, s=8, alpha=0.6, color="#2563eb")
            ax.set(title=f"FDA1, time={t:g}", xlabel="f1", ylabel="f2")
        fig.suptitle("Same decisions at three explicit times (sample clouds)")
        fig.savefig(output / "dynamic_snapshots.png", dpi=160)
        plt.close(fig)

        fig = plt.figure(figsize=(10, 4.3), constrained_layout=True)
        ax = fig.add_subplot(121)
        ax.plot(x[:, 0], x[:, 1], color="#2563eb")
        ax.set(xlabel="x1", ylabel="x2", title="Eq-DTLZ2: H = 0, tail = 0.5", aspect="equal")
        ax = fig.add_subplot(122, projection="3d")
        ax.scatter(*equalities.F.T, s=8, c=theta, cmap="viridis")
        ax.set(
            xlabel="f1", ylabel="f2", zlabel="f3", title="Objectives of the feasible construction"
        )
        fig.savefig(output / "equality_slice.png", dpi=160)
        plt.close(fig)

    manifest = {
        "library_version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "seed": seed,
        "samples": samples,
        "purpose": "problem/derivative demonstrations; no optimizer or performance ranking",
        "cases": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(records)} cases to {output.resolve()}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/expanded"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    run(args.output, args.samples, args.seed, not args.no_plots)


if __name__ == "__main__":
    main()
