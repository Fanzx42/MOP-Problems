# ruff: noqa: E402 -- set BLAS thread counts before importing NumPy
"""Same-machine ablations: algebraic speedup, starting rho, and adaptive rho."""

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
import hashlib
import json
import platform
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ._engine import solve_admm
from ._io import clean, save
from .experiment import inspect_result
from .model import initial, make_case, weight_grid
from .numerics import NumericalOracle, ResidualBalance, solve_numeric_admm
from .oracle import CachedOracle

ROOT = Path(__file__).resolve().parent
METHODS = ("legacy-fixed-0.3", "numeric-fixed-0.3", "numeric-fixed-1", "numeric-adaptive-1")
SOURCES = tuple(sorted(path.name for path in ROOT.glob("*.py")))


def initialize(out, repeats):
    contract = dict(
        source_sha256={s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in SOURCES},
        methods=METHODS,
        validation_seed=101,
        test_seeds=[202, 303],
        starts=[0, 1],
        weights=weight_grid().tolist(),
        repeats=repeats,
        tolerance=1e-4,
        inner_tolerance=2e-6,
        solver_tolerance=1e-9,
        dose_dual_tolerance=1e-12,
        balance=asdict(ResidualBalance()),
        eta=1.0,
        kappa=1.0,
        initial_penalties={
            "legacy-fixed-0.3": 0.3,
            "numeric-fixed-0.3": 0.3,
            "numeric-fixed-1": 1.0,
            "numeric-adaptive-1": 1.0,
        },
        selection="Legacy rho=.3 was selected on seed 101 in the preceding study; "
        "rho=1 and balance constants are prescribed defaults, no test tuning.",
        timing="One CPU thread; solver time includes construction/checks and excludes "
        "independent post-run LP/KKT/dominance audits. All nine weights run sequentially.",
        data="One-isocenter synthetic Gaussian dose map, not clinical data.",
        references=[
            "https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf",
            "https://arxiv.org/abs/1704.06209",
        ],
    )
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.json"
    # JSON normalization makes tuples compare consistently on resume.
    contract = json.loads(json.dumps(contract))
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8"))["contract"] != contract:
            raise RuntimeError("Source/configuration changed: use another output directory")
    else:
        save(
            manifest,
            dict(
                contract=contract,
                python=sys.version,
                numpy=np.__version__,
                platform=platform.platform(),
            ),
        )
        for source in SOURCES:
            target = out / "source_snapshot" / source
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / source).read_bytes())


def read_rows(out):
    path = out / "runs.jsonl"
    return (
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if path.exists()
        else []
    )


def run(out, repeats):
    initialize(out, repeats)
    done = {r["id"] for r in read_rows(out)}
    warm = make_case(11)
    o = NumericalOracle(warm.problem)
    o.close()
    cases = [("validation", 101, 0, 0, [0, 4, 8])]
    cases += [
        ("test", seed, start, repeat, list(range(9)))
        for repeat in range(repeats)
        for seed in (202, 303)
        for start in (0, 1)
    ]
    for stage, seed, start, repeat, indices in cases:
        case = make_case(seed)
        x, z = initial(case, start)
        methods = METHODS if (seed + start + repeat) % 2 else METHODS[::-1]
        for method in methods:
            for index in indices:
                rid = f"{stage}_{seed}_s{start}_r{repeat}_{method}_w{index}"
                if rid in done:
                    continue
                w = weight_grid()[index]
                penalty = 0.3 if method.endswith("0.3") else 1.0
                if method.startswith("legacy"):
                    result = solve_admm(
                        case.problem,
                        x,
                        z,
                        w,
                        penalty=penalty,
                        seconds_limit=60.0,
                        oracle_factory=CachedOracle,
                    )
                else:
                    balance = ResidualBalance() if method == "numeric-adaptive-1" else None
                    result = solve_numeric_admm(
                        case.problem, x, z, w, penalty=penalty, seconds_limit=60.0, balance=balance
                    )
                result["initial_weight"] = w
                inspected = inspect_result(case, result, 1e-4)
                state_file = f"states/{rid}.npz"
                (out / "states").mkdir(exist_ok=True)
                np.savez_compressed(
                    out / state_file,
                    **{k: result[k] for k in ("x", "z", "y", "w", "beta")},
                    x0=x,
                    z0=z,
                    w0=w,
                )
                save(out / f"histories/{rid}.json", result["history"])
                row = dict(
                    id=rid,
                    method=method,
                    stage=stage,
                    seed=seed,
                    start=start,
                    repeat=repeat,
                    weight_index=index,
                    initial_weight=w,
                    penalty=penalty,
                    state_file=state_file,
                    **{
                        k: result.get(k)
                        for k in (
                            "seconds",
                            "iterations",
                            "status",
                            "reason",
                            "residual",
                            "max_inner_gap",
                            "solves",
                            "barrier_iterations",
                            "final_penalty",
                            "penalty_updates",
                            "operator_work",
                        )
                    },
                    **inspected,
                )
                with (out / "runs.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(clean(row), ensure_ascii=False, allow_nan=False) + "\n")
                print(
                    rid,
                    result["status"],
                    "valid=" + str(row["validated"]),
                    "seconds=" + str(round(result["seconds"], 4)),
                    "iterations=" + str(result["iterations"]),
                    "domination=" + str(round(row["domination_gap_sum"], 7)),
                    flush=True,
                )
                done.add(rid)


def summarize(out, repeats):
    rows = read_rows(out)
    tests = [r for r in rows if r["stage"] == "test"]
    summary, sets = {}, []
    for method in METHODS:
        group = [r for r in tests if r["method"] == method]
        for seed in (202, 303):
            case = make_case(seed)
            refs = [case.lp(w)["value"] for w in weight_grid()]
            for start in (0, 1):
                for repeat in range(repeats):
                    batch = [
                        r
                        for r in group
                        if (r["seed"], r["start"], r["repeat"]) == (seed, start, repeat)
                    ]
                    if len(batch) != 9:
                        continue
                    objectives = np.array([r["recovered_objectives"] for r in batch])
                    regret = [
                        max(0.0, float(np.min(objectives @ w) - ref)) / (1 + abs(ref))
                        for w, ref in zip(weight_grid(), refs, strict=False)
                    ]
                    sets.append(
                        dict(
                            method=method,
                            seed=seed,
                            start=start,
                            repeat=repeat,
                            seconds=sum(r["seconds"] for r in batch),
                            max_grid_regret=max(regret),
                            iterations=sum(r["iterations"] for r in batch),
                        )
                    )
        own = [r for r in sets if r["method"] == method]
        summary[method] = dict(
            runs=len(group),
            validated=sum(r["validated"] for r in group),
            materially_dominated=sum(r["materially_dominated"] for r in group),
            median_set_seconds=statistics.median(r["seconds"] for r in own),
            median_run_iterations=statistics.median(r["iterations"] for r in group),
            median_set_iterations=statistics.median(r["iterations"] for r in own),
            median_max_grid_regret=statistics.median(r["max_grid_regret"] for r in own),
            max_residual=max(r["audit"]["residual"] for r in group),
            max_supporting_gap=max(r["supporting_gap"] for r in group),
            max_domination_gap=max(r["domination_gap_sum"] for r in group),
            min_final_penalty=min(r["final_penalty"] or r["penalty"] for r in group),
            max_final_penalty=max(r["final_penalty"] or r["penalty"] for r in group),
        )
    save(out / "summary.json", dict(methods=summary, sets=sets))
    lines = [
        "# ADMM numerical implementation and penalty ablations",
        "",
        "Two small synthetic dose instances, two starts, nine weights and "
        + str(repeats)
        + " timing repeats. All weights run sequentially on one CPU thread. "
        "Same local models, eta=kappa=1, KKT/consensus tolerance 1e-4. "
        "The six numerical-equivalence/penalty tests and independent physical LP audits "
        "are separate from measured solver time.",
        "",
        "| Method | Median seconds / nine outputs | Median iterations / output "
        "| Numerical audit | Dominated >1e-3 | Worst-grid regret (median) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, s in summary.items():
        lines.append(
            f"| {method} | {s['median_set_seconds']:.6f} | {s['median_run_iterations']:g} | "
            f"{s['validated']}/{s['runs']} | {s['materially_dominated']}/{s['runs']} | "
            f"{s['median_max_grid_regret']:.6g} |"
        )
    lines += [
        "",
        "## What changed",
        "",
        "* Form A.T@A once; cache its symbolic quadratic and rebuild the quadra"
        "tic objective only when rho changes. Update linear objective coefficie"
        "nts and constraint RHS numerically.",
        "* Cache Ax between the time update, dose update, multiplier update and"
        " certificate. B=-I uses a sign operation. Coordinate hinges use compil"
        "ed index arrays and vectorized formulas; no dense selector products.",
        "* The same local error tolerances and native solver tolerances are ret"
        "ained. This is not a speedup from relaxing accuracy.",
        "* Adaptive rho starts at 1, compares normalized physical primal/dual r"
        "esiduals every 10 iterations, uses imbalance ratio 10 and multiplicati"
        "ve factor 2, clips to [1e-4,1e4], and freezes after iteration 200.",
        "* Relative residual denominators are max(1, ||Ax||, ||z||, ||c||) and "
        "max(1, ||A.T y||); the unit floor guards small denominators and is not"
        " a claim of complete scale invariance.",
        "* y is an unscaled multiplier and is left unchanged at a rho change. E"
        "ta and kappa remain fixed: scalar ADMM does not supply an update theor"
        "em for multiobjective weight regularization.",
        "* Physical residual balancing is a parameter heuristic, not a sufficie"
        "nt MOO stopping criterion. Fresh common-weight KKT and alpha/beta agre"
        "ement still control termination.",
        "",
        "## Interpretation",
        "",
        "Compare legacy-fixed-0.3 with numeric-fixed-0.3 to isolate model/operator implementation. "
        "Compare numeric-fixed-1 with numeric-adaptive-1 to isolate penalty adaptation. "
        "Final preferences may differ between penalties, so finite-grid coverage and domination "
        "are reported in addition to residuals. Classical scalar ADMM convergence theorems "
        "do not automatically prove this multiobjective adaptive variant convergent.",
        "",
        "The original frozen results remain unchanged. This benchmark does not retest ALM or "
        "repair its weak-efficiency stopping behavior.",
        "",
        "## References",
        "",
        "[Boyd et al., Sections 3.3-3.4]"
        "(https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf); "
        "[Wohlberg, residual balancing and scaling](https://arxiv.org/abs/1704.06209).",
        "",
        "Raw runs: runs.jsonl; source/configuration contract: manifest.json; all terminal and "
        "initial states: states/; independent re-audit: audit.json.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("SUMMARY " + json.dumps(summary), flush=True)


def audit(out):
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["contract"]
    sources = {
        s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() == digest
        for s, digest in manifest["source_sha256"].items()
    }
    if not all(sources.values()):
        raise RuntimeError("Source hashes changed")
    checks = []
    for row in read_rows(out):
        case = make_case(row["seed"])
        state = np.load(out / row["state_file"])
        result = {key: state[key] for key in ("x", "z", "y", "w", "beta")}
        result.update(initial_weight=state["w0"], status=row["status"])
        checked = inspect_result(case, result, 1e-4)
        checks.append(
            dict(
                id=row["id"],
                validated=checked["validated"],
                residual=checked["audit"]["residual"],
                support_gap=checked["supporting_gap"],
                domination_gap=checked["domination_gap_sum"],
            )
        )
    save(
        out / "audit.json",
        dict(
            source_matches=sources,
            rows=len(checks),
            validated=sum(c["validated"] for c in checks),
            checks=checks,
        ),
    )
    print("AUDIT", len(checks), sum(c["validated"] for c in checks), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--stage", choices=["all", "run", "report", "audit"], default="all")
    args = parser.parse_args()
    if args.stage in ("all", "run"):
        run(args.output, args.repeats)
    if args.stage in ("all", "report"):
        summarize(args.output, args.repeats)
    if args.stage in ("all", "audit"):
        audit(args.output)


if __name__ == "__main__":
    main()
