"""Small three-method comparison and independent physical-model audits."""

import numpy as np

from .oracle import CachedOracle


def inspect_result(case, result, tol):
    p = case.problem
    o = CachedOracle(p)
    try:
        cert = o.certificate(result["x"], result["z"], result["y"], result["w"])
    finally:
        o.close()
    consensus = float(np.max(np.abs(result["w"] - result["beta"])))
    cert.update(consensus=consensus, residual=max(cert["kkt"], consensus))
    physical, dose = case.physical(result["x"])
    obj = case.objectives(physical)
    ref = case.lp(result["w"])
    gap = float(result["w"] @ obj - ref["value"])
    dom = case.lp(np.ones(4), caps=obj, cap_slack=1e-9)
    dominance = max(0.0, float(obj.sum() - dom["objectives"].sum()))
    checked = (
        result["status"] == "converged"
        and cert["residual"] <= tol * 1.05 + 1e-8
        and -1e-8 <= gap / (1 + abs(ref["value"])) <= 5 * tol
        and p.violation(physical, dose) <= 1e-9
    )
    return dict(
        audit=cert,
        validated=bool(checked),
        raw_objectives=p.values(result["x"], result["z"]),
        recovered_objectives=obj,
        reference_value=ref["value"],
        supporting_gap=gap,
        relative_supporting_gap=gap / (1 + abs(ref["value"])),
        physical_recovery_distance=float(np.max(np.abs(physical - result["x"]))),
        domination_gap_sum=dominance,
        materially_dominated=dominance > 1e-3,
        dominance_witness_objectives=dom["objectives"],
        dominance_witness_x=dom["x"],
        final_weight=result["w"],
        weight_drift=float(np.linalg.norm(result["w"] - result["initial_weight"], 1)),
    )


def run_small(output, seed=202, start=0, indices=(0, 4, 8), methods=("scalarized", "admm", "alm")):
    """Run paper weighted ADMM, numerical MO-ADMM, and the joint MO-ALM baseline.

    All methods use the same synthetic dose map. MO weights adapt whereas paper
    weights stay fixed. Save raw states even when a quality audit fails.
    """
    import hashlib
    from pathlib import Path

    from ._engine import solve_alm
    from ._io import save
    from .model import initial, make_case, weight_grid
    from .numerics import solve_numeric_admm
    from .scalarized import solve_batch

    output = Path(output)
    if (output / "manifest.json").exists():
        raise FileExistsError("Use a new output directory for each three-method comparison")
    if not indices or any(index not in range(9) for index in indices):
        raise ValueError("Weight indices must be nonempty and between 0 and 8")
    if any(method not in {"scalarized", "admm", "alm"} for method in methods):
        raise ValueError("Unknown method")
    case = make_case(seed)
    x, z = initial(case, start)
    weights = weight_grid()[list(indices)]
    source = Path(__file__).resolve().parent
    save(
        output / "manifest.json",
        dict(
            seed=seed,
            start=start,
            weight_indices=list(indices),
            weights=weights,
            methods=list(methods),
            data="Synthetic dose maps; not clinical data",
            tolerance=1e-4,
            threads=1,
            source_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(source.glob("*.py"))
            },
            timing="Includes setup and solver checks; excludes independent post-run audits. "
            "Scalarized solver batches weights; MO methods solve them sequentially.",
            preferences="Equal initial/fixed weights do not imply equal final preferences.",
        ),
    )
    rows = []
    for method in methods:
        if method == "scalarized":
            results = solve_batch(case, weights, x, penalty=3.0, seconds_limit=60.0)
        elif method == "admm":
            results = [
                solve_numeric_admm(case.problem, x, z, w, penalty=1.0, seconds_limit=60.0)
                for w in weights
            ]
        else:
            results = [
                solve_alm(
                    case.problem,
                    x,
                    z,
                    w,
                    penalty=3.0,
                    seconds_limit=60.0,
                    oracle_factory=CachedOracle,
                )
                for w in weights
            ]
        for index, w, result in zip(indices, weights, results, strict=True):
            result["initial_weight"] = w
            audit = inspect_result(case, result, 1e-4)
            rid = f"{method}_w{index}"
            state = output / "states" / f"{rid}.npz"
            state.parent.mkdir(exist_ok=True)
            np.savez_compressed(
                state, **{k: result[k] for k in ("x", "z", "y", "w", "beta")}, x0=x, z0=z, w0=w
            )
            save(output / "histories" / f"{rid}.json", result["history"])
            row = dict(
                id=rid,
                method=method,
                weight_index=index,
                initial_weight=w,
                seconds=result["seconds"],
                iterations=result["iterations"],
                status=result["status"],
                reason=result.get("reason"),
                state_file=f"states/{rid}.npz",
                **audit,
            )
            rows.append(row)
            print(
                rid,
                row["status"],
                "validated=" + str(row["validated"]),
                "dominated=" + str(row["materially_dominated"]),
                flush=True,
            )
    save(output / "runs.json", rows)
    return rows


def audit_reference(directory):
    """Re-audit the compact archived states with the portable implementation."""
    import json
    from pathlib import Path

    from .model import make_case

    directory = Path(directory)
    metadata = json.loads((directory / "runs.json").read_text(encoding="utf-8"))
    states = np.load(directory / "states.npz", allow_pickle=False)
    cases = {seed: make_case(seed) for seed in {row["seed"] for row in metadata}}
    checks = []
    for i, row in enumerate(metadata):
        result = {key: states[key][i] for key in ("x", "z", "y", "w", "beta", "initial_weight")}
        result["status"] = row["status"]
        audit = inspect_result(cases[row["seed"]], result, 1e-4)
        checks.append(
            dict(
                id=row["id"],
                validated=audit["validated"],
                residual=audit["audit"]["residual"],
                materially_dominated=audit["materially_dominated"],
                domination_gap_sum=audit["domination_gap_sum"],
            )
        )
    return dict(rows=len(checks), validated=sum(row["validated"] for row in checks), checks=checks)
