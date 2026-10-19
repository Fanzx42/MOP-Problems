# ruff: noqa: E402 -- set BLAS thread counts before importing NumPy
"""Run the standalone synthetic Gamma Knife application after installing the extra."""

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from pathlib import Path

from mop_problems.applications.radiosurgery._io import save
from mop_problems.applications.radiosurgery.experiment import audit_reference, run_small


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/radiosurgery"))
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--start", type=int, choices=(0, 1), default=0)
    parser.add_argument("--weights", type=int, nargs="+", default=[0, 4, 8])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("scalarized", "admm", "alm"),
        default=["scalarized", "admm", "alm"],
    )
    parser.add_argument(
        "--audit-reference",
        type=Path,
        help="Recheck one archived docs/experiments/radiosurgery dataset",
    )
    args = parser.parse_args()
    if args.audit_reference:
        audit = audit_reference(args.audit_reference)
        save(args.output / "audit.json", audit)
        print("Independent reference audit:", audit["validated"], "/", audit["rows"])
        if audit["validated"] != audit["rows"]:
            raise SystemExit(1)
    else:
        rows = run_small(
            args.output, args.seed, args.start, tuple(args.weights), tuple(args.methods)
        )
        if not all(row["validated"] for row in rows):
            raise SystemExit("At least one solver output failed its numerical audit; see runs.json")


if __name__ == "__main__":
    main()
