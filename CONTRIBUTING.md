# Contributing

Use Python 3.10+ and install `python -m pip install -e ".[dev,reference]"`.
Small, independently verifiable changes are easier to review than a large
unverified port of a benchmark collection.

## Adding a problem

1. Identify the defining paper, exact variant and author/reference implementation.
   Record its version and any known errata. Check licensing before copying code.
2. Establish dimensions, variable types, bounds, original objective directions,
   inequalities/equalities, and time/difficulty parameters. A continuous relaxation
   of a discrete problem must receive a different identifier.
3. Implement a `Problem` subclass under `src/mop_problems/problems/`, following the
   [API contract](docs/api.md). Return vectorized `F`, raw `G <= 0` and raw `H = 0`.
4. Add independent numerical checks: hand-computed nontrivial values, feasible and
   infeasible points, boundary cases and, when available, a versioned reference
   comparison. Merely comparing batch and scalar paths does not validate a formula.
5. Update `src/mop_problems/data/catalog.json`, mathematical definitions and sources.
   Register only implementations that pass the checks. Keep planned families honest.
6. Run `python scripts/build_catalog.py`. Do not edit generated problem cards by hand.
7. Run the checks below and describe the final behavior and evidence in the PR.

```sh
python -m pytest --cov=mop_problems --cov-report=term-missing
python -m ruff check .
python -m ruff format --check .
python scripts/build_catalog.py --check
python -m build
```

Do not silently change a published benchmark's bounds, coefficients, constraint
normalization, objective signs or default dimensions. Document bug fixes and
behavior changes in `CHANGELOG.md`; retain a regression example.

## Reference fronts and reproducibility

Keep true/analytical fronts, approximate reference sets and arbitrary samples
distinct. Document the origin, generation method, resolution and license of any
reference data. Do not commit large experimental outputs or proprietary data.
Use explicit random seeds and, for future dynamic suites, explicit benchmark time.

The initial version is intentionally limited to real-valued static implementations.
Adding equality or dynamic benchmarks also requires tests of the associated
numerical/time conventions, not only an extra directory.
