# Validation and practical limits

## Numerical evidence

- Analytical anchors for every implemented problem family, including classic
  objective values and both feasible/infeasible constrained points.
- DTLZ distance-minimum identities across 2, 3 and 5 objectives.
- ZDT3 nondominance/gap checks and the nonzero minimum of ZDT6's first objective.
- Optional cross-library comparisons with **pymoo 0.6.2** for all five continuous
  ZDT problems, DTLZ1–7 and BNH/TNK/OSY/SRN. ZDT dimensions include 2, 10 and 30;
  DTLZ tests include different objective counts, k=1 and a non-default DTLZ4 alpha.
- Random interior batches and decision-boundary checks, single-vector consistency,
  empty batches, reproducible seeds, and invalid input/parameter rejection.
- Generic equality residual/tolerance checks with a deliberately simple fixture.
  The fixture is not registered or presented as an implemented benchmark.

Reference comparisons account for declared differences:

| Problem | Convert pymoo G to this repository's raw G |
| --- | --- |
| BNH | Multiply columns by `(25, 7.7)` |
| TNK | Multiply columns by `(1, 0.5)` |
| OSY | Multiply columns by `(2, 6, 2, 2, 4, 4)` |
| SRN | No rescaling |

TNK uses `atan2(x1, x2)` to define the box axes without division by zero; at the
infeasible origin the angle is set to zero. Random reference points avoid this
source-specific boundary convention. No other residual normalization is implicit.

SCH, FON and POL use analytical anchors and the documented MOEA Framework variant
cross-checks rather than a pymoo comparison, since that reference does not expose
these three problems. Numerical tests are evidence for implementation correctness,
not formal proofs of all analytical or Pareto properties.

## Repository and distribution evidence

Tests verify the exact requested taxonomy, unique canonical names, coherent
implementation statuses, local documentation links, generated catalogue freshness,
the CLI, and package use outside the repository working directory.
The GitHub workflow also builds an sdist and wheel, checks distribution metadata,
and installs the wheel in a separate environment before smoke testing.

CI is configured for Linux on Python 3.10–3.14, and Windows/macOS on Python 3.12.
Those are configured checks, not a claim that the hosted matrix has already run.
The local verification environment and results for initial preparation are recorded
in [initial verification](initial-verification.md).

## Limitations

The initial release contains 19 executable problems and 34 planned entries.
There are no implemented dynamic or equality-constrained benchmark suites yet,
no gradient/Hessian API, no optimizer, and no bundled reference-front datasets.
Higher-dimensional front samplers are finite seeded samples rather than exact
uniform meshes. ZDT3 segment limits are numerical approximations.

Planned real-world subsets require variable-type and source-version audits.
Do not treat a linked source, a directory name or a planned member name as proof
that the problem is callable or numerically validated.
