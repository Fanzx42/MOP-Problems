# Validation and practical limits

## Numerical evidence

Version 0.3.0 adds [derivative and algorithm verification](gradient-validation.md):
independent differences for every benchmark, analytical Jacobian anchors,
exceptional-point checks, certified minimax/proximal subproblem comparisons,
BB curvature cases, acceleration recurrence, explicit failure/budget semantics,
and paired multistart experiments. See the linked record for executed checks.

- Analytical anchors for every implemented problem family, including classic
  objective values and both feasible/infeasible constrained points.
- DTLZ distance-minimum identities across 2, 3 and 5 objectives.
- WFG1–9 reference checks across 2, 3, 5 and 8 objectives, different partitions,
  bounds and transformation breakpoints; analytical anchors and invalid partitions.
- UF1–10 explicit scalar-oracle checks at n=5,7,30, plus n=3 for UF1–7,
  analytical linkage minima, boundary values and uneven index groups.
- Five C-DTLZ variants against pymoo at 2, 3, 5 and 13 objectives, k=1 and k=10;
  raw constraint signs, radius defaults and parameter validation.
- Omni-test and simple/rotated SYM-PART reference values and all nine known
  equivalent decision subsets at default dimensions.
- Welded-Beam objectives, raw residuals, feasible/infeasible anchors and the
  explicitly selected pymoo 0.6.2 variant.
- Exported experiment CSV reproducibility and feasibility, with nondominance
  checked against the exhaustive definition including ties and empty sets.
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
| C-DTLZ | No rescaling |
| Welded-Beam | Multiply columns by `(13600, 30000, 4.875, 6000)` |

TNK uses `atan2(x1, x2)` to define the box axes without division by zero; at the
infeasible origin the angle is set to zero. Random reference points avoid this
source-specific boundary convention. No other residual normalization is implicit.

SCH, FON and POL use analytical anchors and the documented MOEA Framework variant
cross-checks rather than a pymoo comparison, since that reference does not expose
these three problems. Numerical tests are evidence for implementation correctness,
not formal proofs of all analytical or Pareto properties. UF's scalar oracle
is an independent transcription, not a claim that PlatEMO was executed locally.

## Repository and distribution evidence

Tests verify the exact requested taxonomy, unique canonical names, coherent
implementation statuses, local documentation links, generated catalogue freshness,
the CLI, and package use outside the repository working directory.
The GitHub workflow also builds an sdist and wheel, checks distribution metadata,
and installs the wheel in a separate environment before smoke testing.

CI is configured for Linux on Python 3.10–3.14, and Windows/macOS on Python 3.12.
Those are configured checks, not a claim that the hosted matrix has already run.
The local verification environment and results for initial preparation are recorded
in [initial verification](initial-verification.md); current experiment reproduction
is documented in [the experiment guide](experiments.md).

## Limitations

Version 0.3.0 contains 47 executable problems and 28 planned catalogue entries.
There are no implemented dynamic or equality-constrained benchmark suites yet,
no Hessian API, and no bundled reference-front datasets. Forward Jacobians and
five first-order algorithm families are implemented. The algorithms handle boxes
and explicit proximable convex terms, not general nonlinear G/H constraints;
their derivative oracles remain available to external constrained methods.
Piecewise/singular formulas do not acquire smoothness merely because derivatives
are available on their regular branches. Stationarity is not global optimality.
Higher-dimensional front samplers are finite seeded samples rather than exact
uniform meshes. ZDT3 segment limits are numerical approximations. DTLZ5/6 front
sampling is restricted to M<=3; higher-dimensional g=0 curves can omit tradeoffs.

Planned real-world subsets require variable-type and source-version audits.
Do not treat a linked source, a directory name or a planned member name as proof
that the problem is callable or numerically validated.
