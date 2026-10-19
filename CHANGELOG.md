# Changelog

## Unreleased

- Add a standalone four-objective synthetic Gamma Knife application with explicit
  dose/time splitting, weighted dual ADMM, multiobjective ADMM and joint MO-ALM.
- Cache numerical operators and QP objectives; add bounded, finite residual-based
  penalty adaptation with common-weight KKT and weight-consensus checks.
- Add optional SciPy/COPT dependencies, runnable examples, independent LP/solver
  tests, and compact archived measurements and states; no patient data is bundled.
- Preserve the 343-problem benchmark registry and the NumPy-only core installation.

## 0.4.0 — 2026-09-23

- Complete all 28 planned families: 296 new benchmarks, 343 callable problems in all 43 catalogue entries.
- Add per-problem papers, reference-code versions, domain and variant notes, and a generated bibliography index.
- Add explicit FDA/JY time snapshots, signed Eq-DTLZ/IDTLZ equalities, large-scale, multimodal and real engineering suites.
- Provide forward Jacobians at regular points; scale finite differences to small engineering variable ranges.
- Add 482 independent reference cases covering every new problem, domain/parameter checks and reproducible CSV/JSON/plot examples.
- Preserve source-specific variants explicitly, including the MaF10 continuous transform, LSCM aggregate violation and RCM5/14 continuous relaxations; fix RCM50 batch dependence.
- Include required MIT notices for RE/CRE and RWA adaptations.

## 0.3.0 — 2026-09-23

- Add NumPy-only forward Jacobians for all 47 implemented benchmarks, including
  raw inequality/equality derivatives, fused evaluation, conservative branch checks,
  explicit finite differences, independent derivative checks and smoothness metadata.
- Add MGDA/GD, per-objective BBDMO, multiobjective proximal gradient, and the
  smooth/composite TFY Nesterov-style accelerated methods with objective offsets.
- Support fixed, Armijo, max/average nonmonotone gradient steps; fixed/model-backtracking
  proximal steps; box-aware minimax models and distinct shifted weighted L1 penalties.
- Certify inner solves by primal-dual gaps; unify stationarity diagnostics and record
  accepted steps, BB scales, value/Jacobian/prox costs, restarts, timings and failures.
- Add paired multistart experiments, raw NPZ/JSON/CSV exports and scientific plots;
  document exact paper variants, boundary behavior, nonlinear-constraint limitations,
  reproducibility and the difference between stationarity and Pareto optimality.
- Add optional SciPy algorithms dependency, analytical/independent solver tests,
  derivative-enabled distribution checks and gradient-example CI smoke tests.

## 0.2.0 — 2026-09-23

- Add 28 benchmarks: WFG1–9, UF1–10, five C-DTLZ variants, Omni-test,
  simple/rotated SYM-PART and the specified four-constraint Welded-Beam variant.
- Reach 47 implemented problems across 15 of the 43 catalogue entries;
  the remaining 28 entries stay explicitly planned.
- Add analytical front samplers for UF1/2/3/4/7/8/10, Omni-test and both SYM-PART variants.
- Add 14 reproducible sampling cases, equivalent decision-set constructions,
  constrained-surface analysis, raw CSV exports, environment manifests and three figures.
- Extend analytical, scalar-oracle and pinned pymoo comparisons, test the example's
  dominance/feasibility/reproducibility, and add a plotting smoke job in CI.
- Document all new equations, variant differences and defaults; set the actual GitHub URLs.
- Limit DTLZ5/6 analytical front sampling to 2 or 3 objectives: their g=0 curve
  alone can omit tradeoffs at higher dimensions. Objective evaluation remains available.

## 0.1.0 — 2026-09-23

- Establish the full 43-entry taxonomy supplied for MOP-Problems.
- Implement ZDT1/2/3/4/6, DTLZ1–7, SCH, FON, POL, BNH, TNK, OSY and SRN.
- Add vectorized evaluation, strict dimensions/bounds, raw constraints, explicit
  equality tolerance, seeded box sampling and selected analytical front samplers.
- Add machine-readable metadata, generated problem cards, English/Chinese READMEs,
  formulas, references, examples, contribution guides and GitHub community files.
- Add analytical and optional pymoo 0.6.2 numerical checks, catalogue consistency,
  CLI tests, CI matrix and isolated wheel verification.
- Mark the remaining 34 entries as planned; no unsupported formulas are registered.
