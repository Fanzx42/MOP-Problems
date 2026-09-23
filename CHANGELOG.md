# Changelog

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
