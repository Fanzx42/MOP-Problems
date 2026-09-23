# Python API and numerical contract

```python
from mop_problems import (
    get_problem,
    get_problem_info,
    get_suite,
    list_problems,
    list_suites,
    catalog,
    Problem,
    Evaluation,
    PlannedProblemError,
)
```

`get_problem(name, **parameters)` constructs a fresh problem. Names are case
insensitive; spaces, underscores and hyphens are ignored when matching.
`get_problem("zdt-1", n_var=12)` therefore selects `ZDT1`. Misspellings raise
`KeyError`; known planned entries raise `PlannedProblemError`. No surrogate,
fallback or placeholder objective is substituted. A planned suite whose membership
is still unverified is discoverable by suite name; arbitrary numbered members are
not automatically accepted.

`list_problems(suite=None, category=None)` returns **implemented** names only.
`list_suites(category=None, status=None)` returns all metadata, including planned
families. A category can be a full path or a prefix, e.g. `constrained`.
`get_problem_info(name)` describes default dimensions, bounds and parameters;
inspect the constructed instance for dimensions after parameter overrides.
Catalogue functions return detached copies.

## Dimensions and bounds

| Property | Meaning |
| --- | --- |
| `name` | Canonical benchmark identifier |
| `n_var` | Number of real decision variables |
| `n_obj` | Number of objectives (all minimized) |
| `xl`, `xu` | Read-only lower/upper bound arrays of shape `(n_var,)` |
| `n_ieq_constr` | Number of explicit inequalities |
| `n_eq_constr` | Number of explicit equalities |

ZDT accepts `n_var >= 2`; FON accepts `n_var >= 2`. DTLZ accepts `n_obj >= 2`
and `n_var >= n_obj`. Omitting `n_var` gives `n_obj + k - 1`, with `k=5` for
DTLZ1, `k=20` for DTLZ7, and `k=10` otherwise. DTLZ4 also accepts finite
`alpha > 0`, default 100. Other implemented classics have fixed dimensions.
Non-integer dimensions and unknown keyword parameters are rejected.

| Additional family | Parameters and restrictions |
| --- | --- |
| WFG1–9 | `n_obj>=2`, positive `k` divisible by `n_obj-1`, `n_var>k`; WFG2/3 require even `n_var-k`. Default `n_obj=3`, `k=4`, `n_var=24`. With changed objectives, default `k=4` for 2 objectives, otherwise `2*(n_obj-1)`; default distance count stays 20. |
| UF1–7 | `n_var>=3`, default 30; fixed 2 objectives |
| UF8–10 | `n_var>=5`, default 30; fixed 3 objectives |
| Omni-test | `n_var>=2`, default 2; fixed 2 objectives |
| SYM-PART | Fixed 2 variables and 2 objectives |
| SYM-PART-Rotated | Fixed dimensions; finite `angle` in radians, default `pi/4` |
| C-DTLZ | `n_obj>=2`, `n_var>=n_obj`; default `n_var=n_obj+9`. C1-DTLZ3/C2-DTLZ2 accept positive `r`; C3-DTLZ4 accepts positive `alpha`. |
| Welded-Beam | Fixed 4 variables, 2 objectives and 4 inequalities |

```python
problem = get_problem("WFG9", n_var=34, n_obj=8, k=14)
result = problem.evaluate(problem.sample(100, seed=42))
constrained = get_problem("C2-DTLZ2", n_obj=5, r=0.5)
```

## Evaluation

`problem.evaluate(X, *, check_bounds=True, equality_tol=1e-6)` accepts `(n_var,)`
or `(N, n_var)` input. It returns an `Evaluation` whose numerical arrays are
read-only copies:

| Field | Shape | Interpretation |
| --- | --- | --- |
| `F` | `(N, n_obj)` | Objective values |
| `G` | `(N, n_ieq_constr)` | Inequality residuals; feasible if all `<= 0`; LSCMOP uses author aggregate |
| `H` | `(N, n_eq_constr)` | Raw equality residuals; feasible within tolerance |
| `CV` | `(N,)` | Total constraint and bound violation |
| `feasible` | `(N,)` | Boolean `CV == 0` |

`N=0` is supported as an explicitly shaped array. A single input vector retains
the batch axis. Missing constraints are represented by `(N, 0)` arrays, not `None`.

For each row the violation is

$$
CV=\sum_i\max(G_i,0)+\sum_j\max(|H_j|-\epsilon,0)
 +\sum_k\bigl[\max(l_k-x_k,0)+\max(x_k-u_k,0)\bigr].
$$

Inequalities have no hidden tolerance. Equality tolerance must be finite and
nonnegative; it affects feasibility and `CV`, never the returned `H`.
Residuals are not normalized, so their scale and `CV` are problem dependent.
Comparing `CV` magnitudes across different problems is not meaningful without an
explicit scaling policy.

Out-of-bound decisions raise `ValueError` by default. `check_bounds=False` permits
evaluating a formula outside its nominal box **only if that formula is defined
there**; bound violations still contribute to `CV`. Non-finite inputs are always
rejected; non-finite outputs raise `FloatingPointError`. Inputs are not clipped,
modified or automatically normalized. Problem configuration should be treated as
fixed after construction.

## Sampling and Pareto fronts

`problem.sample(n, seed=None)` samples uniformly from the decision box using
NumPy's `default_rng`; pass an integer or an existing `Generator`. No global RNG
state is changed. This is **not** a feasible-region sampler. MaF9 also excludes its invalid polygons.

`problem.pareto_front(n_points=100, seed=0)` returns objective-space samples where
an analytical sampler is supported. It returns exactly the requested number of
points (`n_points >= 2`). ZDT3 uses numerically known segment endpoints; small
sample sizes need not cover all five segments. It must be plotted as separate
segments or scatter points. For higher-dimensional DTLZ1–4, the sampler uses
seeded simplex/sphere sampling and includes coordinate extremes when space permits;
it is not an evenly spaced grid. DTLZ5/6 sample the degenerate curve only for
2 or 3 objectives. For more objectives the g=0 curve can omit tradeoffs, so
front sampling raises `NotImplementedError` while objective evaluation remains supported.

UF1/2/3/4/7 provide two-dimensional front curves; UF8/10 provide seeded
nonnegative unit-sphere samples. Omni-test and both SYM-PART variants return
objective-space fronts, not all of their decision-space Pareto subsets.
See [the experiment example](../examples/representative_suite.py) for explicit
construction of all nine subsets at default dimensions.

DTLZ7, POL, WFG1–9, UF5/6/9 and all constrained problems raise
`NotImplementedError` for reference-front sampling. Inspect the `pareto_front`
flag from `get_problem_info()` (which describes default parameters) and the
parameter-dependent restrictions above before requesting one. The existence of an
evaluation formula does not imply the existence of a verified reference sampler.

## Extending the library

First derivatives use `problem.jacobian(X)` or `problem.evaluate_with_jacobian(X)`.
`dF`, `dG`, `dH` have shape `(N, n_functions, n_var)` and retain raw constraint signs.
Read [the derivative contract](gradients.md) for smoothness and exceptional points,
and [the algorithm guide](algorithms.md) for the five built-in first-order families.

Subclass `Problem` and implement `_evaluate(X) -> (F, G, H)`. It receives a
validated batch and must not mutate it. Return `None` for a constraint family only
when its count is zero. The base class checks all result shapes and finite values.
See [contribution requirements](../CONTRIBUTING.md) before registration.

The finite NumPy operation subset used by the registered formulas supports forward
chain rules. A custom problem may override `evaluate_with_jacobian` with its own
validated `(Evaluation, Jacobian)` implementation; unsupported AD operations fail
explicitly. Unknown custom smoothness is reported as `unknown`.

## Expanded families, time and sources

All 43 catalogue entries are implemented in 0.4.0; `planned` filtering remains
available for future additions. `get_problem_info(name)["references"]` lists the
original papers, and `["reference_implementation"]` identifies the comparison
source and version. Metadata describes **default** parameters; record overrides
alongside it in an experiment manifest.

Eq-DTLZ1–4 and Eq-IDTLZ1/2 accept `n_equalities=1` or `2` and return signed `H`.
RCM also contains genuine equality-constrained engineering models.
DOC retains its published tolerance bands as inequalities. LSCMOP preserves the
author's aggregate positive violation in its single `G` column, including cutoffs;
it does not expose the underlying individual residuals.

FDA1–5 and JY1–8 accept `time=0.0`. `p.at_time(t)` creates another fixed snapshot;
evaluation does not advance time. `benchmark_time(step, frequency, severity)` in
`mop_problems.problems.composite.fda` computes the standard discrete schedule.
Jacobians differentiate decisions at fixed time. MaF9's `domain_mask(X)` reports
excluded polygon regions (separately from box membership); `evaluate` rejects them
even with `check_bounds=False`, and `sample` uses reproducible rejection sampling.

See [parameters and examples](expanded-benchmarks.md) and
[source-specific differences](expanded-validation.md) before combining implementations.
