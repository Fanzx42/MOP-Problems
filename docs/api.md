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

## Evaluation

`problem.evaluate(X, *, check_bounds=True, equality_tol=1e-6)` accepts `(n_var,)`
or `(N, n_var)` input. It returns an `Evaluation` whose numerical arrays are
read-only copies:

| Field | Shape | Interpretation |
| --- | --- | --- |
| `F` | `(N, n_obj)` | Objective values |
| `G` | `(N, n_ieq_constr)` | Raw inequality residuals; feasible if all `<= 0` |
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
state is changed. This is **not** a feasible-region sampler.

`problem.pareto_front(n_points=100, seed=0)` returns objective-space samples where
an analytical sampler is supported. It returns exactly the requested number of
points (`n_points >= 2`). ZDT3 uses numerically known segment endpoints; small
sample sizes need not cover all five segments. It must be plotted as separate
segments or scatter points. For higher-dimensional DTLZ1–4, the sampler uses
seeded simplex/sphere sampling and includes coordinate extremes when space permits;
it is not an evenly spaced grid. DTLZ5/6 sample the degenerate curve.

DTLZ7, POL and the four constrained classics raise `NotImplementedError` for
reference-front sampling. The existence of an evaluation formula does not imply
the existence of a verified reference front.

## Extending the library

Subclass `Problem` and implement `_evaluate(X) -> (F, G, H)`. It receives a
validated batch and must not mutate it. Return `None` for a constraint family only
when its count is zero. The base class checks all result shapes and finite values.
See [contribution requirements](../CONTRIBUTING.md) before registration.

Equality result handling is present in the base API and tested with an analytical
test fixture. **No real equality-constrained benchmark is implemented in 0.1.0.**
Dynamic suites are also planned: a future dynamic API needs explicit benchmark
time and reproducible environment parameters; it is not currently provided.
