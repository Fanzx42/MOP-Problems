# ADMM application catalogue

[完整中文模型与文献](admm-applications.zh-CN.md) ·
[Structured index](admm-applications.json) · [BibTeX](admm-applications.bib)

Verified on **2026-10-18**. This research supplement collects **12 application models**
with a two-block interpretation
$F_i(x,z)=f_i(x)+g_i(z)$, $Ax+Bz=c$, retaining local domains explicitly.

The seven A entries explicitly discuss multiple objectives and ADMM in the source.
The five B entries are **catalogue vectorizations of application loss/tradeoff terms**;
they do not claim that the source implements a direct multiobjective ADMM.
The Chinese cards specify source locations, actual scalarization, mathematical split,
convexity qualifications, data availability and what remains unverified.

| ID | Application | Objectives | Main qualification |
| --- | --- | --- | --- |
| A01 | Residential batteries | Load shaping; flexibility | Convex with local battery domains |
| A02 | Hybrid AC/DC distribution | Loss; voltage deviation; regulation cost | Cone relaxation and discrete device choices |
| A03 | Integrated energy park | Operating cost; carbon | Device constraints need a convexity audit |
| A04 | Networked microgrids | Economic and network objectives | Institutional abstract only; split is a candidate template |
| A05 | Gamma Knife planning | Target dose; healthy tissue dose; time | Fixed geometry/sampling LP; source solves a dual split |
| A06 | Urban traffic MPC | Admitted inflow; internal traffic quality | Lexicographic priorities, not an ordinary Pareto weight sweep |
| A07 | IPM motor design | Negative torque; iron loss | Mixed-variable nonconvex heuristic |
| B01 | Sparse portfolio | Risk; negative return | Nonconvex cardinality; budget penalty differs from equality |
| B02 | TV image restoration | Fidelity; total variation | Convex nonsmooth vector extension |
| B03 | Robust PCA | Nuclear norm; entrywise L1 | Convex nonsmooth vector extension |
| B04 | Hyperspectral unmixing | Fidelity; abundance L1; spatial TV | Fixed dictionary; sum-to-one can trivialize L1 |
| B05 | Portfolio with turnover | Risk; negative return; turnover | Simplified single-period convex vector extension |

These are literature/model cards, **not registered callable problems**. The existing
343-problem count and generated benchmark catalogue remain unchanged.
No clinical, image, hyperspectral or financial dataset is redistributed.

A05 now also has a [runnable four-objective synthetic application](radiosurgery.md),
including source-specific weighted ADMM, MO-ADMM/MO-ALM comparisons and numerical
implementation/penalty ablations. It is an optional application module outside
the benchmark registry and does not redistribute clinical dose matrices.

A separable convex model does not by itself satisfy an adaptive-weight theorem
requiring every objective to have bounded sublevel sets on the full product space.
Zero block costs, linear return terms and hard local domains need separate analysis.
Changing them by regularization, relaxation or penalties creates a different variant.
Fixed-weight scalar ADMM and a direct vector-valued algorithm must also be distinguished.

The NumPy-only construction checks run with:

~~~sh
python examples/admm_application_splits.py
~~~

Four hand-computable synthetic fixtures check TV, robust PCA, hyperspectral unmixing
and portfolio turnover. They verify objective decompositions and affine residuals,
including infeasible perturbations. They do not run an optimizer, reproduce source
experiments, establish Pareto optimality or benchmark ADMM speed.
