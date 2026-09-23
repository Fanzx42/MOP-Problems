# Taxonomy, variants and scope

The directory hierarchy follows the supplied list. It is an organizational
taxonomy, not a proof that every member has identical mathematical properties.

- `unconstrained_smooth`: traditional continuous unconstrained problems. Fractional
  powers in ZDT and DTLZ6 can fail to be differentiable at boundaries.
- `composite`: transformed, linked, irregular, large-scale, multimodal, dynamic
  and real-world problem families. “Composite” is a grouping label here.
- `constrained`: explicit inequalities/equalities in addition to variable bounds.

The [catalogue](../problems/README.md) contains every supplied family, all implemented in 0.4.0. Counts of catalogue entries, suites and actual
callable instances are kept separate. `ZDT` and `SCH`, for example, each occupy one
catalogue entry but contain five and one implemented instances respectively.

## Variants that must remain explicit

| Label | Policy |
| --- | --- |
| ZDT | Real-valued ZDT1/2/3/4/6; binary ZDT5 excluded |
| DTLZ | Unconstrained DTLZ1–7; constrained DTLZ8/9 are not silently added here |
| SCH | Schaffer N.1, with the declared decision interval |
| FON | Scalable Fonseca–Fleming (MOEA Framework `Fonseca2`) |
| POL | Positive minimization values, not negated maximization values |
| IDTLZ, SDTLZ | Named inversion/scaling variants and their parameters |
| MMF | Original MMF1–8; later extensions require version metadata |
| SYM-PART | `SYM-PART` is simple; `SYM-PART-Rotated` rotates by pi/4 by default; distortion is excluded |
| WFG | All nine, with an explicit position/distance partition and even distance count in WFG2/3 |
| UF | CEC 2009 UF1–10, fixed competition constants and one-based parity/remainder sets |
| C-DTLZ | Exactly C1-DTLZ1/3, C2-DTLZ2 and C3-DTLZ1/4; all default to n=M+9 |
| Welded-Beam | The four-constraint pymoo 0.6.2 variant; J coefficient differs from some engineering versions |
| Eq-IDTLZ | Original dataset's 1–2; later extensions need separate references |
| LSCMOP | LSCMOP1–12 map exactly to author LSCM1–12; aliases accepted |
| CF, ZXH_CF | Distinct benchmark families |
| CMMOP | Distinguish the named suite from CMMF and from the generic problem class |

## Continuous real-world subsets

`RE_continuous` contains RE21/24/31/32/33/34/37/41/42/61.
Discrete or rounded RE22/23/25/35/36 and stochastic RE91 are excluded.
CRE21/22/23/31/32/51 are included; CRE24/25 are discrete and excluded.

`RCM_continuous` preserves author numbering: 2/3/4/5/8/9/10/11/12/14/16/17/18/20/21/22/23/24/30–50.
RCM5/14 are the **author-code continuous relaxations of disk counts**, explicitly
tagged `continuous-relaxation`. These are not mixed-integer manufacturing models.
Integer/discrete 1/6/7/13/15/19/25–29 are excluded, even when the author's evaluator
omits a rounding operation. No missing numbers are silently reassigned.

RWA1–10 follow the pinned jMetalPy equations. Maximized responses are negated;
per-problem notes give original response names and signs. RE's published violation
objectives remain objectives. CRE returns the signed residuals before clipping.
Engineering suite papers describe the selected benchmark versions; references do
not assert that every historical application model used identical equations.

## Status semantics

`planned` means a catalogue record, source notes and a contribution target.
`implemented` means an actual registered formula with contract and numerical tests.
The source status (`identified`, `pending`, `membership-pending`) is independent:
finding a paper or implementation does not certify its formulas or numerical results.

No runtime entry is generated just because a name appears in a directory.
Version 0.4.0 provides first derivatives and five first-order algorithm families;
see [derivative caveats](gradients.md) and [algorithm scope](algorithms.md).
Explicit dynamic snapshots are provided; no Hessians or external reference-front datasets are claimed.
The reproducible examples generate their analytical reference points locally from
documented formulas. A catalogue category does not certify smoothness or convexity.
