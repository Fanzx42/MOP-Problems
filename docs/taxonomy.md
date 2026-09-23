# Taxonomy, variants and scope

The directory hierarchy follows the supplied list. It is an organizational
taxonomy, not a proof that every member has identical mathematical properties.

- `unconstrained_smooth`: traditional continuous unconstrained problems. Fractional
  powers in ZDT and DTLZ6 can fail to be differentiable at boundaries.
- `composite`: transformed, linked, irregular, large-scale, multimodal, dynamic
  and real-world problem families. “Composite” is a grouping label here.
- `constrained`: explicit inequalities/equalities in addition to variable bounds.

The [catalogue](../problems/README.md) contains every supplied family, including
families that are not implemented. Counts of catalogue entries, suites and actual
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
| MMF | Initial original MMF1–8 target; later extensions require version metadata |
| SYM-PART | Simple/rotated/distorted variants require distinct identifiers |
| Eq-IDTLZ | Original dataset's 1–2; later extensions need separate references |
| LSCMOP | Source uses `LSCM`; verify mapping before registering members |
| CF, ZXH_CF | Distinct benchmark families |
| CMMOP | Distinguish the named suite from CMMF and from the generic problem class |

## Continuous real-world subsets

`RE_continuous` and `RCM_continuous` deliberately have no fabricated member list.
For each candidate, inspect the original variables and constraints. A continuous
relaxation of an integer/discrete problem is a different benchmark and needs a
different name. Record excluded members and the reason when finalizing each subset.
Audit stochastic outputs as well as variable types. The full CRE suite also
contains non-continuous members; do not register those in a real-only API.

Mixed minimization/maximization models (e.g. RWA) must document their original
objective directions and the transformation to the all-minimization API.

## Status semantics

`planned` means a catalogue record, source notes and a contribution target.
`implemented` means an actual registered formula with contract and numerical tests.
The source status (`identified`, `pending`, `membership-pending`) is independent:
finding a paper or implementation does not certify its formulas or numerical results.

No runtime entry is generated just because a name appears in a directory.
No derivatives, Hessians, optimization algorithms, dynamic-time interface or external
reference-front datasets are claimed by version 0.1.0.
