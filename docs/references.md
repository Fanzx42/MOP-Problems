# References and provenance

The problem cards contain per-family sources and outstanding version questions.
Sources were consulted on 2026-09-23. This repository implements the formulas in
its own NumPy code; no external source files are bundled.

## Implemented definitions

- Zitzler, Deb and Thiele (2000), [Comparison of multiobjective evolutionary
  algorithms: Empirical results](https://doi.org/10.1162/106365600568202).
  [Authors' test-problem materials](https://sop.tik.ee.ethz.ch/download/supplementary/testProblemSuite/).
- Deb, Thiele, Laumanns and Zitzler, [Scalable Test Problems for Evolutionary
  Multiobjective Optimization](https://doi.org/10.1007/1-84628-137-7_6).
  [Authors' PISA DTLZ material](https://sop.tik.ee.ethz.ch/pisa/variators/dtlz/index.html)
  links the CEC 2002 paper and the expanded chapter.
- SCH, FON and POL: [MOEA Framework problem sources](https://github.com/MOEAFramework/MOEAFramework/tree/9f2ce5a4e4ae4e0842b4fa51d99be142af54d0e5/src/org/moeaframework/problem/misc).
  FON corresponds to `Fonseca2`. The SCH decision interval differs between
  implementations; we use `[-1000,1000]`. POL's positive minimization form is
  equivalent to maximizing the negatives in MOEA Framework. Original POL article
  details are still to be completed; they are not invented in the bibliography.
- BNH, TNK, OSY and SRN: [pymoo test problems](https://pymoo.org/problems/test_problems.html)
  and its [versioned source](https://github.com/anyoptimization/pymoo/tree/0.6.2/pymoo/problems/multi).
  Raw constraint scales in this repository are documented in [formulas](formulas.md).

## Planned suites and source policy

[PlatEMO](https://github.com/BIMK/PlatEMO) provides reference definitions for many
listed families. The catalogue pins consulted links to commit
`d25e65d1ffba58dbf4d7e1b5259786187d12968a`; paper references are recorded from its
reference sections. This is provenance, not a redistribution of PlatEMO code.

[Tanabe and Ishibuchi's original RE/CRE repository](https://github.com/ryojitanabe/reproblems)
documents corrections to engineering formulas. Continuous membership and the
specific corrected version must be established before implementation.
[jMetalPy's problem documentation](https://jmetal.github.io/jMetalPy/api/problems/)
identifies the RWA engineering models and original objective directions.

The [equality benchmark dataset](https://doi.org/10.1016/j.dib.2020.105130)
identifies Eq-DTLZ1–4 and Eq-IDTLZ1–2. It must not be expanded with unverified
variants under the same version label.

NCTP and CMMOP still need original-source verification. Their folders are present
and explicitly marked pending, with no guessed formulas or source attribution.

## License and citation policy

The MIT license applies to this repository's original code and documentation.
External libraries, papers and data retain their own licenses. A public source
link is not permission to copy and relicense code. Future ports must identify
their provenance and satisfy the source license; independent formula implementations
should cite the defining publication and any cross-check implementation.

Use [references.bib](references.bib) for the central ZDT/DTLZ and validation-library
citations; use each problem card's original paper for other suites. Cite this
software using [CITATION.cff](../CITATION.cff) after adding the actual repository
URL during publication. This file does not claim a software DOI or published paper.
