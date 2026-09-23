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
- WFG1–9: Huband, Hingston, Barone and While (2006),
  [A Review of Multiobjective Test Problems and a Scalable Test Problem Toolkit](https://doi.org/10.1109/TEVC.2005.861417).
  Transformation stages are compared against
  [pymoo 0.6.2 WFG](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/problems/many/wfg.py).
- UF1–10: Zhang, Zhou, Zhao, Suganthan, Liu and Tiwari (2009),
  [CEC 2009 multiobjective test instances, report CES-487](https://al-roomi.org/multimedia/CEC_Database/CEC2009/MultiObjectiveEA/CEC2009_MultiObjectiveEA_TechnicalReport.pdf).
  One-based groups and constants were checked against
  [pinned PlatEMO UF definitions](https://github.com/BIMK/PlatEMO/tree/d25e65d1ffba58dbf4d7e1b5259786187d12968a/PlatEMO/Problems/Multi-objective%20optimization/UF).
  The UF numerical oracle is a separate scalar transcription of the report,
  plus analytical linkage minima; it is not an executed MATLAB comparison.
- C-DTLZ: Jain and Deb (2014),
  [NSGA-III Part II: Handling Constraints and Extending to an Adaptive Approach](https://doi.org/10.1109/TEVC.2013.2281534).
  Named C1/C2/C3 formulas and radius conventions are compared with
  [pymoo 0.6.2 constrained DTLZ](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/problems/many/cdtlz.py).
  This repository explicitly sets default n=M+9 for all five variants.
- Omni-test: Deb and Tiwari (2008),
  [Omni-optimizer](https://doi.org/10.1016/j.ejor.2006.06.042), using the
  [pymoo 0.6.2 OmniTest convention](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/problems/multi/omnitest.py).
- SYM-PART: Rudolph, Naujoks and Preuss (2007),
  [Capabilities of EMOA to Detect and Preserve Equivalent Pareto Subsets](https://doi.org/10.1007/978-3-540-70928-2_7).
  The simple and rotated variants use the fixed parameters and decision box in
  [pymoo 0.6.2](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/problems/multi/sympart.py).
  No rotated-distorted variant is included.
- Welded-Beam: the exact version is the
  [pymoo 0.6.2 two-objective formulation](https://github.com/anyoptimization/pymoo/blob/0.6.2/pymoo/problems/multi/welded_beam.py).
  Its J coefficient, engineering constants, decision bounds and four residual
  scales are specified in [the formula card](formulas.md#welded-beam).
  Other welded-beam formulations can have different stresses and feasible regions.

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

## First-order algorithms

- Fliege, J.; Svaiter, B. F. (2000). Steepest descent methods for multicriteria
  optimization. Mathematical Methods of Operations Research 51, 479–494.
  [DOI](https://doi.org/10.1007/s001860000043).
- Désidéri, J.-A. (2012). Multiple-gradient descent algorithm (MGDA) for
  multiobjective optimization. Comptes Rendus Mathématique 350, 313–318.
  [Author's site](https://mgda.inria.fr/mgda),
  [paper](https://www.numdam.org/item/CRMATH_2012__350_5-6_313_0.pdf).
- Chen, J.; Tang, L.; Yang, X. (2023). A Barzilai-Borwein descent method for
  multiobjective optimization problems. European Journal of Operational Research
  311, 196–209. [Publisher](https://www.sciencedirect.com/science/article/pii/S0377221723003028).
  Implemented equations were checked against the
  [authors' arXiv:2204.08616v1](https://arxiv.org/abs/2204.08616v1),
  Eq. (22), Algorithm 5 and its monotone/max/average line searches.
- Tanabe, H.; Fukuda, E. H.; Yamashita, N. (2019). Proximal gradient methods for
  multiobjective optimization and their applications. Computational Optimization
  and Applications 72, 339–361. [DOI](https://doi.org/10.1007/s10589-018-0043-x).
- Tanabe, H.; Fukuda, E. H.; Yamashita, N. (2023). An accelerated proximal gradient
  method for multiobjective optimization. Computational Optimization and
  Applications 86, 421–455. [DOI](https://doi.org/10.1007/s10589-023-00497-w).
  The implemented minimax model, recurrence and dual follow
  [arXiv:2202.10994v3](https://arxiv.org/html/2202.10994v3), Sections 3, 4 and 6.
  The smooth accelerated baseline is its `g_i=0` specialization.

Code is independently implemented from mathematical definitions. Box handling,
domain restarts, inner solvers and the common stopping diagnostic are documented
implementation choices in [the algorithm guide](algorithms.md); the source links
do not imply bit-for-bit reproduction of the authors' software or numerical tables.

## Software license and citation

The MIT license applies to this repository's original code and documentation.
External libraries, papers and data retain their own licenses. A public source
link is not permission to copy and relicense code. Future ports must identify
their provenance and satisfy the source license; independent formula implementations
should cite the defining publication and any cross-check implementation.

Use [references.bib](references.bib) for the central benchmark and validation-library
citations; use each problem card's original paper for other suites. Cite this
software using [CITATION.cff](../CITATION.cff), which links the actual repository.
This file does not claim a software DOI or published paper.
