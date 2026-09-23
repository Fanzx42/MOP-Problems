# References and provenance

The problem cards contain per-family sources and outstanding version questions.
Sources were consulted on 2026-09-23. This repository implements the formulas in
NumPy code, with declared MIT engineering adaptations and source attributions.

## Problem papers and exact implementation versions

The [complete paper index](problem-references.md) covers all 343 callable problems.
Every [problem card](../problems/README.md) records its paper, formula implementation,
reference implementation URL/version, dimensions, derivative notes and variants.
Use `get_problem_info(name)["references"]` to export citations programmatically.
Where a DOI has not been verified, a reference-code bibliography or author material
is linked instead; the article title and authors are still recorded.

The [expanded validation record](expanded-validation.md) specifies independent
reference cases and differences between paper and code conventions. Numerical
agreement with a pinned version does not imply agreement with every later revision.

Original sources include pinned PlatEMO, jMetalPy and pymoo definitions; Tanabe and
Ishibuchi's RE/CRE code; Cuate et al.'s equality dataset; Jiang and Yang's JY header;
Li et al.'s NCTP archive; Gong's CMMOP archive; Qiao et al.'s LSCM archive; and
Suganthan's CEC 2021 real-world suite. URLs and immutable revisions or archive hashes
are retained per problem. POL uses the corrected Van Veldhuizen definition cited
in MOEA Framework, with the original Poloni publication also acknowledged.

RE/CRE and RWA code adaptations retain their MIT notices in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md), included in distributions.
Other benchmark implementations use vectorized mathematical formulations;
the downloaded reference archives are not redistributed.

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
