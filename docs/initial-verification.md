# Initial preparation verification

Local environment: Windows, CPython 3.14.6, NumPy 2.5.3 and pymoo 0.6.2.
Date: 2026-09-23. Results here concern the local checkout; the hosted GitHub Actions
matrix will run after upload.

The full local test pass completed with **185 passing tests**, including 40 optional
pymoo reference test cases. Combined statement/branch coverage reported by
pytest-cov was **98%**. Coverage alone is not evidence of numerical correctness;
the analytical and independent reference checks are described in the validation guide.

Ruff lint and formatting checks passed. All **73 generated catalogue pages** matched
the source metadata, and local documentation links passed validation. The sdist and
wheel built successfully, and both passed `twine check`.

The wheel was installed in a fresh temporary virtual environment, then imported
outside the checkout with Python isolated mode. All 19 registered problems were
evaluated; the 43-entry catalogue and CLI were available, and pymoo was not imported.

The Git repository is initialized on `main`; there is no configured remote or
published release. GitHub Actions is configured but has not run on GitHub yet.

Reproduce the checks using the commands in [validation](validation.md) and
[publishing](publishing.md).
