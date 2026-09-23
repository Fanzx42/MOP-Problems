# Uploading to GitHub / 上传说明

The local project is prepared for GitHub. No remote repository is created by
these files, and nothing is automatically pushed or published to PyPI.

## Before the first upload

From the repository root, run:

```sh
python -m pip install -e ".[dev,reference]"
python -m pytest
python -m ruff check .
python -m ruff format --check .
python scripts/build_catalog.py --check
python -m build
python -m twine check dist/*
python scripts/check_wheel.py dist/mop_problems-0.1.0-py3-none-any.whl
```

PowerShell users can pass the two generated files explicitly to `twine check`:

```powershell
python -m twine check dist/mop_problems-0.1.0-py3-none-any.whl dist/mop_problems-0.1.0.tar.gz
```

The default license for the original repository code is MIT. Confirm the maintainer
name you want displayed in `LICENSE`, `CITATION.cff` and `pyproject.toml`.
These currently use the neutral `MOP-Problems contributors` label.

## Create and push the remote

1. Create an **empty** GitHub repository named `MOP-Problems` (or your preferred name).
   Do not initialize another README/license in that remote.
2. If this folder has not already been initialized, run `git init -b main` once.
3. Review `git status --short`; `.venv`, IDE files, build outputs and local results
   are ignored. Configure your Git identity if Git requests it.
4. Replace `YOUR_USERNAME` and use the actual repository URL below:

```sh
git add .
git commit -m "Initialize MOP benchmark library and catalogue"
git remote add origin https://github.com/YOUR_USERNAME/MOP-Problems.git
git push -u origin main
```

If `origin` already exists, inspect `git remote -v` and reuse or correct it rather
than adding a second origin. Credentials are managed by Git/GitHub; never put a
token into source files or the remote URL.

## Repository settings and first release

- Enable Actions and inspect the first CI run. Local success does not prove that
  the hosted Linux/macOS/Python matrix has already run.
- Add a description and relevant topics, such as `multi-objective-optimization`,
  `benchmark-functions`, `python` and `scientific-computing`.
- Enable private vulnerability reporting if desired. Set branch protection to
  require the CI checks before merging pull requests.
- Add the real repository URL to `CITATION.cff` and `[project.urls]` in
  `pyproject.toml`. No owner, homepage badge, DOI or remote URL is fabricated.
- After CI passes and the release scope is accepted, update the changelog from
  `Unreleased`, tag a release, and include the documented 19-problem coverage.
  PyPI publication is a separate future step; this repository has no upload workflow.

首版有完整分类，但只有 19 个已实现问题。发布说明应保留这一事实，
不能把目录数量写成已实现函数数量。其余问题按来源核验、实现、测试的顺序逐步加入。
