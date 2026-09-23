"""Build-independent wheel smoke check in a fresh, disposable virtual environment.

Usage: python scripts/check_wheel.py dist/mop_problems-0.1.0-py3-none-any.whl
The wheel's declared runtime dependencies are installed; no editable checkout is used.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SMOKE = """
import pathlib
import sys
import numpy as np
import mop_problems
from mop_problems import get_problem, list_problems, list_suites

location = pathlib.Path(mop_problems.__file__).resolve()
assert location.is_relative_to(pathlib.Path(sys.prefix).resolve()), location
assert len(list_problems()) == 19
assert len(list_suites()) == 43
for name in list_problems():
    problem = get_problem(name)
    result = problem.evaluate(problem.sample(8, seed=42))
    assert result.F.shape == (8, problem.n_obj)
    assert np.isfinite(result.F).all()
assert get_problem('BNH').evaluate([1, 1]).feasible[0]
assert 'pymoo' not in sys.modules
print('Installed wheel passed: 19 problems, 43 catalogue entries, NumPy-only runtime.')
print('Imported from:', location)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve(strict=True)
    if wheel.suffix != ".whl":
        parser.error("expected a .whl distribution")
    with tempfile.TemporaryDirectory(prefix="mop-problems-wheel-") as temp:
        root = Path(temp).resolve()
        # All temporary files and cleanup stay inside this newly created directory.
        assert root.is_relative_to(Path(tempfile.gettempdir()).resolve())
        env = root / "env"
        subprocess.run([sys.executable, "-m", "venv", str(env)], check=True)
        python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--quiet",
                "--disable-pip-version-check",
                str(wheel),
            ],
            cwd=root,
            check=True,
        )
        subprocess.run([str(python), "-I", "-c", SMOKE], cwd=root, check=True)
        subprocess.run(
            [str(python), "-I", "-m", "mop_problems", "list", "--suite", "ZDT"],
            cwd=root,
            check=True,
        )


if __name__ == "__main__":
    main()
