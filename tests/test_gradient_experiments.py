"""The comparison example must retain budgets, failures and its exact problem."""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

SCRIPT = Path(__file__).resolve().parents[1] / "examples/gradient_comparison.py"
SPEC = importlib.util.spec_from_file_location("gradient_comparison", SCRIPT)
experiment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = experiment
SPEC.loader.exec_module(experiment)


def test_paired_experiment_exports_reproduce_exact_values_and_keep_failures(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    experiment.run(a, starts=2, seed=73, max_iter=5, plots=False)
    experiment.run(b, starts=2, seed=73, max_iter=5, plots=False)
    manifest = json.loads((a / "manifest.json").read_text())
    assert manifest["starts"] == 2 and manifest["seed"] == 73
    assert manifest["derivative_check"]["passed"]
    with (a / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 16
    assert any(r["status"] == "max_iter" for r in rows)
    p = experiment.JOS1Demo()
    reg = experiment.L1([0.1, 0.05], [0, 1]).bind(2, 10)
    for path in a.glob("*/*.npz"):
        other = b / path.relative_to(a)
        with np.load(path) as data, np.load(other) as duplicate:
            for key in data.files:
                assert_array_equal(data[key], duplicate[key])
            assert_array_equal(data["X0"], manifest["X0"])
            expected = p.evaluate(data["X"]).F
            assert_allclose(data["smooth_F"], expected)
            if path.parent.name == "composite_l1":
                expected = expected + np.stack([reg.values(x) for x in data["X"]])
            assert_allclose(data["F"], expected)
            results = json.loads(path.with_suffix(".json").read_text())
            for i, result in enumerate(results):
                assert_array_equal(result["x"], data["X"][i])
                assert_array_equal(result["fun"], data["F"][i])
                assert result["n_iter"] <= 5
                assert result["nfev"] >= result["njev"]
                assert result["success"] == (result["status"] == "stationary")
