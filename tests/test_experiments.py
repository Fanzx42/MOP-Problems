"""Reproducibility and correctness of the exported experiment data."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

SCRIPT = Path(__file__).resolve().parents[1] / "examples/representative_suite.py"
SPEC = importlib.util.spec_from_file_location("representative_suite", SCRIPT)
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


@pytest.mark.parametrize("width", [2, 3, 8])
def test_dominance_matches_exhaustive_definition(width):
    rng = np.random.default_rng(174)
    F = rng.integers(-3, 4, size=(71, width)).astype(float)
    expected = np.array(
        [not np.any(np.all(F <= point, axis=1) & np.any(F < point, axis=1)) for point in F]
    )
    assert_array_equal(experiment.nondominated_mask(F), expected)
    assert experiment.nondominated_mask(np.empty((0, width))).shape == (0,)
    assert_array_equal(experiment.nondominated_mask(np.ones((4, width))), np.ones(4, bool))


@pytest.mark.parametrize("values", [[[np.nan, 0]], [[0]], [1, 2]])
def test_dominance_rejects_bad_data(values):
    with pytest.raises(ValueError):
        experiment.nondominated_mask(values)


def test_experiment_csvs_reproduce_and_respect_feasibility(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    summaries = experiment.run(first, samples=19, seed=5, plots=False)
    assert experiment.run(second, samples=19, seed=5, plots=False) == summaries
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["samples_per_case"] == 19
    assert manifest["base_seed"] == 5
    for summary in summaries:
        data = np.genfromtxt(first / f"{summary['case']}.csv", delimiter=",", names=True)
        feasible = data["feasible"].astype(bool)
        nd = data["sample_nd"].astype(bool)
        assert_array_equal(feasible, data["cv"] == 0)
        assert not np.any(nd & ~feasible)
        F = np.column_stack([data[f"f{i + 1}"] for i in range(summary["n_obj"])])
        expected = np.zeros(len(F), bool)
        expected[feasible] = experiment.nondominated_mask(F[feasible])
        assert_array_equal(nd, expected)
        assert int(nd.sum()) == summary["sample_nd"]


def test_invalid_experiment_settings(tmp_path):
    with pytest.raises(ValueError):
        experiment.run(tmp_path, samples=0, plots=False)
    with pytest.raises(ValueError):
        experiment.run(tmp_path, seed=-1, plots=False)
