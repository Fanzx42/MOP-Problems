"""Read-only public access to the packaged problem catalogue."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from functools import lru_cache
from importlib import import_module
from importlib.resources import files
from typing import Any

from mop_problems.core import Problem


class PlannedProblemError(NotImplementedError):
    """A known catalogue entry has no validated Python implementation yet."""


def _key(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("problem or suite name must be a nonempty string")
    return re.sub(r"[\s_-]", "", name).upper()


@lru_cache(maxsize=1)
def _catalogue() -> dict[str, Any]:
    path = files("mop_problems").joinpath("data", "catalog.json")
    return json.loads(path.read_text(encoding="utf-8"))


def catalog() -> dict[str, Any]:
    """Return a detached copy; callers cannot mutate the registry."""
    return deepcopy(_catalogue())


def list_suites(*, category: str | None = None, status: str | None = None) -> list[dict]:
    """List all suites, optionally filtering a category prefix and implementation status."""
    if status not in (None, "implemented", "planned"):
        raise ValueError("status must be 'implemented', 'planned', or None")
    suites = _catalogue()["suites"]
    return deepcopy(
        [
            s
            for s in suites
            if (status is None or s["status"] == status)
            and (
                category is None
                or s["category"] == category
                or s["category"].startswith(category.rstrip("/") + "/")
            )
        ]
    )


def get_suite(name: str) -> dict:
    """Return a suite's metadata, including its source and implementation status."""
    key = _key(name)
    for suite in _catalogue()["suites"]:
        if key in {_key(suite["id"]), _key(suite["name"])}:
            return deepcopy(suite)
    raise KeyError(f"Unknown suite: {name!r}")


def list_problems(*, suite: str | None = None, category: str | None = None) -> list[str]:
    """List callable problems only. Use list_suites() to inspect planned entries."""
    allowed = {s["id"] for s in list_suites(category=category)}
    if suite is not None:
        allowed &= {get_suite(suite)["id"]}
    return [p["name"] for p in _catalogue()["problems"] if p["suite"] in allowed]


def get_problem_info(name: str) -> dict:
    """Return metadata for an implemented problem (dimensions are defaults)."""
    key = _key(name)
    for problem in _catalogue()["problems"]:
        if key == _key(problem["name"]) or key in map(_key, problem.get("aliases", [])):
            return deepcopy(problem)
    for suite in _catalogue()["suites"]:
        candidates = [suite["id"], suite["name"], *suite["members"]]
        if suite["status"] == "planned" and key in map(_key, candidates):
            raise PlannedProblemError(
                f"{name!r} is catalogued under {suite['name']} but is not implemented. "
                f"See {suite['path']}/README.md."
            )
    raise KeyError(f"Unknown problem: {name!r}. Use list_problems() for callable names.")


def get_problem(name: str, **kwargs: Any) -> Problem:
    """Construct a benchmark; unknown parameters are rejected by its constructor."""
    info = get_problem_info(name)
    module, cls = info["implementation"].split(":")
    return getattr(import_module(module), cls)(**kwargs)
