"""MOP-Problems: continuous multi-objective optimization benchmarks."""

from mop_problems.catalog import (
    PlannedProblemError,
    catalog,
    get_problem,
    get_problem_info,
    get_suite,
    list_problems,
    list_suites,
)
from mop_problems.core import Evaluation, Problem

__version__ = "0.1.0"

__all__ = [
    "Evaluation",
    "PlannedProblemError",
    "Problem",
    "catalog",
    "get_problem",
    "get_problem_info",
    "get_suite",
    "list_problems",
    "list_suites",
]
