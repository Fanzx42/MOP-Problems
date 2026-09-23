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
from mop_problems.core import Evaluation, Jacobian, NonDifferentiableError, Problem
from mop_problems.differentiation import check_derivatives, finite_difference

__version__ = "0.3.0"

__all__ = [
    "Evaluation",
    "Jacobian",
    "NonDifferentiableError",
    "PlannedProblemError",
    "Problem",
    "catalog",
    "check_derivatives",
    "finite_difference",
    "get_problem",
    "get_problem_info",
    "get_suite",
    "list_problems",
    "list_suites",
]
