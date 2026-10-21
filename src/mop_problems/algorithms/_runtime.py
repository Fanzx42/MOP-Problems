"""Shared accounting and result contract for deterministic algorithms."""

from dataclasses import asdict
from time import perf_counter

import numpy as np

from ..core import NonDifferentiableError, _readonly
from ._subproblem import SubproblemError, minimax_step
from .first_order import Iteration, OptimizeResult, _BudgetReached
from .proximal import Zero


def positive(value, name, *, zero=False):
    if (
        isinstance(value, (bool, np.bool_))
        or not np.isscalar(value)
        or not np.isfinite(value)
        or (value < 0 if zero else value <= 0)
    ):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")


def freeze(value):
    if isinstance(value, np.ndarray):
        return _readonly(value)
    if isinstance(value, dict):
        return {key: freeze(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    return value


def options_with_L(options):
    options = dict(options)
    if "L" in options:
        if "lipschitz" in options:
            raise ValueError("Specify only one of L and lipschitz")
        options["lipschitz"] = options.pop("L")
    return options


class Run:
    def __init__(self, problem, x0, options, name, regularizer=None, *, constraints=False):
        self.started = perf_counter()
        self.problem, self.o, self.name = problem, options, name
        if not constraints and (problem.n_ieq_constr or problem.n_eq_constr):
            raise ValueError(f"{name} does not support explicit G/H constraints")
        if np.asarray(x0).ndim != 1:
            raise ValueError("x0 must be one decision vector")
        self.x = problem._validate_input(x0, check_bounds=options.bounds == "box")[0].copy()
        self.lower = problem.xl if options.bounds == "box" else np.full(problem.n_var, -np.inf)
        self.upper = problem.xu if options.bounds == "box" else np.full(problem.n_var, np.inf)
        self.penalty = Zero(problem.n_obj) if regularizer is None else regularizer
        if hasattr(self.penalty, "bind"):
            self.penalty = self.penalty.bind(problem.n_obj, problem.n_var)
        self.counts = dict(nfev=0, njev=0, nprox=0, nsubproblem=0, subproblem_iterations=0)
        self.history = []
        self.n_iter = self.restarts = 0
        self.step = 0.0
        self.residual = self.bound = self.gap = None
        self.details = {}
        self.status, self.message = "max_iter", "Iteration budget reached."
        self.ev = self.values(self.x)
        self.smooth = self.ev.F[0]
        self.fun = self.smooth + self.g(self.x)

    def g(self, x):
        g = np.asarray(self.penalty.values(x), dtype=float)
        if g.shape != (self.problem.n_obj,) or not np.isfinite(g).all():
            raise ValueError("regularizer.values must return finite (n_obj,) values")
        return g

    def charge(self, *, jac=False):
        if self.counts["nfev"] >= self.o.max_nfev:
            raise _BudgetReached("max_nfev")
        if jac and self.counts["njev"] >= self.o.max_njev:
            raise _BudgetReached("max_njev")
        self.counts["nfev"] += 1
        self.counts["njev"] += int(jac)

    def values(self, x):
        self.charge()
        return self.problem.evaluate(x, check_bounds=False)

    def joint(self, x):
        self.charge(jac=True)
        return self.problem.evaluate_with_jacobian(x, check_bounds=False)

    def prox_count(self):
        self.counts["nprox"] += 1

    def subproblem(self, y, J, offsets, step=1.0, penalty=None):
        self.counts["nsubproblem"] += 1
        answer = minimax_step(
            y,
            J,
            offsets,
            self.penalty if penalty is None else penalty,
            step,
            self.lower,
            self.upper,
            tol=self.o.subproblem_tol,
            max_iter=self.o.subproblem_max_iter,
            on_prox=self.prox_count,
        )
        self.counts["subproblem_iterations"] += answer.iterations
        return answer

    def stationarity(self, J):
        answer = self.subproblem(self.x, J, -self.g(self.x))
        self.residual = float(np.linalg.norm(answer.x - self.x, ord=np.inf))
        self.gap = answer.gap
        self.bound = self.residual + np.sqrt(2 * answer.gap)
        return answer

    def record(self):
        if not self.o.store_history:
            return
        item = Iteration(
            self.n_iter,
            _readonly(self.x),
            _readonly(self.fun),
            self.residual,
            self.bound,
            self.step,
            self.counts["nfev"],
            self.counts["njev"],
            self.gap,
            (),
            freeze(self.details),
        )
        if self.history and self.history[-1].iteration == self.n_iter:
            self.history[-1] = item
        else:
            self.history.append(item)

    def accept(self, x, ev, step, **details):
        self.x, self.ev = np.array(x, copy=True), ev
        self.smooth = ev.F[0]
        self.fun = self.smooth + self.g(self.x)
        self.n_iter += 1
        self.step = float(step)
        self.residual = self.bound = self.gap = None
        self.details = details

    def stop(self, status, message):
        self.status, self.message = status, message

    def failed(self, error):
        if isinstance(error, _BudgetReached):
            self.stop(str(error), "Evaluation budget reached; last accepted point returned.")
        elif isinstance(error, NonDifferentiableError):
            self.stop("nondifferentiable", str(error))
        elif isinstance(error, SubproblemError):
            self.stop("subproblem_failed", str(error))
        else:
            self.stop("domain_error", str(error))

    def result(self, **extra):
        self.record()
        feasible = (
            np.all(self.x >= self.problem.xl)
            and np.all(self.x <= self.problem.xu)
            and np.all(self.ev.G <= self.o.tol)
            and np.all(np.abs(self.ev.H) <= self.o.tol)
        )
        return OptimizeResult(
            x=_readonly(self.x),
            fun=_readonly(self.fun),
            smooth_fun=_readonly(self.smooth),
            success=self.status == "stationary",
            status=self.status,
            message=self.message,
            n_iter=self.n_iter,
            **self.counts,
            elapsed_seconds=perf_counter() - self.started,
            stationarity=self.residual,
            stationarity_bound=self.bound,
            domain_restarts=self.restarts,
            feasible=bool(feasible),
            algorithm=self.name,
            options=asdict(self.o),
            history=tuple(self.history),
            diagnostics=freeze({**self.details, **extra}),
        )


RUN_ERRORS = (_BudgetReached, NonDifferentiableError, SubproblemError, FloatingPointError)


def feasible_step(x, d, lower, upper):
    ratios = np.full_like(d, np.inf)
    np.divide(upper - x, d, out=ratios, where=d > 0)
    np.divide(lower - x, d, out=ratios, where=d < 0)
    return max(0.0, float(np.min(ratios)))
