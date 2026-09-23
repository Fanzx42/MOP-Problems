"""First-order multiobjective baselines with explicit variants and accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from ..core import NonDifferentiableError, _readonly, integer
from ._subproblem import SubproblemError, minimax_step
from .proximal import Zero


@dataclass(frozen=True)
class SolverOptions:
    step_rule: str = "auto"
    step_size: float = 1.0
    max_iter: int = 500
    tol: float = 1e-5
    max_nfev: int = 10000
    max_njev: int = 2000
    armijo: float = 0.1
    contraction: float = 0.5
    max_line_search: int = 60
    min_step: float = 1e-16
    nonmonotone_window: int = 10
    average_eta: float = 0.8
    bb_min: float = 1e-3
    bb_max: float = 1e3
    bounds: str = "box"
    domain_restart: bool = True
    subproblem_tol: float = 1e-12
    subproblem_max_iter: int = 500
    store_history: bool = True

    def __post_init__(self):
        for name in ("step_size", "tol", "min_step", "bb_min", "bb_max", "subproblem_tol"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
                raise ValueError(f"{name} must be a finite positive scalar")
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive scalar")
        for name in ("armijo", "contraction", "average_eta"):
            value = getattr(self, name)
            if not np.isscalar(value) or not np.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must lie strictly between zero and one")
        for name in ("max_iter", "nonmonotone_window"):
            integer(getattr(self, name), name, 0)
        for name in ("max_nfev", "max_njev", "max_line_search", "subproblem_max_iter"):
            integer(getattr(self, name), name)
        if self.bb_min > self.bb_max or self.min_step > self.step_size:
            raise ValueError("require bb_min <= bb_max and min_step <= step_size")
        if self.bounds not in ("box", "none"):
            raise ValueError("bounds must be 'box' or 'none'")
        if self.step_rule not in (
            "auto",
            "fixed",
            "armijo",
            "nonmonotone",
            "nonmonotone_average",
            "backtracking",
        ):
            raise ValueError("unknown step_rule")
        for name in ("domain_restart", "store_history"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")


@dataclass(frozen=True)
class Iteration:
    iteration: int
    x: np.ndarray
    fun: np.ndarray
    stationarity: float | None
    stationarity_bound: float | None
    step_size: float
    nfev: int
    njev: int
    subproblem_gap: float | None
    bb_scales: tuple[float, ...]


@dataclass(frozen=True)
class OptimizeResult:
    x: np.ndarray
    fun: np.ndarray
    smooth_fun: np.ndarray
    success: bool
    status: str
    message: str
    n_iter: int
    nfev: int
    njev: int
    nprox: int
    nsubproblem: int
    subproblem_iterations: int
    elapsed_seconds: float
    stationarity: float | None
    stationarity_bound: float | None
    domain_restarts: int
    feasible: bool
    algorithm: str
    options: dict
    history: tuple[Iteration, ...]


class _BudgetReached(RuntimeError):
    pass


def bb_scales(s, gradient_difference, minimum=1e-3, maximum=1e3):
    """Chen--Tang--Yang Eq. (22): positive, negative and zero curvature cases."""
    squared = float(s @ s)
    if squared == 0:
        return np.ones(len(gradient_difference))
    curvature = gradient_difference @ s
    scales = np.where(
        curvature > 0,
        curvature / squared,
        np.where(
            curvature < 0, np.linalg.norm(gradient_difference, axis=1) / np.sqrt(squared), minimum
        ),
    )
    return np.clip(scales, minimum, maximum)


class FirstOrderMethod:
    """Immutable run configuration; all iteration state is local to solve()."""

    name = ""
    accelerated = False
    proximal = False

    def __init__(self, *, regularizer=None, **options):
        self.options = SolverOptions(**options)
        if regularizer is not None and not self.proximal:
            raise ValueError("Use ProximalGradient or NesterovAcceleratedProximalGradient for g_i")
        self.regularizer = regularizer
        rule = self.options.step_rule
        model_method = self.proximal or self.accelerated
        self.step_rule = ("backtracking" if model_method else "armijo") if rule == "auto" else rule
        allowed = (
            ("fixed", "backtracking")
            if model_method
            else ("fixed", "armijo", "nonmonotone", "nonmonotone_average")
        )
        if self.step_rule not in allowed:
            raise ValueError(f"{self.name} supports step_rule in {allowed}")

    def solve(self, problem, x0) -> OptimizeResult:
        """Solve from one start. Explicit nonlinear G/H constraints are unsupported.

        With bounds='box', the box is included INSIDE the minimax subproblem.
        bounds='none' explicitly selects the unconstrained formula extension.
        """
        started = perf_counter()
        o = self.options
        if problem.n_ieq_constr or problem.n_eq_constr:
            raise ValueError(
                "These baselines handle a box/proximable convex terms, not explicit nonlinear "
                "G/H constraints. Their values and Jacobians remain available to external solvers."
            )
        raw = np.asarray(x0)
        if raw.ndim != 1:
            raise ValueError("x0 must have shape (n_var,); use multistart for a batch")
        x = problem._validate_input(x0, check_bounds=o.bounds == "box")[0].copy()
        lower = problem.xl if o.bounds == "box" else np.full(problem.n_var, -np.inf)
        upper = problem.xu if o.bounds == "box" else np.full(problem.n_var, np.inf)
        penalty = self.regularizer if self.regularizer is not None else Zero(problem.n_obj)
        if hasattr(penalty, "bind"):
            penalty = penalty.bind(problem.n_obj, problem.n_var)
        counts = dict(nfev=0, njev=0, nprox=0, nsubproblem=0, subproblem_iterations=0)

        def values(point):
            if counts["nfev"] >= o.max_nfev:
                raise _BudgetReached("max_nfev")
            counts["nfev"] += 1
            return problem.evaluate(point, check_bounds=False).F[0]

        def joint(point):
            if counts["nfev"] >= o.max_nfev:
                raise _BudgetReached("max_nfev")
            if counts["njev"] >= o.max_njev:
                raise _BudgetReached("max_njev")
            counts["nfev"] += 1
            counts["njev"] += 1
            ev, jac = problem.evaluate_with_jacobian(point, check_bounds=False)
            return ev.F[0], jac.dF[0]

        def prox_count():
            counts["nprox"] += 1

        def subproblem(point, jac, offsets, term, step):
            counts["nsubproblem"] += 1
            answer = minimax_step(
                point,
                jac,
                offsets,
                term,
                step,
                lower,
                upper,
                tol=o.subproblem_tol,
                max_iter=o.subproblem_max_iter,
                on_prox=prox_count,
            )
            counts["subproblem_iterations"] += answer.iterations
            return answer

        def g_value(point):
            g = np.asarray(penalty.values(point), dtype=float)
            if g.shape != (problem.n_obj,) or not np.isfinite(g).all():
                raise ValueError("regularizer.values must return (n_obj,) finite values")
            return g

        smooth = values(x)
        fun = smooth + g_value(x)
        y, momentum, current_step = x.copy(), 1.0, o.step_size
        previous_x, previous_jac = None, None
        bb = np.ones(problem.n_obj)
        history, past_values = [], [fun.copy()]
        average, average_weight = fun.copy(), 1.0
        n_iter = restarts = 0
        residual = residual_bound = gap = None
        last_step = 0.0
        status, message = "max_iter", "Iteration budget reached."

        def record():
            if o.store_history:
                history.append(
                    Iteration(
                        n_iter,
                        _readonly(x),
                        _readonly(fun),
                        residual,
                        residual_bound,
                        last_step,
                        counts["nfev"],
                        counts["njev"],
                        gap,
                        tuple(map(float, bb)),
                    )
                )

        try:
            for k in range(o.max_iter + 1):
                smooth, J = joint(x)
                gx = g_value(x)
                fun = smooth + gx
                # A common, UNSCALED unit-step residual makes stopping comparable.
                canonical = subproblem(x, J, -gx, penalty, 1.0)
                gap = canonical.gap
                residual = float(np.linalg.norm(canonical.x - x, ord=np.inf))
                residual_bound = residual + np.sqrt(2 * canonical.gap)
                record()
                if residual_bound <= o.tol:
                    status, message = (
                        "stationary",
                        "Certified first-order residual below tolerance.",
                    )
                    break
                if k == o.max_iter:
                    break
                if self.name == "bbdmo" and previous_x is not None:
                    bb = bb_scales(x - previous_x, J - previous_jac, o.bb_min, o.bb_max)
                if self.accelerated and not np.array_equal(y, x):
                    fy, Jy = joint(y)
                else:
                    y, fy, Jy = x.copy(), smooth, J
                model_method = self.proximal or self.accelerated
                if model_method:
                    offsets = fy - fun
                    trial_step = current_step
                else:
                    scaled = J / bb[:, None]
                    direction_model = (
                        canonical
                        if o.step_size == 1 and np.all(bb == 1)
                        else subproblem(x, scaled, np.zeros(problem.n_obj), penalty, o.step_size)
                    )
                    direction = direction_model.x - x
                    slopes = J @ direction
                    if np.any(slopes >= 0):
                        status, message = (
                            "no_descent",
                            "Inner solution is not a strict common descent direction.",
                        )
                        break
                    trial_step = 1.0
                accepted = False
                for _ in range(o.max_line_search):
                    effective_step = trial_step if model_method else trial_step * o.step_size
                    if effective_step < o.min_step:
                        break
                    if model_method:
                        model = subproblem(y, Jy, offsets, penalty, trial_step)
                        trial = model.x
                    else:
                        trial = np.clip(x + trial_step * direction, lower, upper)
                    if np.array_equal(trial, x):
                        # Inertia may cancel a nonzero proximal correction. Do not
                        # mistake this for stationarity; the canonical test above is authoritative.
                        if self.accelerated and not np.array_equal(y, x):
                            y, fy, Jy, momentum = x.copy(), smooth, J, 1.0
                            offsets = -gx
                            restarts += 1
                            continue
                        break
                    try:
                        trial_smooth = values(trial)
                        trial_fun = trial_smooth + g_value(trial)
                    except FloatingPointError:
                        if self.step_rule == "fixed":
                            break
                        trial_step *= o.contraction
                        continue
                    if self.step_rule == "fixed":
                        accepted = True
                    elif model_method:
                        delta = trial - y
                        bound = fy + Jy @ delta + float(delta @ delta) / (2 * trial_step)
                        slack = 16 * np.finfo(float).eps * np.maximum(1, np.abs(fy))
                        accepted = bool(np.all(trial_smooth <= bound + slack))
                    else:
                        reference = fun
                        if self.step_rule == "nonmonotone":
                            reference = np.max(past_values[-(o.nonmonotone_window + 1) :], axis=0)
                        elif self.step_rule == "nonmonotone_average":
                            reference = average
                        bound = reference + o.armijo * trial_step * slopes
                        slack = 16 * np.finfo(float).eps * np.maximum(1, np.abs(reference))
                        accepted = bool(np.all(trial_fun <= bound + slack))
                    if accepted:
                        break
                    trial_step *= o.contraction
                if not accepted:
                    status, message = (
                        "line_search_failed",
                        "No acceptable, distinct step within the search budget.",
                    )
                    break
                previous_x, previous_jac = x.copy(), J.copy()
                x, smooth, fun = trial.copy(), trial_smooth, trial_fun
                past_values.append(fun.copy())
                if self.step_rule == "nonmonotone_average":
                    new_weight = o.average_eta * average_weight + 1
                    average = (o.average_eta * average_weight * average + fun) / new_weight
                    average_weight = new_weight
                n_iter += 1
                last_step = effective_step
                residual = residual_bound = gap = None
                if model_method:
                    current_step = trial_step
                if self.accelerated:
                    next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
                    y = x + (momentum - 1) / next_momentum * (x - previous_x)
                    momentum = next_momentum
                    if (
                        o.domain_restart
                        and o.bounds == "box"
                        and (np.any(y < lower) or np.any(y > upper))
                    ):
                        y, momentum = x.copy(), 1.0
                        restarts += 1
        except _BudgetReached as error:
            status, message = (
                str(error),
                "Evaluation budget reached; the last accepted point is returned.",
            )
        except NonDifferentiableError as error:
            status, message = "nondifferentiable", str(error)
        except SubproblemError as error:
            status, message = "subproblem_failed", str(error)
        except FloatingPointError as error:
            status, message = "domain_error", str(error)
        if not history or history[-1].iteration != n_iter:
            record()
        options = asdict(o)
        options["step_rule"] = self.step_rule
        return OptimizeResult(
            x=_readonly(x),
            fun=_readonly(fun),
            smooth_fun=_readonly(smooth),
            success=status == "stationary",
            status=status,
            message=message,
            n_iter=n_iter,
            **counts,
            elapsed_seconds=perf_counter() - started,
            stationarity=residual,
            stationarity_bound=residual_bound,
            domain_restarts=restarts,
            feasible=bool(np.all(x >= problem.xl) and np.all(x <= problem.xu)),
            algorithm=self.name,
            options=options,
            history=tuple(history),
        )


class MGDA(FirstOrderMethod):
    """Minimax steepest descent / minimum-norm gradient combination (GD alias)."""

    name = "mgda"


GD = MGDA


class BBDMO(FirstOrderMethod):
    """Per-objective BB scaling, Chen--Tang--Yang (2023), Eq. (22), Algorithm 5."""

    name = "bbdmo"


class ProximalGradient(FirstOrderMethod):
    """Multiobjective proximal gradient, Tanabe--Fukuda--Yamashita (2019)."""

    name = "proximal_gradient"
    proximal = True


class NesterovAcceleratedGradient(FirstOrderMethod):
    """TFY (2023) accelerated minimax method with g_i=0 (not naive extrapolated MGDA)."""

    name = "nesterov"
    accelerated = True


class NesterovAcceleratedProximalGradient(FirstOrderMethod):
    """TFY (2023) Algorithm 2, including f_i(y)-F_i(x) in every subproblem."""

    name = "nesterov_proximal"
    proximal = True
    accelerated = True
