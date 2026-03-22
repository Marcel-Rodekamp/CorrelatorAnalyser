from __future__ import annotations

from collections.abc import Callable

import numpy as np
import multiprocess as mp

from .data import Data
from .prior import Prior
from .fitResult import FitResult

from .fit_helper import _validate_inputs, _get_p0, _get_abscissa, _compute_cov_inv

# Cost-function builders — plain iminuit path
from .fit_iminuit import (
    _build_uncorrelated_cost,
    _build_correlated_cost,
    _run_minuit,
)

# Cost-function builders and executor — variable-projection path
from .fit_iminuit_variable_projection import (
    _build_varproj_uncorrelated_cost,
    _build_varproj_correlated_cost,
    _run_minuit_varproj,
)

# =============================================================================
# Module-level constant
# =============================================================================

_FD_STEP: float = 1e-5   # relative step for central finite-difference gradient
_MAX_ADAM_STEPS:int = 10_000

# =============================================================================
# ADAM step
# =============================================================================

def _numerical_gradient(cost, theta: np.ndarray) -> np.ndarray:
    """
    Central finite-difference gradient of ``cost`` at ``theta``.

    Uses a relative step h = _FD_STEP * |theta_i| (with a floor of _FD_STEP
    so that parameters near zero are still perturbed).
    """
    g = np.empty_like(theta)
    for i in range(len(theta)):
        h      = _FD_STEP * max(abs(theta[i]), 1.0)
        t_fwd  = theta.copy(); t_fwd[i]  += h
        t_bwd  = theta.copy(); t_bwd[i]  -= h
        g[i]   = (cost(*t_fwd) - cost(*t_bwd)) / (2.0 * h)
    return g


def _run_adam(
    cost,
    theta0: np.ndarray,
    *,
    alpha:     float,
    beta1:     float,
    beta2:     float,
    eps:       float,
    precision: float,
    length:    int,
    limits:    dict | None,
    param_names: list[str],
) -> tuple[np.ndarray, list[float]]:
    """
    Run the ADAM optimiser on *cost* until the handover criterion is met.

    Parameters
    ----------
    cost : callable
        Cost function whose positional arguments correspond to *param_names*.
        May optionally have a ``.grad`` attribute returning the gradient.
    theta0 : np.ndarray, shape (Nparams,)
        Initial parameter vector.
    alpha, beta1, beta2, eps : float
        ADAM hyperparameters (learning rate, moment decays, numerical floor).
    precision : float
        Handover threshold.  ADAM stops when the maximum relative change in
        chi² over the last *length* iterations is below this value.
    length : int
        Number of consecutive iterations used to evaluate the criterion.
    limits : dict | None
        Parameter bounds ``{name: (lo, hi)}``.  ``None`` values mean
        unbounded.  Applied by clipping after each ADAM update.
    param_names : list[str]
        Ordered list of parameter names (same order as *theta0*).

    Returns
    -------
    theta_best : np.ndarray
        Parameter vector at the lowest cost seen during ADAM.
    chi2_history : list[float]
        Cost value at every ADAM iteration.
    """
    has_analytic_grad = hasattr(cost, "grad")

    # Build limit arrays for fast clipping (None → ±inf)
    lo = np.full(len(theta0), -np.inf)
    hi = np.full(len(theta0),  np.inf)
    if limits is not None:
        for i, name in enumerate(param_names):
            if name in limits:
                bound = limits[name]
                if bound[0] is not None:
                    lo[i] = bound[0]
                if bound[1] is not None:
                    hi[i] = bound[1]

    theta    = theta0.copy()
    m        = np.zeros_like(theta)   # first moment
    v        = np.zeros_like(theta)   # second moment
    t        = 0                      # iteration counter

    chi2_history: list[float] = []
    theta_best  = theta.copy()
    chi2_best   = cost(*theta)
    chi2_history.append(chi2_best)

    while t < _MAX_ADAM_STEPS:
        t += 1

        # --- gradient ---
        if has_analytic_grad:
            g = np.asarray(cost.grad(*theta), dtype=float)
        else:
            g = _numerical_gradient(cost, theta)

        # --- ADAM moment updates ---
        m  = beta1 * m + (1.0 - beta1) * g
        v  = beta2 * v + (1.0 - beta2) * g ** 2

        # Bias-corrected moments
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)

        # --- parameter update with projection onto feasible region ---
        theta -= alpha * m_hat / (np.sqrt(v_hat) + eps)
        theta  = np.clip(theta, lo, hi)

        chi2 = float(cost(*theta))
        chi2_history.append(chi2)
        
        if chi2 < chi2_best:
            chi2_best  = chi2
            theta_best = theta.copy()

        # --- handover criterion ---
        if len(chi2_history) >= length + 1:
            window = chi2_history[-(length + 1):]
            
            # Maximum relative change between consecutive values in the window.
            total_improvement = abs(window[0] - window[-1]) / max(abs(window[0]), 1e-300)
            if total_improvement < precision:
                break

    return theta_best, chi2_history


# =============================================================================
# Single-fit executor — plain iminuit path
# =============================================================================

def _run_adam_then_minuit(fit_args: dict) -> dict:
    """
    Run ADAM then hand off to iminuit.  Plain (non-varproj) path.

    Parameters
    ----------
    fit_args : dict
        Keys: ``least_square``, ``p0``, ``param_names``, ``limits``,
        ``adam_alpha``, ``adam_beta1``, ``adam_beta2``, ``adam_eps``,
        ``adam_precision``, ``adam_length``.

    Returns
    -------
    dict with keys ``minuit`` and ``error``.
    """
    out = {"minuit": None, "adam_chi2_history":None, "error": None}
    try:
        cost        = fit_args["least_square"]
        param_names = fit_args["param_names"]
        p0_dict     = fit_args["p0"]
        limits      = fit_args.get("limits")
        maxiter     = fit_args.get("maxiter", 10_000)

        theta0 = np.array([p0_dict[k] for k in param_names], dtype=float)

        theta_best, cost_hisory = _run_adam(
            cost        = cost,
            theta0      = theta0,
            alpha       = fit_args["adam_alpha"],
            beta1       = fit_args["adam_beta1"],
            beta2       = fit_args["adam_beta2"],
            eps         = fit_args["adam_eps"],
            precision   = fit_args["adam_precision"],
            length      = fit_args["adam_length"],
            limits      = limits,
            param_names = param_names,
        )

        p0_iminuit = dict(zip(param_names, theta_best))
        out["minuit"] = _run_minuit(cost, p0_iminuit, limits, maxiter, fit_args["tol"], fit_args["strategy"])
        out["adam_chi2_history"] = cost_hisory
    except Exception as e:
        out["error"] = e
    return out

def _execute_fits_plain(fit_args, nres=None):
    """
    Run one or many plain-iminuit hybrid fits.

    Same calling convention as ``fit_iminuit._execute_fits``.
    """
    if nres is None:
        result = _run_adam_then_minuit(fit_args)
        return {
            "nres": None, 
            "minuit": result["minuit"], 
            "adam_chi2_history": result["adam_chi2_history"],
            "error": result["error"]
        }

    n   = len(nres)
    out = {"nres": nres, "minuit": [None] * n, "adam_chi2_history": [None] * n, "error": [None] * n}
    for res_id in range(n):
        result = _run_adam_then_minuit(fit_args[res_id])
        out["minuit"][res_id] = result["minuit"]
        out["adam_chi2_history"][res_id] = result["adam_chi2_history"]
        out["error"][res_id]  = result["error"]
    return out

def _execute_parallel_plain(args, Nres, Nproc):
    """
    Split resample fits across Nproc workers and collect results.

    Returns a flat list of (nres, minuit_or_none, error_or_none) tuples.
    """
    blockSize = Nres // Nproc
    Nrest     = Nres % Nproc

    slices = [np.s_[b * blockSize : (b + 1) * blockSize] for b in range(Nproc)]
    if Nrest:
        slices.append(np.s_[Nproc * blockSize :])

    inputs = [
        (args[sl], np.arange(Nres)[sl].tolist())
        for sl in slices
    ]

    with mp.Pool(processes=Nproc) as pool:
        results = pool.starmap(_execute_fits_plain, inputs)

    flat = []
    for result in results:
        for res_id, nres in enumerate(result["nres"]):
            flat.append((
                nres, 
                result["minuit"][res_id], 
                result["adam_chi2_history"][res_id],
                result["error"][res_id],
            ))

    return flat



# =============================================================================
# Single-fit executor — variable-projection path
# =============================================================================

def _run_adam_then_minuit_varproj(fit_args: dict) -> dict:
    """
    Run ADAM then hand off to ``_run_minuit_varproj``.  Varproj path.

    ADAM operates on the nonlinear parameters only (the cost function's
    signature already hides the linear parameters).

    Parameters
    ----------
    fit_args : dict
        Same as ``_run_adam_then_minuit`` plus ``maxiter`` (forwarded to
        ``_run_minuit_varproj``).

    Returns
    -------
    dict with keys ``minuit``, ``varproj``, ``error``.
    """
    out = {"minuit": None, "varproj": None, "varproj_hessians":None, "adam_chi2_history": None, "error": None}
    try:
        cost        = fit_args["least_square"]
        param_names = cost._nonlinear_params   # only the nonlinear names
        p0_dict     = fit_args["p0"]
        limits      = fit_args.get("limits")
        maxiter     = fit_args.get("maxiter", 10_000)

        # Strip linear params from p0 for ADAM's initial vector.
        nl_p0_dict = {k: p0_dict[k] for k in param_names if k in p0_dict}
        theta0     = np.array([nl_p0_dict[k] for k in param_names], dtype=float)

        theta_best, chi2_history = _run_adam(
            cost        = cost,
            theta0      = theta0,
            alpha       = fit_args["adam_alpha"],
            beta1       = fit_args["adam_beta1"],
            beta2       = fit_args["adam_beta2"],
            eps         = fit_args["adam_eps"],
            precision   = fit_args["adam_precision"],
            length      = fit_args["adam_length"],
            limits      = limits,
            param_names = param_names,
        )

        p0_iminuit = dict(zip(param_names, theta_best))
        minuit, varproj, varproj_hessians = _run_minuit_varproj(cost, p0_iminuit, limits, maxiter, fit_args["tol"], fit_args["strategy"])
        out["minuit"]  = minuit
        out["varproj"] = varproj
        out["varproj_hessians"] = varproj_hessians
        out["adam_chi2_history"] = chi2_history
    except Exception as e:
        out["error"] = e
    return out

def _execute_fits_vp(fit_args, nres=None):
    """
    Run one or many varproj hybrid fits.

    Returns the same structure as
    ``fit_iminuit_variable_projection._execute_fits`` so that
    ``_run_parallel_varproj`` can be reused directly.
    """
    if nres is None:
        result = _run_adam_then_minuit_varproj(fit_args)
        return {
            "nres":    None,
            "minuit":  result["minuit"],
            "varproj": result["varproj"],
            "varproj_hessians": result["varproj_hessians"],
            "error":   result["error"],
            "adam_chi2_history": result["adam_chi2_history"]
        }

    n   = len(nres)
    out = {
        "nres":    nres,
        "minuit":  [None] * n,
        "varproj": [None] * n,
        "varproj_hessians": [None] * n,
        "adam_chi2_history": [None] * n,
        "error":   [None] * n,
    }
    for res_id in range(n):
        result = _run_adam_then_minuit_varproj(fit_args[res_id])
        out["minuit"][res_id]  = result["minuit"]
        out["varproj"][res_id] = result["varproj"]
        out["varproj_hessians"][res_id] = result["varproj_hessians"]
        out["adam_chi2_history"][res_id]   = result["adam_chi2_history"]
        out["error"][res_id]   = result["error"]
    return out

def _execute_parallel_varproj(args, Nres, Nproc):
    """
    Like _run_parallel but also unpacks the varproj field.
    Returns list of (nres, minuit, varproj, error).
    """
    blockSize = Nres // Nproc
    Nrest     = Nres % Nproc

    slices = [np.s_[b * blockSize : (b + 1) * blockSize] for b in range(Nproc)]
    if Nrest:
        slices.append(np.s_[Nproc * blockSize :])

    inputs = [(args[sl], np.arange(Nres)[sl].tolist()) for sl in slices]

    with mp.Pool(processes=Nproc) as pool:
        results = pool.starmap(_execute_fits_vp, inputs)

    flat = []
    for result in results:
        for res_id, nres in enumerate(result["nres"]):
            flat.append((
                nres,
                result["minuit"][res_id],
                result["varproj"][res_id],
                result["varproj_hessians"][res_id],
                result["adam_chi2_history"][res_id],
                result["error"][res_id],
            ))
    return flat

# =============================================================================
# Public interface
# =============================================================================

def fit_adam_iminuit_hybrid(
    *,
    abscissa: Data | np.ndarray,
    ordinate: Data,
    # Fit strategy
    central_value_fit: bool = True,
    central_value_fit_correlated: bool = False,
    resample_fit: bool = False,
    resample_fit_correlated: bool = False,
    # Model and parameters
    model: Callable | None = None,
    prior: dict[str, Prior] | None = None,
    p0: dict | None = None,
    linear_params: list[str] | None = None,
    # ADAM hyperparameters
    adam_hyperparam_alpha: float     = 0.01,
    adam_hyperparam_beta1: float     = 0.9,
    adam_hyperparam_beta2: float     = 0.999,
    adam_hyperparam_eps:   float     = 1e-8,
    adam_handover_precision: float   = 0.50,
    adam_handover_length:    int     = 10,
    # iminuit / shared parameters
    limits:   dict | None  = None,
    svdcut:   float | None = None,
    maxiter:  int          = 10_000,
    tolerance: int = 0.1,
    strategy: int = 0,
    # Parallelisation
    Nproc: int | None = None,
) -> FitResult:
    r"""
    Fit using an ADAM pre-optimisation stage followed by iminuit (Minuit2).

    ADAM explores the cost-function landscape from the user-supplied starting
    point *p0* / *prior* and terminates once the chi² converges (see
    handover criterion below).  The resulting parameter estimates are then
    passed to iminuit as a warm start, allowing Minuit to converge quickly
    to the precise minimum.

    This backend is particularly useful for models with highly non-convex or
    flat chi² landscapes (e.g. multi-exponential fits) where Minuit can
    struggle from a cold start.

    Parameters
    ----------
    abscissa : Data | np.ndarray
        Independent variable(s).
    ordinate : Data
        Dependent variable with resample information.
    central_value_fit : bool
        Fit to the central value (mean) of the ordinate.  (default: True)
    central_value_fit_correlated : bool
        Use the full covariance matrix for the central-value fit.  (default: False)
    resample_fit : bool
        Fit every resample.  (default: False)
    resample_fit_correlated : bool
        Use the full covariance matrix for resample fits.  (default: False)
    model : callable
        Model function ``model(abscissa, params_dict) -> np.ndarray``.
        May optionally carry a ``.grad`` attribute for analytic gradients.
    prior : dict[str, Prior] | None
        Priors keyed by parameter name.
    p0 : dict | None
        Initial parameter values (used when *prior* is None).
    linear_params : list[str] | None
        Parameter names that enter the model linearly.  When provided the
        variable-projection (Golub-Pereyra) cost builders are used: ADAM and
        iminuit both operate on the nonlinear parameters only, and the linear
        parameters are recovered analytically after convergence.
        (default: None — use the plain iminuit path)
    adam_hyperparam_alpha : float
        ADAM learning rate.  (default: 0.01)
    adam_hyperparam_beta1 : float
        ADAM first-moment decay rate.  (default: 0.9)
    adam_hyperparam_beta2 : float
        ADAM second-moment decay rate.  (default: 0.999)
    adam_hyperparam_eps : float
        ADAM numerical stability floor for the denominator.  (default: 1e-8)
    adam_handover_precision : float
        ADAM stops once the maximum relative change in chi² across the last
        *adam_handover_length* consecutive iterations drops below this value.
        (default: 0.50)
    adam_handover_length : int
        Number of consecutive iterations evaluated by the handover criterion.
        (default: 10)
    limits : dict | None
        Parameter bounds forwarded to both ADAM (via clipping) and iminuit.
    svdcut : float | None
        Relative SVD cut for the covariance matrix in correlated fits.
    maxiter : int
        Maximum Minuit iterations.  (default: 10 000)
    tol: float
        Controls stopping criterium of the minimizer:
            EDM < 0.002 * tol 
        (default: 0.1)
    strategy: int
        "
        0: Fast. Does not check a user-provided gradient. Does not improve 
        Hesse matrix at minimum. Extra call to hesse() after migrad() is 
        always needed for good error estimates. If you pass a user-provided 
        gradient to MINUIT, convergence is faster.

        1: Default. Checks user-provided gradient against numerical gradient. 
        Checks and usually improves Hesse matrix at minimum. Extra call to 
        hesse() after migrad() is usually superfluous. If you pass a 
        user-provided gradient to MINUIT, convergence is slower.

        2: Careful. Like 1, but does extra checks of intermediate Hessian 
        matrix during minimization. The effect in benchmarks is a somewhat 
        improved accuracy at the cost of more function evaluations. A similar 
        effect can be achieved by reducing the tolerance tol for convergence 
        at any strategy level.
        "
        (default: 0)
    Nproc : int | None
        Number of parallel processes for resample fits.  Serial if ``None``.

    Returns
    -------
    FitResult

    Notes
    -----
    Handover criterion
        Let :math:`f_t` be the cost value at ADAM iteration *t*.  The
        handover fires when

        .. math::

            \max_{i \in [t-L, t-1]}
            \frac{|f_{i+1} - f_i|}{\max(|f_{i+1}|, \varepsilon)}
            < \delta

        where :math:`L` = ``adam_handover_length`` and
        :math:`\delta` = ``adam_handover_precision``.

    ADAM gradient
        If ``model.grad`` is provided and no prior is used on the plain path,
        ``cost.grad`` is available and ADAM uses it directly.  Otherwise
        central finite differences are used with relative step ``1e-5``.
        On the varproj path the analytic gradient is available when
        ``model.grad`` is defined and the prior-gradient extension is enabled
        (both correlated and uncorrelated varproj cost functions now attach
        ``cost.grad`` unconditionally when ``model.grad`` is present).
    """
    if not (central_value_fit or resample_fit):
        raise ValueError("At least one of central_value_fit or resample_fit must be True.")

    _validate_inputs(abscissa, ordinate, model, prior, p0)

    use_varproj = linear_params is not None and len(linear_params) > 0

    # Silently drop priors on linear parameters (matches varproj backend behaviour).
    if use_varproj and prior is not None:
        for key in list(linear_params):
            if key in prior:
                prior.pop(key)
                print(f"Warning: prior for linear parameter '{key}' was ignored.")

    Nres        = ordinate.Nresample
    has_grad = hasattr(model, "grad")
    has_hess = has_grad and hasattr(model, "hessian")
    start_vals  = _get_p0(prior, p0)
 
    if has_hess:
        print("Using gradient and Hessian information from model.grad / model.hessian")
    elif has_grad:
        print("Using gradient information from model.grad")
    elif hasattr(model, "hessian"):
        print(
            "Warning: model.hessian is present but model.grad is missing. "
            "The analytic Hessian will not be used."
        )

    if use_varproj:
        nl_params = [k for k in start_vals if k not in linear_params]
    else:
        nl_params = list(start_vals.keys())

    # Pack ADAM hyperparameters once for reuse in every fit-args dict.
    adam_kwargs = dict(
        adam_alpha     = adam_hyperparam_alpha,
        adam_beta1     = adam_hyperparam_beta1,
        adam_beta2     = adam_hyperparam_beta2,
        adam_eps       = adam_hyperparam_eps,
        adam_precision = adam_handover_precision,
        adam_length    = adam_handover_length,
    )

    cov_inv = LT = None
    if central_value_fit_correlated or resample_fit_correlated:
        cov_inv, LT = _compute_cov_inv(ordinate, svdcut)

    # ------------------------------------------------------------------
    # Initialise FitResult
    # ------------------------------------------------------------------
    fit_result = FitResult(
        abscissa      = abscissa,
        Nresample     = Nres if resample_fit else None,
        resample_type = ordinate.resample_type if resample_fit else None,
    )

    # ------------------------------------------------------------------
    # Helper: build a cost function for a given (x, y, correlated) config
    # ------------------------------------------------------------------
    def _build_cost(x, y, correlated):
        if use_varproj:
            if correlated:
                return _build_varproj_correlated_cost(
                    x, y, cov_inv, model, nl_params, linear_params, prior, has_grad, LT
                )
            return _build_varproj_uncorrelated_cost(
                x, y, 1.0 / ordinate.serr, model, nl_params, linear_params, prior, has_grad
            )
        else:
            if correlated:
                return _build_correlated_cost(
                    x, y, cov_inv, model, nl_params, prior, has_grad, has_hess
                )
            return _build_uncorrelated_cost(
                x, y, 1.0 / ordinate.serr, model, nl_params, prior, has_grad, has_hess 
            )

    # Convenience: choose the right executor based on the path.
    _execute  = _execute_fits_vp   if use_varproj else _execute_fits_plain
    Ndata     = int(np.prod(ordinate.shape))

    # ------------------------------------------------------------------
    # Central value fit
    # ------------------------------------------------------------------
    if central_value_fit:
        x_cv = _get_abscissa(abscissa)
        y_cv = ordinate.mean
        cost = _build_cost(x_cv, y_cv, central_value_fit_correlated)
        if central_value_fit_correlated:
            W = cov_inv
        else:
            W = np.diag(1/ordinate.serr**2)
        fit_args_cv = {
            "least_square": cost,
            "p0":           start_vals,
            "param_names":  nl_params,
            "limits":       limits,
            "maxiter":      maxiter,
            "tol": tolerance,
            "strategy": strategy,
            **adam_kwargs,
        }
        res = _execute(fit_args_cv)
        if res["error"] is not None:
            raise res["error"]

        if use_varproj:
            fit_result.import_from_adam(
                cost_history=res["adam_chi2_history"]
            )
            fit_result.import_from_iminuit(
                minuit              = res["minuit"],
                model               = model,
                variable_projection = res["varproj"],
                variable_projection_hessians = res["varproj_hessians"],
                prior               = prior,
                cov = ordinate.cov,
                W = W,
                Ndata               = Ndata,
            )
        else:
            fit_result.import_from_adam(
                cost_history=res["adam_chi2_history"]
            )
            fit_result.import_from_iminuit(
                minuit = res["minuit"],
                model  = model,
                prior  = prior,
                cov = ordinate.cov,
                W = W,
                Ndata  = Ndata,
            )

        # fit_result.import_from_adam(
        #     cost_history = res["adam"]["cost_history"]
        # )

    if not resample_fit:
        return fit_result

    # ------------------------------------------------------------------
    # Build per-resample fit arguments
    # ------------------------------------------------------------------

    if central_value_fit:
        start_vals = {k: v.mean for k,v in fit_result.params.items()}

    args = np.empty(Nres, dtype=object)
    if resample_fit_correlated:
        W = cov_inv
    else:
        W = np.diag(1/ordinate.serr**2)

    for nres in range(Nres):
        x_rs = _get_abscissa(abscissa, nres)
        y_rs = ordinate.rspl[nres]
        cost = _build_cost(x_rs, y_rs, resample_fit_correlated)

        args[nres] = {
            "least_square": cost,
            "p0":           start_vals,
            "param_names":  nl_params,
            "limits":       limits,
            "maxiter":      maxiter,
            "tol":          tolerance,
            "strategy":     strategy,
            **adam_kwargs,
        }

    # ------------------------------------------------------------------
    # Execute resample fits (serial or parallel)
    # ------------------------------------------------------------------
    if Nproc is None:
        out = _execute(args, nres=list(range(Nres)))

        if use_varproj:
            errors = [
                (i, out["error"][i]) for i in range(Nres)
                if out["error"][i] is not None
            ]
            if errors:
                nres, err = errors[0]
                raise RuntimeError(f"Resample fit failed at nres={nres}: {err}") from err
            for nres in range(Nres):
                fit_result.import_from_adam(
                    cost_history = out["adam_chi2_history"][nres],
                    nres = nres
                )
                fit_result.import_from_iminuit(
                    out["minuit"][nres],
                    model               = model,
                    variable_projection = out["varproj"][nres],
                    variable_projection_hessians = out["varproj_hessians"][nres],
                    prior               = prior,
                    Ndata               = Ndata,
                    cov                 = ordinate.cov,
                    W                   = W,
                    nres                = nres,
                )
        else:
            errors = [
                (i, out["error"][i]) for i in range(Nres)
                if out["error"][i] is not None
            ]
            if errors:
                nres, err = errors[0]
                raise RuntimeError(f"Resample fit failed at nres={nres}: {err}") from err
            for nres in range(Nres):
                fit_result.import_from_adam(
                    cost_history = out["adam_chi2_history"][nres],
                    nres = nres
                )
                fit_result.import_from_iminuit(
                    out["minuit"][nres],
                    model  = model,
                    prior  = prior,
                    Ndata  = Ndata,
                    cov    = ordinate.cov,
                    W      = W,
                    nres   = nres,
                )

    else:
        # Parallel path — reuse the same parallel helpers as the plain backends.
        if use_varproj:
            flat_vp = _execute_parallel_varproj(args, Nres, Nproc)
            errors  = [(nres, err) for nres, _, _, _, _, err in flat_vp if err is not None]

            if errors:
                for nres, err in errors:
                    print(f"Resample fit failed at nres={nres}: {err}")
                raise RuntimeError(f"{len(errors)} resample fit(s) failed.")

            for nres, minuit, varproj, varproj_hessians, adam, _ in flat_vp:
                fit_result.import_from_adam(
                    cost_history = adam,
                    nres = nres
                )
                fit_result.import_from_iminuit(
                    minuit,
                    model               = model,
                    variable_projection = varproj,
                    variable_projection_hessians = varproj_hessians,
                    prior               = prior,
                    cov    = ordinate.cov,
                    W      = W,
                    Ndata               = Ndata,
                    nres                = nres,
                )
        else:
            flat = _execute_parallel_plain(args, Nres, Nproc)
            errors = [(nres, err) for nres, _, _, err in flat if err is not None]

            if errors:
                for nres, err in errors:
                    print(f"Resample fit failed at nres={nres}: {err}")
                raise RuntimeError(f"{len(errors)} resample fit(s) failed.")
            
            for nres, minuit, adam, _ in flat:
                fit_result.import_from_adam(
                    cost_history = adam,
                    nres = nres
                )
                fit_result.import_from_iminuit(
                    minuit,
                    model  = model,
                    prior  = prior,
                    cov    = ordinate.cov,
                    W      = W,
                    Ndata  = Ndata,
                    nres   = nres,
                )

    return fit_result