from collections.abc import Callable

import numpy as np
import iminuit
import multiprocess as mp

from .data import Data
from .prior import Prior
from .fitResult import FitResult
from .fit_helper import _validate_inputs, _get_p0, _get_abscissa, _compute_cov_inv

# =============================================================================
# Cost function constructors
# =============================================================================

def _build_correlated_cost(abscissa, y_data, cov_inv, model, param_names, priors=None, model_has_grad=False):
    """Correlated (full covariance) chi-squared cost function for iminuit."""

    def cost(*args):
        params = dict(zip(param_names, args))
        delta  = y_data - model(abscissa, params)
        chi2   = np.einsum("i,ij,j", delta, cov_inv, delta)
        if priors:
            chi2 += sum(priors[k](params[k]) for k in priors if k in params)
        return chi2

    def grad(*args):
        params   = dict(zip(param_names, args))
        delta    = y_data - model(abscissa, params)
        # For chi² = delta^T C^{-1} delta, the gradient w.r.t. theta_k is
        #   d(chi²)/d(theta_k) = -2 (C^{-1} delta)^T · d(model)/d(theta_k)
        # C^{-1} delta is a plain matrix-vector product, unlike the uncorrelated
        # case where C^{-1} is diagonal and reduces to element-wise scaling.
        weighted = cov_inv @ delta          # shape (N,)  — the key difference
        dchi2_df = -2.0 * weighted          # shape (N,)
        J        = np.asarray(model.grad(abscissa, params) * dchi2_df)
        if J.ndim > 1:
            J = J.sum(axis=tuple(range(1, J.ndim)))
        if priors:
            for i, key in enumerate(param_names):
                if key in priors:
                    J[i] += priors[key].grad(params[key])
        return J

    cost.errordef = iminuit.Minuit.LEAST_SQUARES
    cost.ndata    = len(abscissa)

    if model_has_grad:
        cost.grad = grad

    return cost


def _build_uncorrelated_cost(abscissa, y_data, sdev_inv, model, param_names, priors=None, model_has_grad=False):
    """Uncorrelated (diagonal) chi-squared cost function for iminuit."""

    def cost(*args):
        params = dict(zip(param_names, args))
        delta  = y_data - model(abscissa, params)
        chi2   = np.sum((delta * sdev_inv) ** 2)
        if priors:
            chi2 += sum(priors[k](params[k]) for k in priors if k in params)
        return chi2

    def grad(*args):
        params  = dict(zip(param_names, args))
        y_fit   = model(abscissa, params)
        drsqdp  = -2.0 * sdev_inv ** 2 * (y_data - y_fit)
        J       = np.asarray(model.grad(abscissa, params) * drsqdp)
        if J.ndim > 1:
            J = J.sum(axis=tuple(range(1, J.ndim)))
        if priors:
            for i, key in enumerate(param_names):
                if key in priors:
                    J[i] += priors[key].grad(params[key])
        return J

    cost.errordef = iminuit.Minuit.LEAST_SQUARES
    cost.ndata    = len(abscissa)

    if model_has_grad:
        cost.grad = grad

    return cost


# =============================================================================
# Single-fit executor
# =============================================================================

def _run_minuit(least_square, p0, limits=None, maxiter=10_000):
    """Construct and minimise a single Minuit instance. Returns the Minuit object."""
    has_grad = hasattr(least_square, "grad")

    minuit = iminuit.Minuit(
        least_square,
        **p0,
        grad  = least_square.grad if has_grad else None,
        name  = list(p0.keys()),
    )
    if has_grad:
        minuit.strategy = 0   # skip gradient check — saves time

    if limits is not None:
        for key, limit in limits.items():
            if key in minuit.parameters:
                minuit.limits[key] = limit

    # minuit stops if EDM < 0.002 × tol 
    # where EDM is the estimated distance to the minimum (based on gradient and hessian)
    # by default tol = 0.1 
    # This typically means a chi^2 to converge to a minumum up to ~1e-4 if everything goes well
    # The 1e-10 allows to converge to ~1e-12
    # Note, that at this precision, the Hessian esitimate may become unstable rendering the
    # convergence criterium wrong. 
    # minuit.tol = 1e-10

    minuit.migrad(ncall=maxiter)
    return minuit


def _execute_fits(fit_args, nres=None):
    """
    Run one or many Minuit fits.

    Parameters
    ----------
    fit_args : dict | array of dict
        Single args dict (central value) or array indexed by resample.
    nres : None | list[int]
        None  → single central-value fit.
        list  → resample fits for those resample indices.

    Returns
    -------
    dict with keys 'nres', 'minuit', 'error'.
    """
    if nres is None:
        out = {"nres": None, "minuit": None, "error": None}
        try:
            out["minuit"] = _run_minuit(
                fit_args["least_square"],
                fit_args["p0"],
                fit_args.get("limits"),
                fit_args.get("maxiter", 10_000)
            )
        except Exception as e:
            out["error"] = e
        return out

    out = {"nres": nres, "minuit": [None] * len(nres), "error": [None] * len(nres)}
    for res_id in range(len(nres)):
        try:
            out["minuit"][res_id] = _run_minuit(
                fit_args[res_id]["least_square"],
                fit_args[res_id]["p0"],
                fit_args[res_id].get("limits"),
                fit_args[res_id].get("maxiter", 10_000)
            )
        except Exception as e:
            out["error"][res_id] = e
    return out


# =============================================================================
# Parallel execution helper (shared with variable-projection backend)
# =============================================================================

def _run_parallel(execute_fn, args, Nres, Nproc):
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
        results = pool.starmap(execute_fn, inputs)

    flat = []
    for result in results:
        for res_id, nres in enumerate(result["nres"]):
            flat.append((nres, result["minuit"][res_id], result["error"][res_id]))
    return flat



# =============================================================================
# Public interface
# =============================================================================

def fit_iminuit(
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
    limits: dict | None = None,
    svdcut: float | None = None,
    maxiter: int = 10_000,
    # Parallelisation
    Nproc: int | None = None,
) -> FitResult:
    r"""
    Fit using iminuit (Minuit2) as the minimiser.

    Parameters
    ----------
    abscissa : Data | np.ndarray
        Independent variable(s). If Data, resamples are used for resample fits.
    ordinate : Data
        Dependent variable with resample information.
    central_value_fit : bool
        Perform a fit to the central value (mean) of the ordinate. (default: True)
    central_value_fit_correlated : bool
        Use the full covariance matrix for the central value fit. (default: False)
    resample_fit : bool
        Perform a fit on every resample. (default: False)
    resample_fit_correlated : bool
        Use the full covariance matrix for resample fits. (default: False)
    model : callable
        Model function with signature ``model(abscissa, params_dict) -> np.ndarray``.
    prior : dict[str, Prior] | None
        Gaussian priors keyed by parameter name.
    p0 : dict | None
        Initial parameter values. Used when prior is None.
    limits : dict | None
        Parameter bounds passed to Minuit, e.g. ``{"E0": (0, None)}``.
    svdcut : float | None
        Relative SVD cut applied to the covariance matrix for correlated fits.
    maxiter : int
        Maximum number of Minuit iterations. (default: 10 000)
    Nproc : int | None
        Number of parallel processes for resample fits. Serial if None.

    Returns
    -------
    FitResult
    """
    if not (central_value_fit or resample_fit):
        raise ValueError("At least one of central_value_fit or resample_fit must be True.")

    _validate_inputs(abscissa, ordinate, model, prior, p0)

    Nres       = ordinate.Nresample
    has_grad   = hasattr(model, "grad")
    start_vals = _get_p0(prior, p0)
    param_names = list(start_vals.keys())

    if has_grad:
        print("Using gradient information from model.grad")

    # Pre-compute inverse covariance once if needed for any correlated fit
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
    # Central value fit
    # ------------------------------------------------------------------
    if central_value_fit:
        x_cv = _get_abscissa(abscissa)
        y_cv = ordinate.mean

        if central_value_fit_correlated:
            W = cov_inv
            least_square = _build_correlated_cost(x_cv, y_cv, W, model, param_names, prior, has_grad)
        else:
            least_square = _build_uncorrelated_cost(
                x_cv, y_cv, 1.0 / ordinate.serr, model, param_names, prior, has_grad
            )
            W = np.diag(1.0 / ordinate.serr**2)

        res = _execute_fits({"least_square": least_square, "p0": start_vals, "limits": limits, "maxiter": maxiter})
        if res["error"] is not None:
            raise res["error"]

        fit_result.import_from_iminuit(
            minuit= res["minuit"], 
            model = model, 
            cov   = ordinate.cov,
            W     = W,
            prior = prior, 
            Ndata = int(np.prod(ordinate.shape))
        )

    if not resample_fit:
        return fit_result

    # ------------------------------------------------------------------
    # Build per-resample fit arguments
    # ------------------------------------------------------------------
    args = np.empty(Nres, dtype=object)
    if resample_fit_correlated:
        W = cov_inv
    else:
        W = np.diag(1/ordinate.serr**2)

    for nres in range(Nres):
        x_rs = _get_abscissa(abscissa, nres)
        y_rs = ordinate.rspl[nres]

        if resample_fit_correlated:
            least_square = _build_correlated_cost(x_rs, y_rs, cov_inv, model, param_names, prior, has_grad)
        else:
            least_square = _build_uncorrelated_cost(
                x_rs, y_rs, 1.0 / ordinate.serr, model, param_names, prior, has_grad
            )

        args[nres] = {"least_square": least_square, "p0": start_vals, "limits": limits, "maxiter":maxiter}

    # ------------------------------------------------------------------
    # Execute resample fits (serial or parallel)
    # ------------------------------------------------------------------
    if Nproc is None:
        out = _execute_fits(args, nres=list(range(Nres)))
        errors = [(nres, out["error"][nres]) for nres in range(Nres) if out["error"][nres] is not None]
        if errors:
            nres, err = errors[0]
            raise RuntimeError(f"Resample fit failed at nres={nres}: {err}") from err
        for nres in range(Nres):
            fit_result.import_from_iminuit(
                out["minuit"][nres], 
                model=model, 
                prior=prior,
                cov = ordinate.cov,
                W = W,
                Ndata=int(np.prod(ordinate.shape)), nres=nres
            )
    else:
        flat = _run_parallel(_execute_fits, args, Nres, Nproc)
        errors = [(nres, err) for nres, _, err in flat if err is not None]
        if errors:
            for nres, err in errors:
                print(f"Resample fit failed at nres={nres}: {err}")
            raise RuntimeError(f"{len(errors)} resample fit(s) failed.")
        for nres, minuit, _ in flat:
            fit_result.import_from_iminuit(
                minuit, model=model, prior=prior,
                Ndata=int(np.prod(ordinate.shape)), nres=nres
            )

    return fit_result
