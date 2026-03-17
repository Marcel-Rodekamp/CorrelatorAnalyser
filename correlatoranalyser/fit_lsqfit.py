from collections.abc import Callable

import numpy as np
import gvar as gv
import lsqfit
import multiprocess as mp
from dill import dumps, loads
import warnings

from .data import Data
from .prior import Prior
from .fitResult import FitResult
from .fit_helper import _validate_inputs, _get_p0, _get_abscissa, _compute_cov_inv

# =============================================================================
# lsqfit argument builder
# =============================================================================

def _build_lsqfit_args(abscissa, ordinate_gvar, model, prior, p0, correlated, svdcut, maxiter):
    """Return a kwargs dict ready to pass to lsqfit.nonlinear_fit."""
    args = {"fcn": model, "maxit": maxiter}

    if svdcut is not None:
        args["svdcut"] = svdcut

    if prior is not None:
        args["prior"] = {
            (k if prior[k].dist == "normal" else f"log({k})"): (
                prior[k].gvar()                          # normal  → gvar(mean, sdev)
                if prior[k].dist == "normal"
                else gv.gvar(prior[k].mean, prior[k].sdev)  # log-normal → plain gvar
            )
            for k in prior
        }
    else:
        args["p0"] = p0

    data_key = "data" if correlated else "udata"
    args[data_key] = (abscissa, ordinate_gvar)

    return args


# =============================================================================
# Single-fit executor
# =============================================================================

def _execute_fit(fit_args, nres=None, pickle=False):
    """
    Run one or many lsqfit.nonlinear_fit calls.

    Parameters
    ----------
    fit_args : dict | array of dict
        Single args dict (central value) or array indexed by resample.
    nres : None | list[int]
        None  → single central-value fit.
        list  → resample fits for those indices.
    pickle : bool
        Serialise the nlf objects with dill (needed for multiprocessing).

    Returns
    -------
    dict with keys 'nres', 'nlf', 'error'.
    """
    def _fit_one(args):
        return lsqfit.nonlinear_fit(**args)

    if nres is None:
        out = {"nres": None, "nlf": None, "error": None}
        try:
            nlf = _fit_one(fit_args)
            out["nlf"] = dumps(nlf) if pickle else nlf
        except Exception as e:
            out["error"] = e
        return out

    n   = len(nres)
    out = {"nres": nres, "nlf": [None] * n, "error": [None] * n}
    for res_id in range(n):
        try:
            nlf = _fit_one(fit_args[res_id])
            out["nlf"][res_id] = dumps(nlf) if pickle else nlf
        except Exception as e:
            out["error"][res_id] = e
    return out


def _execute_fit_parallel(fit_args, nres, pickle=True):
    """Wrapper with pickle=True for use in mp.Pool.starmap."""
    return _execute_fit(fit_args, nres=nres, pickle=pickle)


# =============================================================================
# Public interface
# =============================================================================

def fit_lsqfit(
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
    svdcut: float | None = None,
    maxiter: int = 10_000,
    # Parallelisation
    Nproc: int | None = None,
) -> FitResult:
    r"""
    Fit using lsqfit (Peter Lepage) as the backend.

    Parameters
    ----------
    abscissa : Data | np.ndarray
        Independent variable(s). If Data, resamples are used for resample fits.
    ordinate : Data
        Dependent variable with resample information.
    central_value_fit : bool
        Fit to the central value (mean) of the ordinate. (default: True)
    central_value_fit_correlated : bool
        Use the full covariance matrix for the central value fit. (default: False)
    resample_fit : bool
        Fit every resample. (default: False)
    resample_fit_correlated : bool
        Use the full covariance matrix for resample fits. (default: False)
    model : callable
        Model function with signature ``model(abscissa, params_dict) -> np.ndarray``.
        Must be compatible with gvar arithmetic.
    prior : dict[str, Prior] | None
        Gaussian or log-normal priors keyed by parameter name.
    p0 : dict | None
        Initial parameter values. Used when prior is None.
    svdcut : float | None
        SVD cut passed directly to lsqfit.
    maxiter : int
        Maximum iterations. (default: 10 000)
    Nproc : int | None
        Parallel processes for resample fits. Serial if None.

    Returns
    -------
    FitResult
    """
    if not (central_value_fit or resample_fit):
        raise ValueError("At least one of central_value_fit or resample_fit must be True.")

    _validate_inputs(abscissa, ordinate, model, prior, p0)

    if Nproc is not None and (central_value_fit_correlated or resample_fit_correlated):
        raise ValueError(
            "Correlated fits are not supported with parallel execution (Nproc is not None) "
            "in the lsqfit backend: gvar objects in the fit arguments cannot be safely "
            "pickled across process boundaries. Either set Nproc=None or use uncorrelated fits."
        )

    Nres       = ordinate.Nresample
    start_vals = _get_p0(prior, p0)

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
        x_cv          = _get_abscissa(abscissa)
        ordinate_gvar = ordinate.gvar(correlated=central_value_fit_correlated)
        if central_value_fit_correlated:
            W, _ = _compute_cov_inv(ordinate, svdcut=svdcut)
        else:
            W = np.diag(1/ordinate.serr**2)

        fit_args = _build_lsqfit_args(
            x_cv, ordinate_gvar, model, prior, start_vals,
            central_value_fit_correlated, svdcut, maxiter
        )

        res = _execute_fit(fit_args)
        if res["error"] is not None:
            raise res["error"]

        fit_result.import_from_lsqfit(nlf=res["nlf"], cov=ordinate.cov, W = W)

    if not resample_fit:
        return fit_result

    # ------------------------------------------------------------------
    # Build per-resample fit arguments
    # ------------------------------------------------------------------
    args = np.empty(Nres, dtype=object)

    if central_value_fit:
        start_vals = {k: v.mean for k,v in fit_result.params.items()}

    for nres in range(Nres):
        x_rs = _get_abscissa(abscissa, nres)

        ordinate_gvar = gv.gvar(
            ordinate.rspl[nres],
            ordinate.cov if resample_fit_correlated else ordinate.serr,
        )
        if central_value_fit_correlated:
            W, _ = _compute_cov_inv(ordinate, svdcut=svdcut)
        else:
            W = np.diag(1/ordinate.serr**2)

        args[nres] = _build_lsqfit_args(
            x_rs, ordinate_gvar, model, prior, start_vals,
            resample_fit_correlated, svdcut, maxiter
        )

    # ------------------------------------------------------------------
    # Execute resample fits (serial or parallel)
    # ------------------------------------------------------------------
    if Nproc is None:
        out = _execute_fit(args, nres=list(range(Nres)))
        errors = [(n, out["error"][n]) for n in range(Nres) if out["error"][n] is not None]
        if errors:
            n, err = errors[0]
            raise RuntimeError(f"Resample fit failed at nres={n}: {err}") from err
        for nres in range(Nres):
            fit_result.import_from_lsqfit(out["nlf"][nres], cov=ordinate.cov, W = W, nres=nres)

    else:
        blockSize = Nres // Nproc
        Nrest     = Nres % Nproc

        slices = [np.s_[b * blockSize : (b + 1) * blockSize] for b in range(Nproc)]
        if Nrest:
            slices.append(np.s_[Nproc * blockSize :])

        inputs = [
            (args[sl], np.arange(Nres)[sl].tolist(), True)
            for sl in slices
        ]

        with warnings.catch_warnings():
            # we explicitly ignore correlations otherwise 
            # an error would have been thrown earlier. 
            # Thus we can ignore the warnings here
            warnings.filterwarnings(
                "ignore",
                message="Pickling GVars.*loses correlations",
                category=UserWarning,
            )
            with mp.Pool(processes=Nproc) as pool:
                results = pool.starmap(_execute_fit_parallel, inputs)

        errors = []
        for result in results:
            for res_id, nres in enumerate(result["nres"]):
                if result["error"][res_id] is not None:
                    errors.append((nres, result["error"][res_id]))
                    continue
                try:
                    nlf = loads(result["nlf"][res_id])
                    fit_result.import_from_lsqfit(nlf=nlf, cov=ordinate.cov, W = W, nres=nres)
                except Exception as e:
                    errors.append((nres, f"import_from_lsqfit failed: {e}"))

        if errors:
            for nres, err in errors:
                print(f"Resample fit failed at nres={nres}: {err}")
            raise RuntimeError(f"{len(errors)} resample fit(s) failed.")

    return fit_result
