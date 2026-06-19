from collections.abc import Callable

import numpy as np
import iminuit
import multiprocess as mp

from .data import Data
from .prior import Prior
from .fitResult import FitResult

from .fit_helper import _validate_inputs, _get_p0, _get_abscissa, _compute_cov_inv, _jacobian_fd

# =============================================================================
# Variable-projection cost function constructors
# =============================================================================

def _build_varproj_uncorrelated_cost(abscissa, y_data, sdev_inv, model, nonlinear_params, linear_params, priors=None, model_has_grad=False):
    """
    Uncorrelated variable-projection (Golub-Pereyra) cost function.

    Linear parameters are eliminated analytically at every evaluation by
    solving the weighted least-squares problem

        min_c  || W (y - Phi(nl) c) ||^2

    where Phi is the design matrix and W = diag(sdev_inv).
    """
    x = np.asarray(abscissa)
    y = np.asarray(y_data)
    w = np.asarray(sdev_inv)

    priors         = priors or {}
    linear_params  = list(linear_params)
    nonlinear_params = list(nonlinear_params)

    if not linear_params:
        raise ValueError("Variable projection requires at least one linear parameter.")

    def _design_matrix(nl_dict):
        """Weighted design matrix A (N x n_lin) and RHS b (N,)."""
        n_lin = len(linear_params)
        A = np.zeros((len(x), n_lin))
        for j, lp in enumerate(linear_params):
            test = {**nl_dict, **{p: 0.0 for p in linear_params}, lp: 1.0}
            A[:, j] = model(x, test)
        return A * w[:, None], w * y

    def _solve(A, b):
        try:
            return np.linalg.solve(A.T @ A, A.T @ b)
        except np.linalg.LinAlgError:
            coeffs, *_ = np.linalg.lstsq(A, b, rcond=1e-9)
            return coeffs

    def cost(*nl_values):
        nl_dict = dict(zip(nonlinear_params, nl_values))
        A, b    = _design_matrix(nl_dict)
        coeffs  = _solve(A, b)
        params  = {**dict(zip(linear_params, coeffs)), **nl_dict}
        r       = w * (y - model(x, params))
        chi2    = np.sum(r ** 2)
        chi2   += sum(priors[k](nl_dict[k]) for k in nl_dict if k in priors)
        return chi2

    def grad(*nl_values):
        nl_dict = dict(zip(nonlinear_params, nl_values))
        A, b    = _design_matrix(nl_dict)
        coeffs  = _solve(A, b)
        params  = {**dict(zip(linear_params, coeffs)), **nl_dict}
        drsqdp  = -2.0 * w ** 2 * (y - model(x, params))
        J       = np.asarray(model.grad(x, params) * drsqdp)
        if J.ndim > 1:
            J = J.sum(axis=tuple(range(1, J.ndim)))
        if priors:
            for i, key in enumerate(nonlinear_params):
                if key in priors:
                    J[i] += priors[key].grad(params[key])
        return J

    cost.errordef  = iminuit.Minuit.LEAST_SQUARES
    cost.ndata     = len(x)
    if model_has_grad:
        cost.grad = grad

    # Store helpers so callers can recover linear parameter values after the fit
    cost._design_matrix    = _design_matrix
    cost._solve            = _solve
    cost._linear_params    = linear_params
    cost._nonlinear_params = nonlinear_params

    # Make iminuit see only the nonlinear parameter names in the signature
    cost.func_code = type("", (), {
        "co_varnames": tuple(nonlinear_params),
        "co_argcount": len(nonlinear_params),
    })()

    return cost


def _build_varproj_correlated_cost(abscissa, y_data, cov_inv, model, nonlinear_params, linear_params, priors=None, model_has_grad=False, LT=None):
    """
    Correlated variable-projection cost function.

    The covariance is whitened via C^{-1} = L L^T (Cholesky), giving the
    equivalent ordinary least-squares problem in the whitened space.
    """
    x     = np.asarray(abscissa)
    y     = np.asarray(y_data)
    C_inv = np.asarray(cov_inv)

    priors           = priors or {}
    linear_params    = list(linear_params)
    nonlinear_params = list(nonlinear_params)

    if not linear_params:
        raise ValueError("Variable projection requires at least one linear parameter.")

    if C_inv.shape != (len(y), len(y)):
        raise ValueError(f"cov_inv must be ({len(y)}, {len(y)}), got {C_inv.shape}")

    if LT is None:
        try:
            LT = np.linalg.cholesky(C_inv).T
        except np.linalg.LinAlgError:
            raise ValueError("cov_inv is not positive definite; Cholesky failed.")

    Lty = LT @ y  # pre-whiten once

    def _design_matrix(nl_dict):
        """Whitened design matrix A = L^T Phi and RHS b = L^T y."""
        n_lin = len(linear_params)
        Phi   = np.zeros((len(x), n_lin))
        for j, lp in enumerate(linear_params):
            test = {**nl_dict, **{p: 0.0 for p in linear_params}, lp: 1.0}
            Phi[:, j] = model(x, test)
        return LT @ Phi, Lty


    def _solve(A, b):
        try:
            return np.linalg.solve(A.T @ A, A.T @ b)
        except np.linalg.LinAlgError:
            coeffs, *_ = np.linalg.lstsq(A, b, rcond=1e-9)
            return coeffs

    def cost(*nl_values):
        nl_dict = dict(zip(nonlinear_params, nl_values))
        A, b    = _design_matrix(nl_dict)
        coeffs  = _solve(A, b)
        params  = {**dict(zip(linear_params, coeffs)), **nl_dict}
        r       = y - model(x, params)
        chi2    = r @ C_inv @ r
        chi2   += sum(priors[k](nl_dict[k]) for k in nl_dict if k in priors)
        return chi2

    def grad(*nl_values):
        nl_dict  = dict(zip(nonlinear_params, nl_values))
        A, b     = _design_matrix(nl_dict)
        coeffs   = _solve(A, b)
        params   = {**dict(zip(linear_params, coeffs)), **nl_dict}
        r        = y - model(x, params)
        d_chi2_df = -2.0 * (C_inv @ r)
        J        = np.asarray(model.grad(x, params) * d_chi2_df)
        if J.ndim > 1:
            J = J.sum(axis=tuple(range(1, J.ndim)))
        if priors:
            for i, key in enumerate(nonlinear_params):
                if key in priors:
                    J[i] += priors[key].grad(params[key])
        return J

    cost.errordef  = iminuit.Minuit.LEAST_SQUARES
    cost.ndata     = len(x)
    if model_has_grad:
        cost.grad = grad

    cost._design_matrix    = _design_matrix
    cost._solve            = _solve
    cost._linear_params    = linear_params
    cost._nonlinear_params = nonlinear_params

    cost.func_code = type("", (), {
        "co_varnames": tuple(nonlinear_params),
        "co_argcount": len(nonlinear_params),
    })()

    return cost


# =============================================================================
# Single-fit executor (variable projection)
# =============================================================================

def _run_minuit_varproj(least_square, p0, limits=None, maxiter = 10_000, tol = 0.1, strategy = 0):
    """
    Run Minuit on a variable-projection cost function and resolve the linear
    parameters at the best-fit nonlinear values.

    Returns (minuit, varproj_dict).
    """
    has_grad = hasattr(least_square, "grad")

    # p0 contains only nonlinear parameters
    nl_p0 = {k: v for k, v in p0.items() if k not in least_square._linear_params}

    minuit = iminuit.Minuit(
        least_square,
        **nl_p0,
        grad = least_square.grad if has_grad else None,
        name = list(nl_p0.keys()),
    )
    if has_grad:
        minuit.strategy = 0

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
    minuit.tol = tol
    minuit.strategy = strategy

    minuit.migrad(ncall=maxiter)
    minuit.hesse()

    # Resolve linear parameters at best-fit nonlinear values
    best_nl = {p: minuit.values[p] for p in least_square._nonlinear_params}
    A, b    = least_square._design_matrix(best_nl)
    coeffs  = least_square._solve(A, b)
    residual= b - A @ coeffs
    varproj = dict(zip(least_square._linear_params, coeffs))

    # We reevaluate the hessian form of the error here, as iminuit only
    # knew about the non-linear subspace
    linear_param_names = least_square._linear_params
    nonlinear_param_names = least_square._nonlinear_params
    cv = {}
    cv.update(**best_nl)
    cv.update(**varproj)
    param_names = list(cv.keys())

    AtA_inv = np.linalg.pinv(A.T @ A)
    cov_non_linear = minuit.covariance  
    
    if cov_non_linear is None:
        print(f"Is valid minimum: {minuit.fmin.is_valid}")
        print(f"Has accurate covariance: {minuit.fmin.has_covariance}")
        print(f"Is covariance pos-definite: {minuit.fmin.has_posdef_covar}")

    eps=1e-5
    dalpha_dbeta = np.zeros( (len(linear_param_names), len(nonlinear_param_names)) )
    for i, k in enumerate(nonlinear_param_names):
        nl_up = best_nl.copy()
        nl_dn = best_nl.copy()

        nl_up[k] += eps
        nl_dn[k] -= eps

        A_up, _ = least_square._design_matrix(nl_up)
        A_dn, _ = least_square._design_matrix(nl_dn)

        dA = (A_up - A_dn) / (2 * eps)

        dalpha_dbeta[:,i] = AtA_inv @ ( 
            dA.T @ residual - (A.T @ (dA @ coeffs))
        )

    cov_alpha_total = (
        AtA_inv
        + dalpha_dbeta @ cov_non_linear @ dalpha_dbeta.T
    )

    errors_linear = np.sqrt(np.diag(cov_alpha_total))
    errors_nonlinear = np.sqrt(np.diag(cov_non_linear))

    varproj_hessian = {}
    varproj_hessian.update({
        key: error for key,error in zip(linear_param_names, errors_linear)
    })
    varproj_hessian.update({
        key: error for key,error in zip(nonlinear_param_names, errors_nonlinear)
    })

    return minuit, varproj, varproj_hessian


def _execute_fits(fit_args, nres=None):
    """
    Run one or many variable-projection Minuit fits.

    Returns dict with keys 'nres', 'minuit', 'varproj', 'error'.
    """
    if nres is None:
        out = {"nres": None, "minuit": None, "varproj": None, "varproj_hessians": None, "error": None}
        try:
            minuit, varproj, varproj_hessians = _run_minuit_varproj(
                fit_args["least_square"],
                fit_args["p0"],
                fit_args.get("limits"),
                fit_args.get("maxiter",10_000),
                fit_args["tol"],
                fit_args["strategy"]
            )
            out["minuit"]  = minuit
            out["varproj"] = varproj
            out["varproj_hessians"] = varproj_hessians
        except Exception as e:
            out["error"] = e
        return out

    n = len(nres)
    out = {"nres": nres, "minuit": [None]*n, "varproj": [None]*n, "varproj_hessians": [None]*n, "error": [None]*n}
    for res_id in range(n):
        try:
            minuit, varproj, varproj_hessians = _run_minuit_varproj(
                fit_args[res_id]["least_square"],
                fit_args[res_id]["p0"],
                fit_args[res_id].get("limits"),
                fit_args[res_id].get("maxiter",10_000),
                fit_args[res_id]["tol"],
                fit_args[res_id]["strategy"]
            )
            out["minuit"][res_id]  = minuit
            out["varproj"][res_id] = varproj
            out["varproj_hessians"][res_id] = varproj_hessians
        except Exception as e:
            out["error"][res_id] = e
    return out


def _run_parallel_varproj(args, Nres, Nproc):
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
        results = pool.starmap(_execute_fits, inputs)

    flat = []
    for result in results:
        for res_id, nres in enumerate(result["nres"]):
            flat.append((
                nres,
                result["minuit"][res_id],
                result["varproj"][res_id],
                result["varproj_hessians"][res_id],
                result["error"][res_id],
            ))
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
    # Variable projection
    linear_params: list[str],
    # Model and parameters
    model: Callable | None = None,
    prior: dict[str, Prior] | None = None,
    p0: dict | None = None,
    limits: dict | None = None,
    svdcut: float | None = None,
    maxiter: int = 10_000,
    tolerance: int = 0.1,
    strategy: int = 0,
    # Parallelisation
    Nproc: int | None = None,
) -> FitResult:
    r"""
    Fit using iminuit with variable projection (Golub-Pereyra).

    Linear parameters are analytically eliminated at every cost function
    evaluation. iminuit only minimises over the nonlinear parameters.

    Parameters
    ----------
    abscissa : Data | np.ndarray
        Independent variable(s).
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
    linear_params : list[str]
        Parameter names that enter the model linearly.
    model : callable
        Model function ``model(abscissa, params_dict) -> np.ndarray``.
    prior : dict[str, Prior] | None
        Gaussian priors on *nonlinear* parameters. Priors on linear parameters
        are ignored with a warning.
    p0 : dict | None
        Initial parameter values. Used when prior is None.
    limits : dict | None
        Parameter bounds, e.g. ``{"E0": (0, None)}``.
    svdcut : float | None
        Relative SVD cut for the covariance matrix in correlated fits.
    maxiter : int
        Maximum Minuit iterations. (default: 10 000)
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
        Parallel processes for resample fits. Serial if None.

    Returns
    -------
    FitResult
    """
    if not (central_value_fit or resample_fit):
        raise ValueError("At least one of central_value_fit or resample_fit must be True.")

    _validate_inputs(abscissa, ordinate, model, prior, p0)

    # Silently drop priors on linear parameters — they are meaningless here
    if prior is not None:
        for key in list(linear_params):
            if key in prior:
                prior.pop(key)
                print(f"Warning: prior for linear parameter '{key}' was ignored.")

    Nres         = ordinate.Nresample
    has_grad     = hasattr(model, "grad")
    start_vals   = _get_p0(prior, p0)
    nl_params    = [k for k in start_vals if k not in linear_params]

    if has_grad:
        print("Using gradient information from model.grad")

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
            least_square = _build_varproj_correlated_cost(
                x_cv, y_cv, cov_inv, model, nl_params, linear_params, prior, has_grad, LT
            )
        else:
            least_square = _build_varproj_uncorrelated_cost(
                x_cv, y_cv, 1.0 / ordinate.serr, model, nl_params, linear_params, prior, has_grad
            )
            W = np.diag(1.0 / ordinate.serr**2)

        res = _execute_fits({
            "least_square": least_square, 
            "p0": start_vals, 
            "limits": limits, 
            "maxiter": maxiter,
            "tol": tolerance,
            "strategy": strategy,
        })
        if res["error"] is not None:
            raise res["error"]

        fit_result.import_from_iminuit(
            minuit=res["minuit"], 
            model=model, 
            variable_projection=res["varproj"],
            variable_projection_hessians=res["varproj_hessians"],
            prior=prior, 
            cov   = ordinate.cov,
            W     = W,
            Ndata=int(np.prod(ordinate.shape))
        )

    if not resample_fit:
        return fit_result

    # ------------------------------------------------------------------
    # Determine starting values for resample fits
    # ------------------------------------------------------------------
    if central_value_fit:
        rs_start = {k: v.mean for k,v in fit_result.params.items() if k not in linear_params}
    else:
        rs_start = {k: v for k, v in start_vals.items() if k not in linear_params}

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
            least_square = _build_varproj_correlated_cost(
                x_rs, y_rs, cov_inv, model, nl_params, linear_params, prior, has_grad, LT
            )
        else:
            least_square = _build_varproj_uncorrelated_cost(
                x_rs, y_rs, 1.0 / ordinate.serr, model, nl_params, linear_params, prior, has_grad
            )

        args[nres] = {
            "least_square": least_square, 
            "p0": rs_start, 
            "limits": limits, 
            "maxiter":maxiter,
            "tol": tolerance,
            "strategy": strategy,
        }

    # ------------------------------------------------------------------
    # Execute resample fits
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
                variable_projection=out["varproj"][nres],
                variable_projection_hessians=out["varproj_hessians"][nres],
                prior=prior, 
                cov = ordinate.cov,
                W = W,
                Ndata=int(np.prod(ordinate.shape)), 
                nres=nres
            )
    else:
        # run using a wrapper that returns varproj too
        flat_vp = _run_parallel_varproj(args, Nres, Nproc)
        errors  = [(nres, err) for nres, _, _, _, err in flat_vp if err is not None]
        if errors:
            for nres, err in errors:
                print(f"Resample fit failed at nres={nres}: {err}")
            raise RuntimeError(f"{len(errors)} resample fit(s) failed.")
        for nres, minuit, varproj, varproj_hessians, _ in flat_vp:
            fit_result.import_from_iminuit(
                minuit, 
                model=model, 
                variable_projection=varproj,
                variable_projection_hessians=varproj_hessians,
                prior=prior, 
                cov = ordinate.cov,
                W = W,
                Ndata=int(np.prod(ordinate.shape)), 
                nres=nres
            )

    return fit_result