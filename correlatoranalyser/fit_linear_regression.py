import numpy as np

from .data import Data
from .fitResult import FitResult
from .fit_helper import _get_abscissa, _compute_cov_inv

# =============================================================================
# Core solver
# =============================================================================

def _solve_wls(y, X, W, S=None):
    """
    Weighted least-squares solve: theta = (X^T W X)^{-1} X^T W y.

    Parameters
    ----------
    y : np.ndarray, shape (N,)
        Observations.
    X : np.ndarray, shape (N, p)
        Design matrix.
    W : np.ndarray, shape (N, N)
        Weight matrix (inverse covariance or diagonal of inverse variances).
    S : np.ndarray | None, shape (p, N)
        Pre-computed solution matrix (X^T W X)^{-1} X^T W.
        If provided, X and W are ignored and the solve is skipped.

    Returns
    -------
    params : np.ndarray, shape (p,)
    S      : np.ndarray, shape (p, N)   — solution matrix for reuse
    """
    if S is None:
        S = np.linalg.inv(X.T @ W @ X) @ X.T @ W
    return S @ y, S


def _build_design_matrix(abscissa, has_intercept):
    """Build the (N, p) design matrix from abscissa."""
    x = np.asarray(abscissa)
    if has_intercept:
        return np.column_stack((x, np.ones_like(x)))
    return x.reshape(-1, 1)


def _build_weight_matrix(ordinate, correlated, svdcut=None):
    """Return the (N, N) weight matrix W = C^{-1} or diag(1/sigma^2)."""
    if correlated:
        cov_inv, _ = _compute_cov_inv(ordinate, svdcut)
        return cov_inv
    return np.diag(1.0 / ordinate.serr ** 2)


# =============================================================================
# Public interface
# =============================================================================

def linear_regression(
    *,
    abscissa: Data | np.ndarray,
    ordinate: Data,
    # Fit strategy
    central_value_fit: bool = True,
    central_value_fit_correlated: bool = False,
    resample_fit: bool = False,
    resample_fit_correlated: bool = False,
    # Linear model options
    has_intercept: bool = True,
    parameter_names: tuple[str, ...] | None = None,
    svdcut: float | None = None,
) -> FitResult:
    r"""
    Closed-form weighted linear regression.

    Solves theta = (X^T W X)^{-1} X^T W y analytically — no iterative
    minimiser is used. For a model with intercept the design matrix is
    [x | 1], and parameter_names defaults to ("m", "b"). Without intercept
    it is [x] and parameter_names defaults to ("m",).

    Parameters
    ----------
    abscissa : Data | np.ndarray
        Independent variable(s). If Data, resamples are used for resample fits.
    ordinate : Data
        Dependent variable with resample information.
    central_value_fit : bool
        Fit to the central value (mean) of the ordinate. (default: True)
    central_value_fit_correlated : bool
        Use the full covariance matrix as the weight matrix. (default: False)
    resample_fit : bool
        Fit every resample. (default: False)
    resample_fit_correlated : bool
        Use the full covariance matrix for resample fits. (default: False)
    has_intercept : bool
        Include an intercept term in the model. (default: True)
    parameter_names : tuple[str, ...] | None
        Names for the fit parameters. Defaults to ("m", "b") with intercept,
        ("m",) without.
    svdcut : float | None
        Relative SVD cut for the covariance matrix in correlated fits.

    Returns
    -------
    FitResult
    """
    if not (central_value_fit or resample_fit):
        raise ValueError("At least one of central_value_fit or resample_fit must be True.")

    if not isinstance(ordinate, Data):
        raise TypeError(f"ordinate must be a Data object, got {type(ordinate)}")

    if isinstance(abscissa, Data) and abscissa.Nresample != ordinate.Nresample:
        raise ValueError(
            f"abscissa.Nresample ({abscissa.Nresample}) != "
            f"ordinate.Nresample ({ordinate.Nresample})"
        )

    abscissa_shape0 = abscissa.shape[0] if isinstance(abscissa, Data) else np.asarray(abscissa).shape[0]
    if ordinate.shape[0] != abscissa_shape0:
        raise ValueError(
            f"ordinate.shape[0] ({ordinate.shape[0]}) != "
            f"abscissa.shape[0] ({abscissa_shape0}): shapes must match along the data axis"
        )

    # Default parameter names
    if parameter_names is None:
        parameter_names = ("m", "b") if has_intercept else ("m",)

    expected_nparams = 2 if has_intercept else 1
    if len(np.unique(parameter_names)) != expected_nparams:
        raise ValueError(
            f"Expected {expected_nparams} parameter name(s) for "
            f"has_intercept={has_intercept}, got {len(np.unique(parameter_names))}: {parameter_names}"
        )

    Nres = ordinate.Nresample

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
        X_cv = _build_design_matrix(x_cv, has_intercept)
        W_cv = _build_weight_matrix(ordinate, central_value_fit_correlated, svdcut)

        params_cv, _ = _solve_wls(ordinate.mean, X_cv, W_cv)

        fit_result.import_from_linear_regression(
            target_data     = ordinate.mean,
            result_params   = params_cv,
            design_matrix   = X_cv,
            weight_matrix   = W_cv,
            parameter_names = parameter_names,
            nres            = None,
        )

    if not resample_fit:
        return fit_result

    # ------------------------------------------------------------------
    # Resample fits
    # ------------------------------------------------------------------
    # The weight matrix is the same for all resamples (it is derived from
    # the full-sample covariance / standard errors, not per-resample ones).
    # We therefore build it once and cache the solution matrix S so that
    # the expensive matrix inversion is only performed on the first resample.
    W_rs = _build_weight_matrix(ordinate, resample_fit_correlated, svdcut)

    # For a fixed abscissa the design matrix X is also identical for every
    # resample, so S = (X^T W X)^{-1} X^T W can be reused throughout.
    # For a Data abscissa X changes per resample, so S must be recomputed.
    abscissa_is_fixed = not isinstance(abscissa, Data)
    S_cached = None

    for nres in range(Nres):
        x_rs = _get_abscissa(abscissa, nres)
        X_rs = _build_design_matrix(x_rs, has_intercept)

        params_rs, S_cached = _solve_wls(
            ordinate.rspl[nres], X_rs, W_rs,
            S = S_cached if abscissa_is_fixed else None,
        )

        fit_result.import_from_linear_regression(
            target_data     = ordinate.rspl[nres],
            result_params   = params_rs,
            design_matrix   = X_rs,
            weight_matrix   = W_rs,
            parameter_names = parameter_names,
            nres            = nres,
        )

    return fit_result
