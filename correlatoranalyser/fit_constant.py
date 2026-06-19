import numpy as np

from .data import Data
from .fitResult import FitResult
from .fit_helper import _get_abscissa, _compute_cov_inv

# =============================================================================
# Core solver
# =============================================================================

def _solve(y, W):
    return np.sum( W @ y ) / np.sum(W)

def _build_weight_matrix(ordinate, correlated, svdcut=None):
    """Return the (N, N) weight matrix W = C^{-1} or diag(1/sigma^2)."""
    if correlated:
        cov_inv, _ = _compute_cov_inv(ordinate, svdcut)
        return cov_inv
    return np.diag(1.0 / ordinate.serr ** 2)

# =============================================================================
# Public interface
# =============================================================================

def fit_constant(
    *,
    abscissa: Data | np.ndarray | None,
    ordinate: Data,
    # Fit strategy
    central_value_fit: bool = True,
    central_value_fit_correlated: bool = False,
    resample_fit: bool = False,
    resample_fit_correlated: bool = False,
    # Linear model options
    constant_name: str | None = None,
    svdcut: float | None = None,
) -> FitResult:
    r"""
    Closed-form weighted constant fit.

    Assuming 
        d χ² / d θ = 0 
    
    with:
        χ² = (y - θ).T @ W @ (y - θ)
    Then 
        θ = sum( W @ y ) / sum( W )
    
    Parameters
    ----------
    abscissa : Data | np.ndarray | None
        Independent variable(s). If Data, resamples are used for resample fits.
        If None, integer list is assumed from 0 to len(ordinate)-1
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
    constant_name : str | None
        Name for the fit parameters. Defaults to "a"
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

    if abscissa is not None:
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
    else:
        abscissa = np.arange( ordinate.shape[0] )

    # Default parameter names
    if constant_name is None:
        constant_name = "a"

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
        W_cv = _build_weight_matrix(ordinate, central_value_fit_correlated, svdcut)

        params_cv = _solve(ordinate.mean, W_cv)

        fit_result.import_from_constant_fit(
            target_data     = ordinate.mean,
            result_constant = params_cv,
            weight_matrix   = W_cv,
            constant_name   = constant_name,
            cov             = ordinate.cov,
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

    for nres in range(Nres):

        params_rs = _solve(
            ordinate.rspl[nres], W_rs,
        )
        fit_result.import_from_constant_fit(
            target_data     = ordinate.rspl[nres],
            result_constant = params_rs,
            weight_matrix   = W_rs,
            constant_name   = constant_name,
            cov             = ordinate.cov,
            nres            = nres,
        )

    return fit_result
