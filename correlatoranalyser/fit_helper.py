import numpy as np
from .data import Data

# =============================================================================
# Validation helpers (shared with all backends)
# =============================================================================

def _validate_inputs(abscissa, ordinate, model, prior, p0):
    """Raise informative errors for common misconfigurations."""

    if not isinstance(ordinate, Data):
        raise TypeError(f"ordinate must be a Data object, got {type(ordinate)}")

    if model is None:
        raise ValueError("A model function is required: model(abscissa, params) -> array")

    if prior is None and p0 is None:
        raise ValueError("Provide at least one of: prior or p0")

    if isinstance(abscissa, Data):
        if abscissa.Nresample != ordinate.Nresample:
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


def _get_p0(prior, p0):
    """Return a plain float dict of starting values, preferring prior means."""
    if prior is not None:
        return {k: prior[k].mean for k in prior}
    return dict(p0)


def _get_abscissa(abscissa, nres=None):
    """Return abscissa as a plain array for one resample (or the central value)."""
    if isinstance(abscissa, Data):
        return abscissa.rspl[nres] if nres is not None else abscissa.mean
    return abscissa


def _compute_cov_inv(ordinate, svdcut=None):
    """Invert the ordinate covariance, optionally applying an SVD cut."""
    if svdcut is None:
        return np.linalg.inv(ordinate.cov), None

    from scipy.linalg import eigh
    eigvals, eigvecs = eigh(ordinate.cov)
    threshold = svdcut * eigvals[-1]
    keep      = eigvals > threshold
    n_cut     = int((~keep).sum())
    print(f"svdcut={svdcut}: removing {n_cut} of {len(eigvals)} modes")

    inv_sqrt = np.where(keep, 1.0 / np.sqrt(np.where(keep, eigvals, 1.0)), 0.0)
    LT       = (eigvecs * inv_sqrt).T
    cov_inv  = (eigvecs * np.where(keep, 1.0 / np.where(keep, eigvals, 1.0), 0.0)) @ eigvecs.T
    return cov_inv, LT


def _jacobian_fd(model, abscissa, params: dict, rel_step: float = 1e-5) -> np.ndarray:
    """
    Central finite-difference Jacobian of model(abscissa, params).

    Returns
    -------
    J : np.ndarray, shape (Nparams, Nx)
        J[i, j] = dmodel_j / dtheta_i
    """
    keys   = list(params.keys())
    f0     = np.asarray(model(abscissa, params), dtype=float)
    J      = np.empty((len(keys), f0.size), dtype=float)
    for i, key in enumerate(keys):
        h               = rel_step * max(abs(params[key]), 1.0)
        p_fwd           = dict(params); p_fwd[key] += h
        p_bwd           = dict(params); p_bwd[key] -= h
        J[i]            = (np.asarray(model(abscissa, p_fwd), dtype=float)
                         - np.asarray(model(abscissa, p_bwd), dtype=float)) / (2.0 * h)
    return J