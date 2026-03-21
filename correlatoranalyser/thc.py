"""
thc.py
======
Truncated Hankel Correlator (THC) backend.

Implements the THC method from:
    J. Ostmeyer and C. Urbach, arXiv:2510.15500 (2025)

Algorithms 2 and 3 from the paper are implemented as private helpers
``_thc_energies`` and ``_thc_overlaps`` respectively.  The public entry
point ``thc(...)`` mirrors the interface of the other backends in this
library and populates a :class:`FitResult` object.

Unlike the other backends, THC is **not** an iterative minimiser.  It
solves an algebraic eigenvalue problem and therefore:

* ``model``, ``prior``, and ``p0`` are **not** required inputs.  The model
  is constructed internally after the energies are found and stored on
  ``FitResult.fcn`` for later use (e.g. ``eval()``, ``compute_expected_chi2``).

* ``central_value_fit_correlated`` / ``resample_fit_correlated`` control
  only the weight matrix used when evaluating chi2_obs and <chi2> **after**
  the energies and overlaps have been determined.  They do **not** affect
  the internal Hankel weight matrices Omega and W_outer.

Automatic truncation
--------------------
When ``truncation_dimension=None``, the truncation k is determined once
from the central-value Hankel spectrum and then used for every resample.
Two criteria are available via ``truncation_method``:

``"gap"``  (default, eq. 57)
    k = index of the largest ratio |s_i / s_{i+1}| in the sorted
    eigenspectrum.  Identifies the gap between the physical-signal subspace
    (large, well-separated eigenvalues) and the noise subspace (eigenvalues
    of order ~1 due to the Omega normalisation by 1/sigma).  Recommended for
    most applications; robust when noise modes happen to be positive.

``"kpos"`` (eq. 58)
    k = number of consecutive strictly positive leading eigenvalues.  Can
    overestimate when noise modes fluctuate positive, which is common with
    the default Omega weighting.  Useful as a cross-check on cleaner data.

Parameter naming in FitResult
------------------------------
Physical results (params):
    E0, A0, E1, A1, ...    energies and amplitudes, ascending energy order.

Algorithmic result (separate attribute):
    FitResult.lambda_eigs   complex eigenvalues Lambda of the transfer matrix
                            X, stored as a Data object of shape (k,).
                            Lambda_l ≈ exp(-E_l · delta_t) before filtering.
"""

from __future__ import annotations

import multiprocess as mp
import numpy as np
from scipy.linalg import hankel as scipy_hankel

from .data import Data
from .fitResult import FitResult


# =============================================================================
# Module-level model — must live at module scope so inspect.getsource works.
# Both functions are defined here; model.grad = grad is set after both defs
# so that _serialise_fcn can bundle the full source blob for HDF5 storage.
# =============================================================================

def model(t: np.ndarray, p: dict) -> np.ndarray:
    """
    Multi-exponential model: C(t) = sum_k A_k * exp(-E_k * t).

    The number of states is inferred from p by counting keys starting with
    'E'.  Parameters must be named E0, A0, E1, A1, ...
    """
    Nstates = sum(1 for key in p if key.startswith("E"))
    t_arr   = np.asarray(t, dtype=float)
    result  = np.zeros_like(t_arr)
    for k in range(Nstates):
        result = result + p[f"A{k}"] * np.exp(-p[f"E{k}"] * t_arr)
    return result


def grad(t: np.ndarray, p: dict) -> np.ndarray:
    """
    Jacobian of model w.r.t. its parameters, shape (2*Nstates, len(t)).

    Row order: dC/dE0, dC/dA0, dC/dE1, dC/dA1, ...
    """
    t_arr   = np.asarray(t, dtype=float)
    Nstates = sum(1 for key in p if key.startswith("E"))
    J       = np.zeros((2 * Nstates, len(t_arr)), dtype=float)
    for k in range(Nstates):
        exp_k        = np.exp(-p[f"E{k}"] * t_arr)
        J[2 * k]     = -p[f"A{k}"] * t_arr * exp_k   # dC/dE_k
        J[2 * k + 1] = exp_k                           # dC/dA_k
    return J


# Attach gradient so _serialise_fcn in fitResult.py bundles both functions
# into one source blob and the round-trip restores model.grad automatically.
model.grad = grad


# =============================================================================
# Private helpers
# =============================================================================

def _build_hankel_matrix(C: np.ndarray, n: int) -> np.ndarray:
    """
    Build the (n+1)x(n+1) square Hankel matrix H_hat.

    H_hat[i,j] = C[i+j],  0 <= i,j <= n.

    C must already be offset by t0 (i.e. pass C = full_C[t0:]).
    """
    return scipy_hankel(C[:n + 1], C[n: 2 * n + 1])


def _build_default_Omega(sigma: np.ndarray, n: int) -> np.ndarray:
    """
    Build the default inner weight matrix Omega (eq. 47, paper).

    For scalar d=1, Omega is (n+1)x(n+1) diagonal with

        Omega[i,i] = 1 / sqrt( sqrt(#(2i)) * sigma[t0 + 2i] )

    where the multiplicity #(2i) = 1 + 2*min(i, n-i) (eq. 46) accounts
    for the fact that time slice 2i appears more than once in H_hat.

    sigma must already be offset by t0 (i.e. pass sigma = full_sigma[t0:]).
    """
    omega_diag = np.empty(n + 1)
    for i in range(n + 1):
        mult          = 1 + 2 * min(i, n - i)   # eq. (46)
        omega_diag[i] = 1.0 / np.sqrt(np.sqrt(mult) * sigma[2 * i])
    return np.diag(omega_diag)


def _build_default_W_outer(
    sigma: np.ndarray,
    n: int,
    delta_t: int,
    symmetric: bool,
) -> np.ndarray:
    """
    Build the default outer weight matrix W (eqs. 29/30 or 33, paper).

    W is (n-delta_t)x(n-delta_t) diagonal.

    Non-symmetric (eq. 29/30):  W[i,i] = 1 / sqrt(sigma[t0 + delta_t + i])
    Symmetric     (eq. 33):     W[i,i] = sqrt(1/sigma^2[t0+i] + 1/sigma^2[t0+delta_t+i])

    sigma must already be offset by t0 (i.e. pass sigma = full_sigma[t0:]).
    """
    n_rows = n - delta_t
    if symmetric:
        s0  = sigma[:n_rows]
        sdt = sigma[delta_t:n_rows + delta_t]
        w   = np.sqrt(1.0 / s0**2 + 1.0 / sdt**2)
    else:
        w = 1.0 / np.sqrt(sigma[delta_t: n_rows + delta_t])
    return np.diag(w)


def _hankel_eigh(
    C: np.ndarray,
    Omega: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute and sort the eigendecomposition of H_tilde = Omega * H_hat * Omega†.

    Returns eigenvalues and eigenvectors sorted descending by |eigenvalue|.
    Factored out so both ``_determine_k`` and ``_thc_energies`` share a
    single diagonalisation call.

    Returns
    -------
    eigenvalues  : np.ndarray, shape (n+1,)   real, sorted desc by |s|
    eigenvectors : np.ndarray, shape (n+1, n+1)

    C must already be offset by t0 (i.e. pass C = full_C[t0:]).
    """
    H_hat   = _build_hankel_matrix(C, n)
    H_tilde = Omega @ H_hat @ Omega.T
    eigenvalues, eigenvectors = np.linalg.eigh(H_tilde)   # real eigs
    idx          = np.argsort(np.abs(eigenvalues))[::-1]
    return eigenvalues[idx], eigenvectors[:, idx]


def _determine_k(
    C: np.ndarray,
    Omega: np.ndarray,
    n: int,
    method: str,
) -> int:
    """
    Determine truncation dimension k from the central-value Hankel spectrum.

    Parameters
    ----------
    method : str
        ``"gap"``  — k_gap (eq. 57): index of the largest ratio
                     |s_i / s_{i+1}| in the sorted spectrum.  Finds the gap
                     between the physical and noise subspaces.  Robust when
                     noise modes have magnitude ~1 due to Omega normalisation.

        ``"kpos"`` — k_pos (eq. 58): count of consecutive strictly positive
                     leading eigenvalues.  Can overestimate when noise modes
                     fluctuate positive.

    Returns
    -------
    int
        Resolved k, at least 1.


    C must already be offset by t0 (i.e. pass C = full_C[t0:]).
    """
    eigenvalues, _ = _hankel_eigh(C, Omega, n)

    if method == "gap":
        abs_eigs = np.abs(eigenvalues)
        ratios   = abs_eigs[:-1] / np.where(abs_eigs[1:] > 0, abs_eigs[1:], 1e-30)
        k        = int(np.argmax(ratios)) + 1   # +1: count before the gap
    elif method == "kpos":
        k = 0
        for s in eigenvalues:
            if s > 0:
                k += 1
            else:
                break
    else:
        raise ValueError(
            f"truncation_method must be 'gap' or 'kpos', got '{method}'."
        )

    return max(k, 1)


def _thc_energies(
    C: np.ndarray,
    Omega: np.ndarray,
    W_outer: np.ndarray,
    n: int,
    delta_t: int,
    k: int,
    symmetric: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Algorithm 2 — extract raw energies from the truncated Hankel matrix.

    Parameters
    ----------
    k : int
        Truncation dimension.  Always a resolved integer; auto-detection
        is handled upstream by ``_determine_k`` before this is called.

    Returns
    -------
    energies : np.ndarray (complex), shape (k,)
        E_l = -log(Lambda_l) / delta_t  before filtering.
    Lambda : np.ndarray (complex), shape (k,)
        Raw eigenvalues of X: Lambda_l ≈ exp(-E_l * delta_t).

    C must already be offset by t0 (i.e. pass C = full_C[t0:]).
    """
    eigenvalues, eigenvectors = _hankel_eigh(C, Omega, n)

    k  = min(k, n + 1)
    Uk = eigenvectors[:, :k]    # (n+1, k)

    # Build M0 and M_delta_t (eq. 26)
    # Omega is diagonal so its inverse is trivial
    Omega_inv    = np.diag(1.0 / np.diag(Omega))
    Omega_inv_Uk = Omega_inv @ Uk                                        # (n+1, k)

    n_rows = n - delta_t
    w_vec  = np.diag(W_outer)                                            # (n_rows,)
    M0     = w_vec[:, None] * Omega_inv_Uk[:n_rows, :]                   # rows 0..n-dt-1
    Mdt    = w_vec[:, None] * Omega_inv_Uk[delta_t: delta_t + n_rows, :] # rows dt..n-1

    # Solve for transfer matrix X (eq. 28 or eq. 31)
    if symmetric:
        Mbar = 0.5 * (M0 + Mdt)
        X    = np.linalg.solve(Mbar.T @ M0, Mbar.T @ Mdt)   # eq. (31)
    else:
        X    = np.linalg.solve(M0.T @ M0, M0.T @ Mdt)       # eq. (28)

    # Diagonalise X: Lambda_l ≈ exp(-E_l * delta_t)
    Lambda, _ = np.linalg.eig(X)   # X is not symmetric -> eig
    Lambda     = Lambda.astype(complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        energies_out = -np.log(Lambda) / delta_t

    return energies_out, Lambda


def _filter_energies(
    energies: np.ndarray,
    epsilon_real: float,
    epsilon_imag: float,
) -> np.ndarray:
    """
    Select physical energies from the raw (complex) eigenvalue spectrum.

    Keep E if:  |Im(E)| < epsilon_imag  AND  Re(E) > epsilon_real.

    Returns surviving real parts sorted ascending.
    """
    physical = [
        E.real
        for E in energies
        if abs(E.imag) < epsilon_imag and E.real > epsilon_real
    ]
    return np.sort(np.asarray(physical, dtype=float))


def _thc_overlaps(
    C: np.ndarray,
    t_fit: np.ndarray,
    sigma: np.ndarray,
    energies: np.ndarray,
) -> np.ndarray:
    """
    Algorithm 3 — amplitude reconstruction via weighted least squares.

    Solves  min_A  sum_t (C(t) - sum_l A_l exp(-E_l t))^2 / sigma^2_t

    using the numerically stable Vandermonde split (eqs. 38-40):

        chi = chi0 * chi_D,  chi_D[l,l] = max_t |chi[t,l]|

    This normalises every column of chi0 to max 1, preventing catastrophic
    overflow/underflow for large T or large E.  The split is ALWAYS applied.

    Returns
    -------
    A : np.ndarray, shape (Nstates,)
        Amplitude coefficients A_l.

    C,sigma,t_fit must already be offset by t0.
    """
    # Vandermonde: chi[t,l] = exp(-E_l * t),  shape (T_total-t0, Nstates)
    chi = np.exp(-energies[None, :] * t_fit[:, None])

    # Scale split (eqs. 38-40): always applied for numerical safety
    chi_D_diag = np.max(np.abs(chi), axis=0)    # max per column
    chi_D_inv  = 1.0 / chi_D_diag
    chi0       = chi * chi_D_inv[None, :]        # every column bounded by 1

    # Weighted normal equations W = diag(1/sigma^2)
    w     = 1.0 / sigma**2
    A_mat = chi0.T @ (w[:, None] * chi0)         # (Nstates, Nstates)
    b_vec = chi0.T @ (w * C)                # (Nstates,)

    # A = chi_D^{-1} * solve(A_mat, b_vec)
    A = chi_D_inv * np.linalg.solve(A_mat, b_vec)

    return A
    


def _compute_chi2(
    C: np.ndarray,
    abscissa: np.ndarray,
    energies: np.ndarray,
    overlaps: np.ndarray,
    W_chi2: np.ndarray,
) -> float:
    """
    Compute chi2_obs = r^T W r,  r = C(t) - model(t, params).

    Returns nan if any parameter is nan (fewer physical states than Nstates).
    """
    if np.any(np.isnan(energies)) or np.any(np.isnan(overlaps)):
        return float("nan")
    Nstates = len(energies)
    params  = {f"E{k}": energies[k] for k in range(Nstates)}
    params.update({f"A{k}": overlaps[k] for k in range(Nstates)})
    t0   = int(abscissa[0])
    pred = model(abscissa, params)
    r    = C - pred
    return float(r @ W_chi2 @ r)


def _run_one_thc(
    C: np.ndarray,
    sigma: np.ndarray,
    Omega: np.ndarray,
    W_outer: np.ndarray,
    t_fit:np.ndarray,
    n: int,
    delta_t: int,
    k: int,
    symmetric: bool,
    epsilon_real: float,
    epsilon_imag: float,
    Nstates: int,
) -> dict:
    """
    Run Algorithms 2 + 3 for a single data vector.

    Parameters
    ----------
    k : int
        Truncation dimension.  Always a resolved integer; determined upstream
        by ``_determine_k`` or supplied directly by the user.

    Returns a dict with keys:
        energies       - physical energies, shape (Nstates,), np.nan padded
        overlaps       - amplitudes,        shape (Nstates,), np.nan padded
        lambda_eigvals - complex Lambda eigenvalues of X, shape (k,)
        n_physical     - number of physical states found before padding
    """
    raw_E, Lambda = _thc_energies(
        C, Omega, W_outer, n, delta_t, k, symmetric, 
    )
    phys   = _filter_energies(raw_E, epsilon_real, epsilon_imag)
    n_phys = len(phys)
    A      = _thc_overlaps(C, t_fit, sigma, phys) if n_phys > 0 else np.array([])

    # Pad / truncate physical results to exactly Nstates entries
    E_out = np.full(Nstates, np.nan)
    A_out = np.full(Nstates, np.nan)
    n_s   = min(n_phys, Nstates)
    E_out[:n_s] = phys[:n_s].real
    A_out[:n_s] = A[:n_s].real

    return {
        "energies":    E_out,
        "overlaps":    A_out,
        "lambda_eigvals": Lambda,
        "n_physical":  n_phys,
    }


# =============================================================================
# Parallel worker
# =============================================================================

def _thc_block(
    nres_list: list[int],
    rspl: np.ndarray,
    sigma: np.ndarray,
    Omega: np.ndarray,
    W_outer: np.ndarray,
    t_fit:np.ndarray,
    n: int,
    delta_t: int,
    k: int,
    symmetric: bool,
    epsilon_real: float,
    epsilon_imag: float,
    Nstates: int,
) -> list[tuple[int, dict]]:
    """Multiprocess worker: run a block of resample indices."""
    return [
        (nres, _run_one_thc(
            rspl[nres], sigma, Omega, W_outer, t_fit,
            n, delta_t, k, symmetric,
            epsilon_real, epsilon_imag, Nstates,
        ))
        for nres in nres_list
    ]


# =============================================================================
# Public interface
# =============================================================================

def thc(
    *,
    ordinate: Data,
    # Fit strategy — flags only control the chi2 evaluation, not the algorithm
    central_value_fit: bool = True,
    central_value_fit_correlated: bool = False,
    resample_fit: bool = True,
    resample_fit_correlated: bool = False,
    # THC algorithm parameters
    delta_t: int = 1,
    truncation_dimension: int | None = None,
    truncation_method: str = "gap",
    symmetric_correlator: bool = False,
    Omega: np.ndarray | None = None,
    W_outer: np.ndarray | None = None,
    epsilon_real: float = 1e-8,
    epsilon_imag: float = 1e-12,
    t0: int = 0,
    # Parallelisation (over resamples only)
    Nproc: int | None = None,
) -> FitResult:
    r"""
    Truncated Hankel Correlator (THC) method (arXiv:2510.15500).

    Extracts energy levels and amplitudes from a Euclidean correlator via
    eigenvalue analysis of a weighted truncated Hankel matrix.  This is a
    closed-form algebraic method — no model function, prior, or starting
    parameters are required.

    The ``correlated`` flags control only the weight matrix W used when
    computing chi2_obs and <chi2> *after* the energies are known.  They
    do NOT affect the internal Hankel weights Omega or W_outer.

    When ``truncation_dimension=None``, k is determined once from the
    central-value Hankel spectrum via ``truncation_method`` and then applied
    to every resample, guaranteeing a consistent bootstrap distribution.

    Parameters
    ----------
    ordinate : Data
        Correlator data C(t), shape (T_total,).
        T_eff = T_total - 1 - t0 must be even.
    central_value_fit : bool
        Run THC on the central-value mean. (default: True)
    central_value_fit_correlated : bool
        Use full covariance for chi2/expected_chi2 evaluation. (default: False)
    resample_fit : bool
        Run THC on every resample. (default: False)
    resample_fit_correlated : bool
        Same as above for resample chi2. (default: False)
    delta_t : int
        Shift parameter delta_t in Algorithm 2. (default: 1)
    truncation_dimension : int | None
        Truncation k.

        * ``int``  — use exactly this value for CV and all resamples.
        * ``None`` — determine k automatically once from the central-value
          Hankel spectrum using ``truncation_method``, then reuse that k
          for all resamples. (default: None)
    truncation_method : str
        Criterion used when ``truncation_dimension=None``.

        ``"gap"`` (default, eq. 57)
            k = index of the largest ratio |s_i / s_{i+1}|.  Identifies the
            gap between the physical-signal subspace and the noise subspace.
            Robust when noise modes have magnitude ~1 (common with the default
            Omega weighting).  Recommended for most applications.

        ``"kpos"`` (eq. 58)
            k = number of consecutive strictly positive leading eigenvalues.
            Can overestimate when noise modes fluctuate positive.  Useful as
            a cross-check on cleaner data.

    symmetric_correlator : bool
        Use the symmetrised estimator (eq. 31) for C(t) = C(T-t).
        Guarantees the energy spectrum is symmetric about zero. (default: False)
    Omega : np.ndarray | None
        Inner weight matrix, shape (n+1, n+1) where n = T_eff//2.
        None -> default eq. (47). (default: None)
    W_outer : np.ndarray | None
        Outer weight matrix, shape (n-delta_t, n-delta_t).
        None -> default eq. (29/30) or eq. (33) for symmetric. (default: None)
    epsilon_real : float
        Minimum Re(E) to keep; filters spurious E~0 states. (default: 1e-8)
    epsilon_imag : float
        Maximum |Im(E)| to keep; filters complex noise states. (default: 1e-12)
    t0 : int
        First timeslice included in the Hankel matrix.  Set t0 > 0 to skip
        early timeslices contaminated by excited states. (default: 0)
    Nproc : int | None
        Parallel processes for resample fits.  None -> serial. (default: None)

    Returns
    -------
    FitResult
        params             : E0, A0, E1, A1, ... (physical results)
        lambda_eigs        : Data of shape (k,)  (algorithmic result)
        chi2, p_value, AIC : fit quality
        expected_chi2      : <chi2> from Bruno & Sommer (arXiv:2209.14188)
        fcn                : multi-exponential model with .grad attached
    """
    if not (central_value_fit or resample_fit):
        raise ValueError("At least one of central_value_fit or resample_fit must be True.")
    if not isinstance(ordinate, Data):
        raise TypeError(f"ordinate must be a Data object, got {type(ordinate)}")
    if truncation_method not in ("gap", "kpos", None):
        raise ValueError(
            f"truncation_method must be 'gap' or 'kpos', got '{truncation_method}'."
        )

    T_total = ordinate.shape[0]
    T_eff   = T_total - 1 - t0

    if T_eff <= 0:
        raise ValueError(
            f"t0={t0} leaves no usable data (T_total={T_total}, T_eff={T_eff} <= 0)."
        )
    if T_eff % 2 != 0:
        raise ValueError(
            f"T_eff = T_total - 1 - t0 = {T_eff} must be even. "
            f"Adjust t0 (currently {t0}) or the length of ordinate ({T_total})."
        )

    if t0 != 0:
        ordinate = ordinate[t0:]

    n = T_eff // 2   # Hankel matrix will be (n+1) x (n+1)

    if isinstance(truncation_dimension, int):
        min_T = 2 * truncation_dimension * delta_t + t0 + 1
        if T_total < min_T:
            raise ValueError(
                f"truncation_dimension={truncation_dimension} with delta_t={delta_t} "
                f"and t0={t0} requires at least {min_T} timeslices; "
                f"ordinate has {T_total}."
            )
        if truncation_dimension > n + 1:
            raise ValueError(
                f"truncation_dimension={truncation_dimension} exceeds the "
                f"Hankel matrix size n+1={n + 1}."
            )

    sigma = ordinate.serr               # (T_total,)
    Nres  = ordinate.Nresample

    # --- Build weight matrices (user override or paper defaults) ---
    Omega_use   = (
        Omega   if Omega   is not None
        else _build_default_Omega(sigma, n)
    )
    W_outer_use = (
        W_outer if W_outer is not None
        else _build_default_W_outer(sigma, n, delta_t, symmetric_correlator)
    )

    # --- Resolve truncation dimension k upfront from central-value data ---
    # When the user supplies an integer it is used directly.
    # When None, k is determined once from the CV Hankel spectrum using the
    # selected criterion, then reused for every resample to guarantee a
    # consistent bootstrap ensemble (all resamples use the same model order).
    if not isinstance(truncation_dimension, int) and truncation_method is None:
        raise RuntimeError("THC requires eather tuncation_dimension:int or a method to determin it (truncation_method)")

    elif not isinstance(truncation_dimension, int):
        truncation_dimension = _determine_k(ordinate.mean, Omega_use, n, truncation_method)
    else:
        # ignore truncation method and use truncation_dimension
        pass 

    Nstates = truncation_dimension

    # Abscissa = timeslices actually used in chi2 evaluation
    abscissa = np.arange(t0, T_total, dtype=float)

    # --- Initialise FitResult ---
    fit_result = FitResult(
        abscissa      = abscissa,
        Nresample     = Nres if resample_fit else None,
        resample_type = ordinate.resample_type if resample_fit else None,
    )
    fit_result.fcn = model
    # dof from eq. (53), scalar d=1: (T_eff + 1) - 2*Nstates
    fit_result.dof = (T_eff + 1) - 2 * Nstates
    max_k = truncation_dimension   # Lambda storage size equals the resolved truncation dimension

    def _W_chi2(correlated: bool) -> np.ndarray:
        """Weight matrix for post-hoc chi2 evaluation."""
        if correlated:
            return np.linalg.inv(ordinate.cov)
        return np.diag(1.0 / sigma**2)

    # --- Central value fit ---
    if central_value_fit:
        W_cv   = _W_chi2(central_value_fit_correlated)
        cv_res = _run_one_thc(
            ordinate.mean, sigma, Omega_use, W_outer_use, abscissa,
            n, delta_t, truncation_dimension, symmetric_correlator,
            epsilon_real, epsilon_imag, Nstates,
        )
        chi2_cv = _compute_chi2(
            ordinate.mean, abscissa,
            cv_res["energies"], cv_res["overlaps"], W_cv,
        )
        fit_result.import_from_thc(
            energies      = cv_res["energies"],
            overlaps      = cv_res["overlaps"],
            lambda_eigvals= cv_res["lambda_eigvals"],
            chi2_val      = chi2_cv,
            max_k         = max_k,
            W_chi2        = W_cv,
            cov           = ordinate.cov,
            abscissa_fit  = abscissa,
            truncation_dimension = truncation_dimension,
            nres          = None,
        )

    if not resample_fit:
        return fit_result

    # --- Resample fits ---
    W_rs = _W_chi2(resample_fit_correlated)

    if Nproc is None:
        # Serial
        for nres in range(Nres):
            rs_res  = _run_one_thc(
                ordinate.rspl[nres], sigma, Omega_use, W_outer_use, abscissa,
                n, delta_t, truncation_dimension, symmetric_correlator,
                epsilon_real, epsilon_imag, Nstates,
            )
            chi2_rs = _compute_chi2(
                ordinate.rspl[nres], abscissa,
                rs_res["energies"], rs_res["overlaps"], W_rs,
            )
            fit_result.import_from_thc(
                energies      = rs_res["energies"],
                overlaps      = rs_res["overlaps"],
                lambda_eigvals= rs_res["lambda_eigvals"],
                chi2_val      = chi2_rs,
                max_k         = max_k,
                W_chi2        = W_rs,
                cov           = ordinate.cov,
                abscissa_fit  = abscissa,
                truncation_dimension = truncation_dimension,
                nres          = nres,
            )
    else:
        # Parallel
        block_size = Nres // Nproc
        n_rest     = Nres % Nproc
        slices     = [list(range(b * block_size, (b + 1) * block_size)) for b in range(Nproc)]
        if n_rest:
            slices.append(list(range(Nproc * block_size, Nres)))

        inputs = [
            (sl, ordinate.rspl, sigma, Omega_use, W_outer_use, abscissa,
             n, delta_t, truncation_dimension, symmetric_correlator,
             epsilon_real, epsilon_imag, Nstates)
            for sl in slices
        ]

        with mp.Pool(processes=Nproc) as pool:
            all_blocks = pool.starmap(_thc_block, inputs)

        for block in all_blocks:
            for nres, rs_res in block:
                chi2_rs = _compute_chi2(
                    ordinate.rspl[nres], abscissa,
                    rs_res["energies"], rs_res["overlaps"], W_rs,
                )
                fit_result.import_from_thc(
                    energies      = rs_res["energies"],
                    overlaps      = rs_res["overlaps"],
                    lambda_eigvals= rs_res["lambda_eigvals"],
                    chi2_val      = chi2_rs,
                    max_k         = max_k,
                    W_chi2        = W_rs,
                    cov           = ordinate.cov,
                    abscissa_fit  = abscissa,
                    truncation_dimension = truncation_dimension,
                    nres          = nres,
                )

    return fit_result
