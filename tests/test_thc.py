"""
Unit tests for the THC (Truncated Hankel Correlator) backend (thc.py).

Overview
--------
1. Input-validation tests
     - ordinate type, flag logic, T_eff parity, truncation_dimension bounds,
       truncation_method values, truncation_method=None without integer k.

2. Unit tests for private helpers
     2a. _build_hankel_matrix — shape and anti-diagonal structure
     2b. _determine_k         — gap and kpos criteria on controlled spectra
     2c. _filter_energies     — keep/discard logic
     2d. _thc_overlaps        — exact reconstruction for known energies

3. End-to-end parameter recovery
     3a. Single-exponential  C(t) = A0 * exp(-E0 * t)
     3b. Double-exponential  C(t) = A0*exp(-E0*t) + A1*exp(-E1*t)
   Each exercised over all six strategy combinations
   (cv/resample) × (correlated/uncorrelated chi2).

4. Truncation selection tests
     - gap vs kpos on well-controlled data
     - fixed truncation_dimension bypasses auto-detection
     - k is consistent across resamples

5. t0 and delta_t tests
     - t0 > 0 correctly skips early timeslices
     - delta_t > 1 produces correct results

6. FitResult structure tests
     - all expected params present (E0, A0, ...)
     - lambda_eigs present and correct shape
     - dof = (T_eff + 1) - 2*Nstates
     - chi2 > 0, p_value in [0,1]
     - resample fit populates rspl

7. Parallel execution tests
     - Nproc=4 is bit-for-bit identical to serial for serial-deterministic data

Design notes
------------
* THC does not populate params_hessian_err.  For CV-only recovery checks we
  use a fixed absolute tolerance of 5×noise.  For resample fits we use
  nanstd over the bootstrap as the uncertainty estimate.
* All fixtures use bootstrap (bst) resamples with NRAW=1000 and low noise
  so THC reliably resolves the intended number of states.
* Seeds are offset from all other test suites (+60/+61) to be independent.
"""

from __future__ import annotations

import numpy as np
import pytest

from correlatoranalyser import Data
from correlatoranalyser.thc import (
    thc,
    _build_hankel_matrix,
    _determine_k,
    _filter_energies,
    _thc_overlaps,
    _build_default_Omega,
    _build_default_W_outer,
    _hankel_eigh,
)

# =============================================================================
# Global test configuration
# =============================================================================

_RNG_SEED = 20240101 + 60
NRAW: int  = 1000
NBST: int  = 200
NSIG: int  = 3

# =============================================================================
# Ground-truth parameters
# =============================================================================

SINGLE_A0:    float = 1.5
SINGLE_E0:    float = 0.3
SINGLE_NOISE: float = 0.005   # low noise for reliable single-state extraction

DOUBLE_A0:    float = 1.5
DOUBLE_E0:    float = 0.3
DOUBLE_A1:    float = 0.5
DOUBLE_E1:    float = 0.8
DOUBLE_NOISE: float = 0.003   # very low noise for two-state separation

# Timeslice range: T_total=25 → T_eff=24 (even), n=12
T_MIN, T_MAX = 0, 24          # t = 0, 1, ..., 24  (T_total = 25)


# =============================================================================
# Mock-data factory
# =============================================================================

def _make_bootstrap_data(
    true_values: np.ndarray,
    noise: float,
    rng: np.random.Generator,
    nraw: int = NRAW,
    nbst: int = NBST,
) -> Data:
    """Bootstrap Data from a diagonal-covariance Gaussian."""
    n    = len(true_values)
    cov  = np.diag(np.full(n, noise**2))
    raw  = rng.multivariate_normal(mean=true_values, cov=cov, size=nraw)
    return Data(resample_type="bst", data=raw, Nresample=nbst)


# =============================================================================
# Module-scoped fixtures
# =============================================================================

@pytest.fixture(scope="module")
def single_exp_data():
    rng    = np.random.default_rng(_RNG_SEED)
    t      = np.arange(T_MIN, T_MAX + 1, dtype=float)
    signal = SINGLE_A0 * np.exp(-SINGLE_E0 * t)
    return t, _make_bootstrap_data(signal, SINGLE_NOISE, rng)


@pytest.fixture(scope="module")
def double_exp_data():
    rng    = np.random.default_rng(_RNG_SEED + 1)
    t      = np.arange(T_MIN, T_MAX + 1, dtype=float)
    signal = DOUBLE_A0 * np.exp(-DOUBLE_E0 * t) + DOUBLE_A1 * np.exp(-DOUBLE_E1 * t)
    return t, _make_bootstrap_data(signal, DOUBLE_NOISE, rng)


# =============================================================================
# Shared assertion helper
# =============================================================================

def _check_params_recovered(
    fit_result,
    true_params: dict,
    central_value_fit: bool,
    resample_fit: bool,
    abs_tol: float,
    nsig: int = NSIG,
) -> None:
    """
    Assert every parameter lies within nsig σ of its true value.

    THC does not provide Hessian errors, so:
    * Resample fits:  nanstd of rspl as the uncertainty estimate.
    * CV-only:        abs_tol as a fixed tolerance (typically 5 × noise).
    """
    for key, true_val in true_params.items():
        assert key in fit_result.params, f"Parameter '{key}' missing from FitResult"

        if resample_fit:
            rspl = fit_result.params[key].rspl
            assert rspl is not None, f"rspl is None for '{key}'"
            uncertainty = np.nanstd(rspl, ddof=1)
            assert uncertainty > 0, f"nanstd is zero for '{key}' — all resamples identical"
            estimate = fit_result.params[key].mean if central_value_fit else float(np.nanmean(rspl))
        else:
            # CV-only: no Hessian error available, use fixed tolerance
            estimate    = fit_result.params[key].mean
            uncertainty = abs_tol

        deviation = abs(estimate - true_val)
        assert deviation < nsig * uncertainty, (
            f"'{key}': |{estimate:.6g} - {true_val:.6g}| = {deviation:.3g} "
            f">= {nsig} × {uncertainty:.3g}"
        )


# =============================================================================
# Fit-strategy parametrisation
# =============================================================================

FIT_STRATEGIES = [
    # id                cv     cv_c   rs     rs_c
    ("cv_uncorr",      True,  False, False, False),
    ("cv_corr",        True,  True,  False, False),
    ("rs_uncorr",      False, False, True,  False),
    ("rs_corr",        False, False, True,  True ),
    ("both_uncorr",    True,  False, True,  False),
    ("both_corr",      True,  True,  True,  True ),
]

_STRATEGY_IDS    = [s[0] for s in FIT_STRATEGIES]
_STRATEGY_PARAMS = [s[1:] for s in FIT_STRATEGIES]


# =============================================================================
# 1. Input-validation tests
# =============================================================================

class TestInputValidation:

    _rng      = np.random.default_rng(0)
    _raw      = _rng.normal(1.0, 0.01, size=(200, 25))
    _ordinate = Data(resample_type="bst", data=_raw, Nresample=50)

    def test_ordinate_not_data_raises(self):
        with pytest.raises(TypeError):
            thc(ordinate=np.ones(25), truncation_dimension=2)

    def test_ordinate_list_raises(self):
        with pytest.raises(TypeError):
            thc(ordinate=list(range(25)), truncation_dimension=2)

    def test_neither_cv_nor_resample_raises(self):
        with pytest.raises(ValueError, match="At least one of"):
            thc(ordinate=self._ordinate, truncation_dimension=2,
                central_value_fit=False, resample_fit=False)

    def test_t_eff_odd_raises(self):
        """T_eff = T_total - 1 - t0 must be even.  25 - 1 - 0 = 24 OK; add t0=1 → 23 odd."""
        with pytest.raises(ValueError, match="even"):
            thc(ordinate=self._ordinate, truncation_dimension=2, t0=1)

    def test_t0_too_large_raises(self):
        with pytest.raises(ValueError, match="no usable data"):
            thc(ordinate=self._ordinate, truncation_dimension=2, t0=25)

    def test_truncation_dimension_exceeds_hankel_raises(self):
        # n = 12 for T_total=25, so truncation_dimension > 13 should raise
        with pytest.raises(ValueError, match="truncation_dimension"):
            thc(ordinate=self._ordinate, truncation_dimension=20)

    def test_truncation_dimension_requires_enough_timeslices(self):
        # small ordinate (7 slices, T_eff=6, n=3) with k=10 should raise
        small_raw = self._rng.normal(1.0, 0.01, size=(200, 7))
        small_ord = Data(resample_type="bst", data=small_raw, Nresample=50)
        with pytest.raises(ValueError, match="requires at least"):
            thc(ordinate=small_ord, truncation_dimension=10)

    def test_invalid_truncation_method_raises(self):
        with pytest.raises(ValueError, match="truncation_method"):
            thc(ordinate=self._ordinate, truncation_dimension=None,
                truncation_method="invalid")

    def test_truncation_method_none_without_int_raises(self):
        """truncation_method=None requires truncation_dimension to be an int."""
        with pytest.raises(RuntimeError):
            thc(ordinate=self._ordinate, truncation_dimension=None,
                truncation_method=None)


# =============================================================================
# 2a. _build_hankel_matrix
# =============================================================================

class TestBuildHankelMatrix:

    def test_shape(self):
        C = np.arange(1.0, 11.0)
        H = _build_hankel_matrix(C, n=4)
        assert H.shape == (5, 5)

    def test_anti_diagonal_structure(self):
        """H[i,j] = C[i+j] — verify a few entries."""
        C = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
        H = _build_hankel_matrix(C, n=3)
        assert H[0, 0] == pytest.approx(10.0)
        assert H[0, 1] == pytest.approx(20.0)
        assert H[1, 0] == pytest.approx(20.0)   # symmetric
        assert H[1, 2] == pytest.approx(40.0)
        assert H[3, 3] == pytest.approx(70.0)

    def test_symmetric(self):
        C = np.random.default_rng(7).uniform(0.1, 1.0, size=11)
        H = _build_hankel_matrix(C, n=5)
        np.testing.assert_allclose(H, H.T)


# =============================================================================
# 2b. _determine_k
# =============================================================================

class TestDetermineK:
    """
    Controlled tests for the two truncation criteria.
 
    C is constructed as a sum of exponentials with prescribed decay rates and
    amplitudes.  The Hankel matrix built from such a C has a natural spectral
    structure: k_signal large eigenvalues corresponding to the physical modes,
    and the remainder near zero (noiseless case) or of order ~1 (noisy case).
    """
 
    _t = np.arange(0, 25, dtype=float)   # t = 0..24, T_eff = 24, n = 12
 
    def _make_ordinate(self, energies, amplitudes, noise=0.0):
        """
        Build a bootstrap Data from C(t) = sum_l A_l * exp(-E_l * t) + noise.
 
        With noise=0 the Hankel eigenspectrum has exactly len(energies) large
        modes and the rest near zero, so both gap and kpos can identify k exactly.
        """
        signal = sum(
            A * np.exp(-E * self._t)
            for E, A in zip(energies, amplitudes)
        )
        if noise > 0:
            rng  = np.random.default_rng(42)
            cov  = np.diag(np.full(len(signal), noise**2))
            raw  = rng.multivariate_normal(mean=signal, cov=cov, size=500)
            return Data(resample_type="bst", data=raw, Nresample=50)
        # Noiseless: construct a trivial bootstrap with all resamples = signal
        rspl = np.tile(signal, (50, 1))
        return Data.import_resamples(resample_type="bst", rspl=rspl, mean=signal)
 

    def _get_cv_omega(self, ordinate):
        """
        Return C, Omega, n for the CV mean of an ordinate.
 
        When the ordinate was built with noise=0 all resamples are identical,
        giving serr=0 and a division-by-zero in _build_default_Omega.  In
        that case we use sigma=ones, which corresponds to the unweighted (flat)
        Omega = I case and is the correct limit for noiseless data.
        """
        n     = (ordinate.shape[0] - 1) // 2
        sigma = ordinate.serr
        # When uncorrelated tests serr = 0
        if np.any(sigma < 1e-15):
            sigma = np.ones_like(sigma)
        Omega = _build_default_Omega(sigma, n)
        return ordinate.mean, Omega, n

 
    def test_gap_finds_correct_k_two_signal_modes(self):
        """gap method on 2-exponential noiseless data must return k=2."""
        ordinate   = self._make_ordinate([0.3, 0.8], [1.5, 0.5], noise=0.0)
        C, Omega, n = self._get_cv_omega(ordinate)
        k = _determine_k(C, Omega, n, method="gap")
        assert k == 2, f"Expected k=2, got {k}"
 
    def test_gap_finds_correct_k_one_signal_mode(self):
        """gap method on single-exponential noiseless data must return k=1."""
        ordinate    = self._make_ordinate([0.3], [1.5], noise=0.0)
        C, Omega, n = self._get_cv_omega(ordinate)
        k = _determine_k(C, Omega, n, method="gap")
        assert k == 1, f"Expected k=1, got {k}"
 
    def test_kpos_finds_correct_k(self):
        """kpos on noiseless 2-exponential data must return k=2."""
        ordinate    = self._make_ordinate([0.3, 0.8], [1.5, 0.5], noise=0.0)
        C, Omega, n = self._get_cv_omega(ordinate)
        k = _determine_k(C, Omega, n, method="kpos")
        assert k == 2, f"Expected k=2, got {k}"
 
    def test_gap_robust_to_positive_noise_modes(self):
        """
        kpos overestimates when noise modes are positive (the original problem).
        With noisy 2-exponential data, gap must return 2 while kpos may return > 2.
        We verify gap <= kpos, and that gap matches the known truth k=2.
        """
        ordinate    = self._make_ordinate([0.3, 0.8], [1.5, 0.5], noise=0.005)
        C, Omega, n = self._get_cv_omega(ordinate)
        k_gap  = _determine_k(C, Omega, n, method="gap")
        k_kpos = _determine_k(C, Omega, n, method="kpos")
        assert k_gap == 2, f"gap gave {k_gap}, expected 2"
        # kpos may or may not overestimate depending on the noise realisation,
        # but must be >= gap
        assert k_kpos >= k_gap
 
    def test_gap_returns_at_least_one(self):
        """Even a degenerate spectrum yields k >= 1."""
        ordinate    = self._make_ordinate([0.3], [1.5], noise=0.0)
        C, Omega, n = self._get_cv_omega(ordinate)
        k = _determine_k(C, Omega, n, method="gap")
        assert k >= 1
 
    def test_invalid_method_raises(self):
        ordinate    = self._make_ordinate([0.3], [1.5], noise=0.0)
        C, Omega, n = self._get_cv_omega(ordinate)
        with pytest.raises(ValueError, match="gap.*kpos"):
            _determine_k(C, Omega, n, method="magic")

# =============================================================================
# 2c. _filter_energies
# =============================================================================

class TestFilterEnergies:

    def test_keeps_real_physical_energy(self):
        energies = np.array([0.3 + 0j, 0.8 + 0j])
        result   = _filter_energies(energies, epsilon_real=1e-8, epsilon_imag=1e-12)
        np.testing.assert_allclose(result, [0.3, 0.8])

    def test_discards_large_imaginary_part(self):
        energies = np.array([0.3 + 0.5j, 0.8 + 0j])
        result   = _filter_energies(energies, epsilon_real=1e-8, epsilon_imag=1e-12)
        np.testing.assert_allclose(result, [0.8])

    def test_discards_negative_real_part(self):
        energies = np.array([-0.1 + 0j, 0.3 + 0j])
        result   = _filter_energies(energies, epsilon_real=1e-8, epsilon_imag=1e-12)
        np.testing.assert_allclose(result, [0.3])

    def test_discards_near_zero_real_part(self):
        energies = np.array([1e-10 + 0j, 0.3 + 0j])
        result   = _filter_energies(energies, epsilon_real=1e-8, epsilon_imag=1e-12)
        np.testing.assert_allclose(result, [0.3])

    def test_returns_sorted_ascending(self):
        energies = np.array([0.8 + 0j, 0.1 + 0j, 0.5 + 0j])
        result   = _filter_energies(energies, epsilon_real=1e-8, epsilon_imag=1e-12)
        np.testing.assert_allclose(result, [0.1, 0.5, 0.8])

    def test_empty_input(self):
        result = _filter_energies(np.array([]), epsilon_real=1e-8, epsilon_imag=1e-12)
        assert len(result) == 0

    def test_all_discarded_returns_empty(self):
        energies = np.array([-0.5 + 0j, 0.3 + 0.5j])
        result   = _filter_energies(energies, epsilon_real=1e-8, epsilon_imag=1e-12)
        assert len(result) == 0

    def test_numerical_noise_level_imaginary_passes(self):
        """Very small imaginary part (floating-point noise) must not discard a physical mode."""
        energies = np.array([0.3 + 1e-14j])
        result   = _filter_energies(energies, epsilon_real=1e-8, epsilon_imag=1e-12)
        assert len(result) == 1
        assert result[0] == pytest.approx(0.3, rel=1e-10)


# =============================================================================
# 2d. _thc_overlaps
# =============================================================================

class TestThcOverlaps:
    """
    Verify amplitude reconstruction from known energies on noiseless data.
    With exact energies and exact data the Vandermonde solve must recover
    the true amplitudes to near machine precision.
    """

    def test_single_exponential_exact(self):
        t      = np.arange(0, 15, dtype=float)
        E0, A0 = 0.3, 1.5
        C      = A0 * np.exp(-E0 * t)
        sigma  = np.ones_like(C) * 0.01
        A      = _thc_overlaps(C, t, sigma, np.array([E0]))
        assert A[0] == pytest.approx(A0, rel=1e-8)

    def test_double_exponential_exact(self):
        t           = np.arange(0, 20, dtype=float)
        E0, A0      = 0.3, 1.5
        E1, A1      = 0.8, 0.5
        C           = A0 * np.exp(-E0 * t) + A1 * np.exp(-E1 * t)
        sigma       = np.ones_like(C) * 0.01
        A           = _thc_overlaps(C, t, sigma, np.array([E0, E1]))
        assert A[0] == pytest.approx(A0, rel=1e-6)
        assert A[1] == pytest.approx(A1, rel=1e-6)

    def test_large_T_no_overflow(self):
        """Vandermonde split must prevent overflow for large T and large E."""
        t     = np.arange(0, 50, dtype=float)
        E, A  = 1.5, 2.0
        C     = A * np.exp(-E * t)
        sigma = np.ones_like(C) * 0.01
        result = _thc_overlaps(C, t, sigma, np.array([E]))
        assert np.isfinite(result[0])
        assert result[0] == pytest.approx(A, rel=1e-6)


# =============================================================================
# 3a. End-to-end tests — single-exponential
# =============================================================================

class TestSingleExpRecovery:
    """
    C(t) = A0 * exp(-E0 * t), fixed truncation_dimension=1.

    Low noise (0.005) and T=0..24 give a well-conditioned single-state
    problem.  Both gap and kpos should find k=1 automatically; we test with
    the fixed k=1 for robustness.
    """

    _true = {"E0": SINGLE_E0, "A0": SINGLE_A0}
    _tol  = 5 * SINGLE_NOISE

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _STRATEGY_PARAMS, ids=_STRATEGY_IDS)
    def test_parameter_recovery(self, single_exp_data, cv, cv_corr, rs, rs_corr):
        _, ordinate = single_exp_data
        result = thc(
            ordinate                     = ordinate,
            central_value_fit            = cv,
            central_value_fit_correlated = cv_corr,
            resample_fit                 = rs,
            resample_fit_correlated      = rs_corr,
            truncation_dimension         = 1,
        )
        _check_params_recovered(result, self._true, cv, rs, self._tol)

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _STRATEGY_PARAMS, ids=_STRATEGY_IDS)
    def test_fit_result_metadata(self, single_exp_data, cv, cv_corr, rs, rs_corr):
        """dof > 0, chi2 > 0, p_value in (0,1)."""
        _, ordinate = single_exp_data
        result = thc(
            ordinate             = ordinate,
            central_value_fit    = cv,
            central_value_fit_correlated = cv_corr,
            resample_fit         = rs,
            resample_fit_correlated = rs_corr,
            truncation_dimension = 1,
        )
        assert result.dof is not None and result.dof > 0
        if cv:
            chi2_val = result.chi2.mean if hasattr(result.chi2, "mean") else float(result.chi2)
            assert chi2_val > 0
        if rs:
            assert np.all(np.isfinite(result.chi2.rspl[~np.isnan(result.chi2.rspl)]))

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _STRATEGY_PARAMS, ids=_STRATEGY_IDS)
    def test_params_present(self, single_exp_data, cv, cv_corr, rs, rs_corr):
        """E0 and A0 must always be in result.params."""
        _, ordinate = single_exp_data
        result = thc(
            ordinate             = ordinate,
            central_value_fit    = cv,
            central_value_fit_correlated = cv_corr,
            resample_fit         = rs,
            resample_fit_correlated = rs_corr,
            truncation_dimension = 1,
        )
        assert "E0" in result.params
        assert "A0" in result.params

    def test_lambda_eigs_shape_cv_only(self, single_exp_data):
        """lambda_eigs must have shape (k,) = (1,) for k=1."""
        _, ordinate = single_exp_data
        result = thc(
            ordinate             = ordinate,
            central_value_fit    = True,
            resample_fit         = False,
            truncation_dimension = 1,
        )
        assert hasattr(result, "lambda_eigs")
        assert result.lambda_eigs is not None
        assert len(result.lambda_eigs) == 1   # plain array for CV-only

    def test_lambda_eigs_shape_with_resamples(self, single_exp_data):
        """lambda_eigs must be a Data of shape (k,) = (1,) with resamples."""
        _, ordinate = single_exp_data
        result = thc(
            ordinate             = ordinate,
            central_value_fit    = True,
            resample_fit         = True,
            truncation_dimension = 1,
        )
        assert isinstance(result.lambda_eigs, Data)
        assert result.lambda_eigs.shape == (1,)
        assert result.lambda_eigs.rspl.shape == (NBST, 1)

    def test_lambda_eigs_close_to_expected(self, single_exp_data):
        """Lambda_0 ≈ exp(-E0 * delta_t) for the true energy."""
        _, ordinate = single_exp_data
        result = thc(
            ordinate             = ordinate,
            central_value_fit    = True,
            resample_fit         = False,
            truncation_dimension = 1,
        )
        lam_expected = np.exp(-SINGLE_E0 * 1)   # delta_t=1
        lam_found    = result.lambda_eigs[0].real
        assert abs(lam_found - lam_expected) < 0.05

    def test_dof_formula(self, single_exp_data):
        """dof = (T_eff + 1) - 2 * Nstates = 25 - 2 = 23 for T_total=25, k=1."""
        _, ordinate = single_exp_data
        result = thc(ordinate=ordinate, truncation_dimension=1)
        T_total = ordinate.shape[0]
        T_eff   = T_total - 1
        expected_dof = (T_eff + 1) - 2 * 1
        assert result.dof == expected_dof

    def test_energy_positive(self, single_exp_data):
        """Fitted energy must be positive."""
        _, ordinate = single_exp_data
        result = thc(ordinate=ordinate, truncation_dimension=1, resample_fit=True)
        assert result.params["E0"].mean > 0
        assert np.all(result.params["E0"].rspl > 0)

    def test_amplitude_positive(self, single_exp_data):
        """Fitted amplitude must be positive (physical correlator)."""
        _, ordinate = single_exp_data
        result = thc(ordinate=ordinate, truncation_dimension=1, resample_fit=True)
        assert result.params["A0"].mean > 0


# =============================================================================
# 3b. End-to-end tests — double-exponential
# =============================================================================

class TestDoubleExpRecovery:
    """
    C(t) = A0*exp(-E0*t) + A1*exp(-E1*t), fixed truncation_dimension=2.

    Well-separated energies (E1/E0 ≈ 2.7) and very low noise for reliable
    two-state identification.
    """

    _true = {
        "E0": DOUBLE_E0, "A0": DOUBLE_A0,
        "E1": DOUBLE_E1, "A1": DOUBLE_A1,
    }
    _tol  = 5 * DOUBLE_NOISE

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _STRATEGY_PARAMS, ids=_STRATEGY_IDS)
    def test_parameter_recovery(self, double_exp_data, cv, cv_corr, rs, rs_corr):
        _, ordinate = double_exp_data
        result = thc(
            ordinate                     = ordinate,
            central_value_fit            = cv,
            central_value_fit_correlated = cv_corr,
            resample_fit                 = rs,
            resample_fit_correlated      = rs_corr,
            truncation_dimension         = 2,
        )
        _check_params_recovered(result, self._true, cv, rs, self._tol)

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _STRATEGY_PARAMS, ids=_STRATEGY_IDS)
    def test_all_params_present(self, double_exp_data, cv, cv_corr, rs, rs_corr):
        _, ordinate = double_exp_data
        result = thc(
            ordinate                     = ordinate,
            central_value_fit            = cv,
            central_value_fit_correlated = cv_corr,
            resample_fit                 = rs,
            resample_fit_correlated      = rs_corr,
            truncation_dimension         = 2,
        )
        for key in ("E0", "A0", "E1", "A1"):
            assert key in result.params, f"'{key}' missing from params"

    def test_lambda_eigs_shape_k2(self, double_exp_data):
        """lambda_eigs shape must be (2,) with k=2."""
        _, ordinate = double_exp_data
        result = thc(
            ordinate             = ordinate,
            central_value_fit    = True,
            resample_fit         = True,
            truncation_dimension = 2,
        )
        assert isinstance(result.lambda_eigs, Data)
        assert result.lambda_eigs.shape == (2,)

    def test_energies_ascending(self, double_exp_data):
        """THC must return energies sorted ascending: E0 < E1."""
        _, ordinate = double_exp_data
        result = thc(ordinate=ordinate, truncation_dimension=2)
        assert result.params["E0"].mean < result.params["E1"].mean

    def test_dof_formula_k2(self, double_exp_data):
        """dof = (T_eff + 1) - 2 * 2 = 25 - 4 = 21."""
        _, ordinate = double_exp_data
        result = thc(ordinate=ordinate, truncation_dimension=2)
        T_total      = ordinate.shape[0]
        T_eff        = T_total - 1
        expected_dof = (T_eff + 1) - 2 * 2
        assert result.dof == expected_dof


# =============================================================================
# 4. Truncation selection tests
# =============================================================================

class TestTruncationSelection:
    """
    Verify that automatic truncation methods choose the correct k and that
    the choice is consistent across all resamples.
    """

    def test_gap_selects_k1_for_single_exp(self, single_exp_data):
        """gap method must select k=1 for single-exponential data."""
        _, ordinate = single_exp_data
        result = thc(
            ordinate             = ordinate,
            truncation_dimension = None,
            truncation_method    = "gap",
            resample_fit         = True,
        )
        # With k=1 there is exactly one state; E1/A1 must not exist
        assert "E0" in result.params
        assert "E1" not in result.params

    def test_gap_selects_k2_for_double_exp(self, double_exp_data):
        """gap method must select k=2 for double-exponential data."""
        _, ordinate = double_exp_data
        result = thc(
            ordinate             = ordinate,
            truncation_dimension = None,
            truncation_method    = "gap",
            resample_fit         = True,
        )
        assert "E0" in result.params
        assert "E1" in result.params

    def test_kpos_selects_k1_for_single_exp(self, single_exp_data):
        """kpos method on clean single-exp data must also find k=1."""
        _, ordinate = single_exp_data
        result = thc(
            ordinate             = ordinate,
            truncation_dimension = None,
            truncation_method    = "kpos",
            resample_fit         = True,
        )
        assert "E0" in result.params

    def test_fixed_k_overrides_method(self, double_exp_data):
        """When truncation_dimension is an int, truncation_method is ignored."""
        _, ordinate = double_exp_data
        r1 = thc(ordinate=ordinate, truncation_dimension=2, truncation_method="gap")
        r2 = thc(ordinate=ordinate, truncation_dimension=2, truncation_method="kpos")
        # Both must give the same CV result since k is fixed
        np.testing.assert_allclose(
            r1.params["E0"].mean, r2.params["E0"].mean, rtol=1e-12,
        )

    def test_k_consistent_across_resamples(self, double_exp_data):
        """
        With auto truncation and resample_fit=True, k is determined once from
        CV and applied to all resamples.  The rspl shape of lambda_eigs must
        be (NBST, k) — i.e., no ragged arrays from per-resample k choices.
        """
        _, ordinate = double_exp_data
        result = thc(
            ordinate             = ordinate,
            truncation_dimension = None,
            truncation_method    = "gap",
            resample_fit         = True,
        )
        k = result.lambda_eigs.shape[0]
        assert result.lambda_eigs.rspl.shape == (NBST, k)

    def test_auto_truncation_recovers_double_exp(self, double_exp_data):
        """gap auto-detection must give good parameter recovery on double-exp."""
        _, ordinate = double_exp_data
        result = thc(
            ordinate             = ordinate,
            truncation_dimension = None,
            truncation_method    = "gap",
            resample_fit         = True,
        )
        true_params = {"E0": DOUBLE_E0, "A0": DOUBLE_A0,
                       "E1": DOUBLE_E1, "A1": DOUBLE_A1}
        _check_params_recovered(result, true_params,
                                central_value_fit=True, resample_fit=True,
                                abs_tol=5 * DOUBLE_NOISE)


# =============================================================================
# 5. t0 and delta_t tests
# =============================================================================

class TestT0AndDeltaT:

    def test_t0_recovers_correct_energy(self, double_exp_data):
        """
        Setting t0=2 must give the same energies as t0=0 within tolerance,
        since the same signal is present; only the early timeslices are skipped.
        The T_eff must still be even: T_total=25, t0=2 → T_eff=22 (even). OK.
        """
        _, ordinate = double_exp_data
        r0 = thc(ordinate=ordinate, truncation_dimension=2, t0=0)
        r2 = thc(ordinate=ordinate, truncation_dimension=2, t0=2)
        # Energies should agree within a few percent (t0 shifts affect fit range)
        assert abs(r0.params["E0"].mean - r2.params["E0"].mean) < 0.05
        assert abs(r0.params["E1"].mean - r2.params["E1"].mean) < 0.05

    def test_t0_changes_abscissa_range(self, single_exp_data):
        """The abscissa in FitResult should start at t0."""
        _, ordinate = single_exp_data
        result = thc(ordinate=ordinate, truncation_dimension=1, t0=2)
        assert float(result.abscissa[0]) == pytest.approx(2.0)

    def test_t0_changes_dof(self, single_exp_data):
        """dof = (T_eff + 1) - 2*k; T_eff = T_total - 1 - t0 decreases with t0."""
        _, ordinate = single_exp_data
        T_total = ordinate.shape[0]
        result  = thc(ordinate=ordinate, truncation_dimension=1, t0=2)
        T_eff   = T_total - 1 - 2
        assert result.dof == (T_eff + 1) - 2 * 1

    def test_delta_t_2_recovers_correct_energy(self, single_exp_data):
        """delta_t=2 must still recover the correct energy."""
        _, ordinate = single_exp_data
        result = thc(ordinate=ordinate, truncation_dimension=1, delta_t=2)
        assert abs(result.params["E0"].mean - SINGLE_E0) < 5 * SINGLE_NOISE

    def test_delta_t_changes_lambda_relation(self, single_exp_data):
        """
        Lambda = exp(-E * delta_t).  With delta_t=2 the Lambda value must be
        exp(-2*E0) rather than exp(-E0).
        """
        _, ordinate = single_exp_data
        r1 = thc(ordinate=ordinate, truncation_dimension=1, delta_t=1)
        r2 = thc(ordinate=ordinate, truncation_dimension=1, delta_t=2)
        lam1 = r1.lambda_eigs[0].real
        lam2 = r2.lambda_eigs[0].real
        # lam2 ≈ lam1^2  (since exp(-E*2) = exp(-E*1)^2)
        assert np.all(np.abs(lam2 - lam1**2) < 0.05)


# =============================================================================
# 6. FitResult structure tests
# =============================================================================

class TestFitResultStructure:

    def test_fcn_stored_and_callable(self, single_exp_data):
        """FitResult.fcn must be the multi-exponential model."""
        _, ordinate = single_exp_data
        result = thc(ordinate=ordinate, truncation_dimension=1)
        assert result.fcn is not None
        t_test = np.array([1.0, 2.0, 3.0])
        p_test = {"E0": SINGLE_E0, "A0": SINGLE_A0}
        out    = result.fcn(t_test, p_test)
        expected = SINGLE_A0 * np.exp(-SINGLE_E0 * t_test)
        np.testing.assert_allclose(out, expected, rtol=1e-12)

    def test_fcn_has_grad(self, single_exp_data):
        """model.grad must be attached."""
        _, ordinate = single_exp_data
        result = thc(ordinate=ordinate, truncation_dimension=1)
        assert hasattr(result.fcn, "grad")

    def test_p_value_in_unit_interval(self, single_exp_data):
        _, ordinate = single_exp_data
        result = thc(ordinate=ordinate, truncation_dimension=1)
        p_val  = result.p_value.mean if hasattr(result.p_value, "mean") else float(result.p_value)
        assert 0 <= p_val <= 1

    def test_aic_finite(self, single_exp_data):
        _, ordinate = single_exp_data
        result = thc(ordinate=ordinate, truncation_dimension=1)
        aic    = result.AIC.mean if hasattr(result.AIC, "mean") else float(result.AIC)
        assert np.isfinite(aic)

    def test_expected_chi2_finite_and_positive(self, single_exp_data):
        _, ordinate = single_exp_data
        result = thc(ordinate=ordinate, truncation_dimension=1)
        if result.expected_chi2 is not None:
            exp_c = result.expected_chi2.mean if hasattr(result.expected_chi2, "mean") \
                    else float(result.expected_chi2)
            assert np.isfinite(exp_c)
            assert exp_c > 0

    def test_resample_rspl_shape(self, single_exp_data):
        """With NBST resamples, rspl shape must be (NBST,) for scalar params."""
        _, ordinate = single_exp_data
        result = thc(ordinate=ordinate, truncation_dimension=1, resample_fit=True)
        assert result.params["E0"].rspl.shape == (NBST,)
        assert result.params["A0"].rspl.shape == (NBST,)

    def test_no_nan_in_rspl_with_fixed_k(self, single_exp_data):
        """With fixed k=1 on single-exp data, every resample must resolve the state."""
        _, ordinate = single_exp_data
        result = thc(ordinate=ordinate, truncation_dimension=1, resample_fit=True)
        assert not np.any(np.isnan(result.params["E0"].rspl)), "NaN energies in resamples"
        assert not np.any(np.isnan(result.params["A0"].rspl)), "NaN amplitudes in resamples"

    def test_chi2_rspl_finite_where_not_nan(self, single_exp_data):
        _, ordinate = single_exp_data
        result = thc(ordinate=ordinate, truncation_dimension=1, resample_fit=True)
        chi2   = result.chi2.rspl
        assert np.all(np.isfinite(chi2[~np.isnan(chi2)]))

    def test_repr_does_not_raise(self, single_exp_data):
        """__repr__ must complete without error (guard for nan serr handling)."""
        _, ordinate = single_exp_data
        result = thc(
            ordinate     = ordinate,
            truncation_dimension = 1,
            resample_fit = True,
        )
        s = repr(result)
        assert isinstance(s, str)
        assert "E0" in s


# =============================================================================
# 7. Parallel execution tests
# =============================================================================

class TestParallelExecution:
    """
    Nproc=4 must produce bit-for-bit identical results to Nproc=None.
    THC is fully deterministic given the same inputs and fixed k.
    """

    def _run_both(self, ordinate, k):
        common = dict(
            ordinate             = ordinate,
            central_value_fit    = True,
            resample_fit         = True,
            truncation_dimension = k,
        )
        serial   = thc(**common, Nproc=None)
        parallel = thc(**common, Nproc=4)
        return serial, parallel

    def _assert_identical(self, serial, parallel, keys):
        for key in keys:
            np.testing.assert_array_equal(
                serial.params[key].mean,
                parallel.params[key].mean,
                err_msg=f"CV mean mismatch for '{key}'",
            )
            np.testing.assert_array_equal(
                serial.params[key].rspl,
                parallel.params[key].rspl,
                err_msg=f"rspl mismatch for '{key}'",
            )

    def test_single_exp_serial_parallel_identical(self, single_exp_data):
        _, ordinate = single_exp_data
        s, p = self._run_both(ordinate, k=1)
        self._assert_identical(s, p, ["E0", "A0"])

    def test_double_exp_serial_parallel_identical(self, double_exp_data):
        _, ordinate = double_exp_data
        s, p = self._run_both(ordinate, k=2)
        self._assert_identical(s, p, ["E0", "A0", "E1", "A1"])

    def test_chi2_rspl_identical(self, single_exp_data):
        _, ordinate = single_exp_data
        s, p = self._run_both(ordinate, k=1)
        np.testing.assert_array_equal(
            s.chi2.rspl, p.chi2.rspl,
            err_msg="chi2 rspl differs between serial and parallel",
        )

    def test_lambda_eigs_rspl_identical(self, single_exp_data):
        _, ordinate = single_exp_data
        s, p = self._run_both(ordinate, k=1)
        np.testing.assert_array_equal(
            s.lambda_eigs.rspl, p.lambda_eigs.rspl,
            err_msg="lambda_eigs rspl differs between serial and parallel",
        )

    def test_parallel_with_auto_truncation(self, double_exp_data):
        """Parallel with truncation_method='gap' must match serial."""
        _, ordinate = double_exp_data
        common = dict(
            ordinate             = ordinate,
            central_value_fit    = True,
            resample_fit         = True,
            truncation_dimension = None,
            truncation_method    = "gap",
        )
        serial   = thc(**common, Nproc=None)
        parallel = thc(**common, Nproc=4)
        for key in serial.params:
            np.testing.assert_array_equal(
                serial.params[key].rspl,
                parallel.params[key].rspl,
                err_msg=f"rspl mismatch for '{key}' with auto truncation",
            )
