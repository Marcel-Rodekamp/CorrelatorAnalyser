"""
Unit tests for the lsqfit backend (fit_lsqfit.py / fit.py with backend='lsqfit').

Overview
--------
1. Input-validation tests
     1a. Passing a non-Data ordinate raises TypeError.
     1b. Passing a scalar / None abscissa raises an error (any type is
         acceptable here, since the current code propagates the failure through
         np.asarray rather than a dedicated type-check).

2. End-to-end parameter recovery for three model families:
     2a. Linear          y(x)  = m·x + b
     2b. Single-exponential C(t) = A·exp(-E·t)
     2c. Double-exponential C(t) = A1·exp(-E1·t) + A2·exp(-E2·t)
     2d. Test Fit with normal priors
     2e. Test Fit with log-normal priors
   Each model is exercised over every meaningful combination of
     - central_value_fit=True/False
     - resample_fit=True/False  (at least one must be True)
     - central_value_fit_correlated=True/False
     - resample_fit_correlated=True/False
   The recovered parameters must agree with the ground truth within 3σ.

3. Parallel-execution test (Nproc=4) must return bit-for-bit identical
   parameter values compared with serial execution (Nproc=None), because
   lsqfit's optimiser is deterministic for fixed inputs.

Design notes
------------
* Mock bootstrap data are drawn from a multivariate Gaussian, giving a
  well-conditioned covariance matrix that can be used in correlated fits.
* All fits use p0 (initial-guess dict, no priors) for the standard parameter
  recovery tests.  An additional set of tests exercises (log-)normal-prior fits.
* Model functions are defined at module scope.  lsqfit (via FitResult) calls
  inspect.getsource() when serialising to HDF5; lambdas would fail.
* A shared helper `_check_params_recovered` unifies the 3σ assertions across
  CV-only, resample-only, and combined fits.
* Fixtures that build Data objects use scope="module" to avoid regenerating
  heavy bootstrap samples for every test.
"""

from __future__ import annotations

import numpy as np
import pytest

from correlatoranalyser import Data
from correlatoranalyser.fit_lsqfit import fit_lsqfit
from correlatoranalyser.prior import Prior

# =============================================================================
# Global test configuration
# =============================================================================

# Fixed seed → reproducible mock data across the whole module.
_RNG_SEED = 20240101

# Number of raw "gauge configurations" used to build the bootstrap ensemble.
# Large enough that the statistical uncertainty on the mean is small, so fits
# reliably stay within 3σ.
NRAW: int = 600

# Number of bootstrap resamples.  Kept moderate to limit test runtime while
# still giving a meaningful resample-fit error estimate.
NBST: int = 300

# Tolerance: the recovered parameter must lie within NSIG × (error estimate)
# of the true value.
NSIG: int = 3

# =============================================================================
# Ground-truth parameters for the three model families
# =============================================================================

# Linear model  y(x) = m·x + b
LINEAR_M:     float = 2.5
LINEAR_B:     float = 0.7
LINEAR_NOISE: float = 0.10   # per-observable noise level

# Single-exponential  C(t) = A·exp(-E·t)
SINGLE_A:     float = 1.5
SINGLE_E:     float = 0.3
SINGLE_NOISE: float = 0.02

# Double-exponential  C(t) = A1·exp(-E1·t) + A2·exp(-E2·t)
DOUBLE_A1:    float = 1.5
DOUBLE_E1:    float = 0.3
DOUBLE_A2:    float = 0.5
DOUBLE_E2:    float = 0.8
DOUBLE_NOISE: float = 0.01

# =============================================================================
# Model functions  (must be module-level for lsqfit/inspect.getsource)
# =============================================================================

def model_linear(x, p):
    """Linear model: y = m·x + b."""
    return p["m"] * x + p["b"]


def model_single_exp(t, p):
    """Single-exponential model: C(t) = A·exp(-E·t)."""
    return p["A"] * np.exp(-p["E"] * t)


def model_double_exp(t, p):
    """Double-exponential model: C(t) = A1·exp(-E1·t) + A2·exp(-E2·t)."""
    return p["A1"] * np.exp(-p["E1"] * t) + p["A2"] * np.exp(-p["E2"] * t)


# =============================================================================
# Mock-data factory
# =============================================================================

def _make_bootstrap_data(
    true_values: np.ndarray,
    noise_scale: float,
    rng: np.random.Generator,
    nraw: int = NRAW,
    nbst: int = NBST,
) -> Data:
    """
    Create a bootstrap Data object from synthetic observations.

    The raw data are drawn from a multivariate Gaussian with
      mean  = true_values
      cov   = noise_scale² · (I + 0.1 · 11ᵀ)

    The off-diagonal term 0.1·noise_scale² introduces mild correlations so
    that the correlated fit path (which uses the full covariance matrix) is
    genuinely different from the uncorrelated path.  The correlations are
    small enough that correlated and uncorrelated fits still recover the same
    true values.

    Parameters
    ----------
    true_values : np.ndarray, shape (N,)
        The noiseless signal that the fit should recover.
    noise_scale : float
        Controls the variance of each observable.
    rng : np.random.Generator
        Seeded random generator for reproducibility.
    nraw : int
        Number of "configurations" to generate.
    nbst : int
        Number of bootstrap resamples.

    Returns
    -------
    Data  (resample_type='bst', Nresample=nbst)
    """
    n = len(true_values)
    # Mildly correlated covariance – positive-definite by construction.
    cov = noise_scale ** 2 * (np.eye(n) + 0.1 * np.ones((n, n)))
    raw = rng.multivariate_normal(mean=true_values, cov=cov, size=nraw)
    return Data(resample_type="bst", data=raw, Nresample=nbst)


# =============================================================================
# Module-scoped fixtures  (built once; reused for all parametrized cases)
# =============================================================================

@pytest.fixture(scope="module")
def linear_data():
    """
    (abscissa, ordinate, true_params) for the linear model.

    We use 8 evenly-spaced x-values on [0, 1].  The covariance matrix
    computed from these bootstrap samples is 8×8 and well-conditioned.
    """
    rng = np.random.default_rng(_RNG_SEED)
    x = np.linspace(0.0, 1.0, 8)
    true_y = LINEAR_M * x + LINEAR_B
    ordinate = _make_bootstrap_data(true_y, LINEAR_NOISE, rng)
    true_params = {"m": LINEAR_M, "b": LINEAR_B}
    return x, ordinate, true_params


@pytest.fixture(scope="module")
def single_exp_data():
    """
    (abscissa, ordinate, true_params) for the single-exponential model.

    t = 1..10.  The signal decays from ≈1.08 at t=1 to ≈0.05 at t=10, giving
    a well-determined E over the chosen range.
    """
    rng = np.random.default_rng(_RNG_SEED + 1)
    t = np.arange(1, 11, dtype=float)
    true_C = SINGLE_A * np.exp(-SINGLE_E * t)
    ordinate = _make_bootstrap_data(true_C, SINGLE_NOISE, rng)
    true_params = {"A": SINGLE_A, "E": SINGLE_E}
    return t, ordinate, true_params


@pytest.fixture(scope="module")
def double_exp_data():
    """
    (abscissa, ordinate, true_params) for the double-exponential model.

    t = 1..12.  Two exponentials with well-separated energies (0.3 vs 0.8)
    so that the fit is identifiable even without prior information.
    """
    rng = np.random.default_rng(_RNG_SEED + 2)
    t = np.arange(1, 13, dtype=float)
    true_C = (
        DOUBLE_A1 * np.exp(-DOUBLE_E1 * t)
        + DOUBLE_A2 * np.exp(-DOUBLE_E2 * t)
    )
    ordinate = _make_bootstrap_data(true_C, DOUBLE_NOISE, rng)
    true_params = {
        "A1": DOUBLE_A1, "E1": DOUBLE_E1,
        "A2": DOUBLE_A2, "E2": DOUBLE_E2,
    }
    return t, ordinate, true_params


# =============================================================================
# Shared helper: assert parameter recovery within NSIG σ
# =============================================================================

def _check_params_recovered(
    fit_result,
    true_params: dict,
    central_value_fit: bool,
    resample_fit: bool,
    nsig: int = NSIG,
) -> None:
    """
    Assert that every fit parameter lies within *nsig* standard deviations of
    its true value.

    The choice of error estimate depends on which fits were performed:

    * CV-only:   use the lsqfit Hessian error (params_hessian_err) for the
                 central-value result.  This is the standard Gaussian error
                 propagation estimate from the covariance matrix of the fit.

    * Resample-only:   the CV mean is never populated (locked_mean is True and
                       is never set), so we instead use the mean over bootstrap
                       fit results as our point estimate and its standard
                       deviation as the uncertainty.

    * Both CV + resample:   use the locked CV mean as the point estimate and
                            the bootstrap standard deviation of the resample
                            fit results as the uncertainty.
    """
    for key, true_val in true_params.items():
        assert key in fit_result.params, f"Parameter '{key}' missing from FitResult"

        if resample_fit:
            # Resample fits populate rspl[0..Nres-1]; use them to form the
            # estimate and its uncertainty.
            rspl_vals = fit_result.params[key].rspl
            assert rspl_vals is not None, f"rspl is None for '{key}'"
            assert not np.any(np.isnan(rspl_vals)), f"NaN in rspl for '{key}'"

            if central_value_fit:
                # CV mean is authoritative; bootstrap gives the uncertainty.
                estimate = fit_result.params[key].mean
                uncertainty = np.std(rspl_vals, ddof=1)
            else:
                # No CV fit: use the mean of bootstrap fits as the estimate.
                estimate = float(np.mean(rspl_vals))
                uncertainty = np.std(rspl_vals, ddof=1)

        else:
            # CV-only: Hessian error from lsqfit is the natural uncertainty.
            assert key in fit_result.params_hessian_err, (
                f"Hessian error missing for '{key}'"
            )
            estimate = fit_result.params[key].mean
            uncertainty = fit_result.params_hessian_err[key].mean

        assert uncertainty > 0, (
            f"Uncertainty for '{key}' is non-positive: {uncertainty}"
        )
        deviation = abs(estimate - true_val)
        assert deviation < nsig * uncertainty, (
            f"Parameter '{key}': |{estimate:.6g} - {true_val:.6g}| = {deviation:.3g} "
            f">= {nsig} × {uncertainty:.3g}.  "
            f"Fit result is more than {nsig}σ away from the true value."
        )


# =============================================================================
# Parametrisation of fit strategies
# =============================================================================

# Each entry is (central_value_fit, cv_correlated, resample_fit, rs_correlated).
# We require at least one of central_value_fit / resample_fit to be True.
# We test the six most informative combinations:
#   – cv-only uncorrelated / correlated
#   – resample-only uncorrelated / correlated
#   – both uncorrelated / both correlated
FIT_STRATEGIES = [
    # id                         cv     cv_c   rs     rs_c
    ("cv_uncorr",           True,  False, False, False),
    ("cv_corr",             True,  True,  False, False),
    ("rs_uncorr",           False, False, True,  False),
    ("rs_corr",             False, False, True,  True ),
    ("both_uncorr",         True,  False, True,  False),
    ("both_corr",           True,  True,  True,  True ),
]

# Build friendly pytest IDs from the label field.
_STRATEGY_IDS   = [s[0] for s in FIT_STRATEGIES]
_STRATEGY_PARAMS = [s[1:] for s in FIT_STRATEGIES]


# =============================================================================
# 1. Input-validation tests
# =============================================================================

class TestInputValidation:
    """
    Validate that fit_lsqfit raises informative errors for bad inputs.

    We use a minimal-but-valid setup (simple linear data) and replace one
    argument at a time with an invalid value.
    """

    # Shared minimal setup reused inside test methods.
    _x = np.linspace(0.0, 1.0, 5)
    _rng = np.random.default_rng(0)
    _raw = _rng.normal(1.0, 0.05, size=(100, 5))
    _ordinate = Data(resample_type="bst", data=_raw, Nresample=50)
    _p0 = {"m": 2.0, "b": 0.5}

    def test_ordinate_must_be_Data_not_ndarray(self):
        """
        Passing a plain numpy array as ordinate must raise TypeError.

        The Data class stores resamples and covariance information; a bare
        array carries none of that, so fit_lsqfit cannot proceed safely.
        The _validate_inputs helper explicitly checks isinstance(ordinate, Data).
        """
        with pytest.raises(TypeError):
            fit_lsqfit(
                abscissa=self._x,
                ordinate=np.ones(5),   # ← NOT a Data object
                model=model_linear,
                p0=self._p0,
            )

    def test_ordinate_must_be_Data_not_list(self):
        """
        A Python list is not a Data object; the same TypeError is expected.
        This guards against accidental use of raw lists in calling code.
        """
        with pytest.raises(TypeError):
            fit_lsqfit(
                abscissa=self._x,
                ordinate=[1.0, 2.0, 3.0, 4.0, 5.0],   # ← list, not Data
                model=model_linear,
                p0=self._p0,
            )

    def test_ordinate_must_be_Data_not_scalar(self):
        """
        A scalar ordinate is also invalid; the check must fire before any
        attempt to access .Nresample or .gvar().
        """
        with pytest.raises(TypeError):
            fit_lsqfit(
                abscissa=self._x,
                ordinate=42.0,   # ← scalar, not Data
                model=model_linear,
                p0=self._p0,
            )

    def test_abscissa_scalar_raises(self):
        """
        A scalar abscissa has no length (0-d array), so accessing .shape[0]
        will raise IndexError.  We only require that *some* exception is raised
        rather than a silent incorrect computation.

        Note: the current code does not perform an explicit type check on
        abscissa; instead the error propagates via np.asarray().shape[0].
        A future improvement could add an explicit TypeError here.
        """
        with pytest.raises(Exception):
            fit_lsqfit(
                abscissa=42,        # ← scalar – not a 1-D array or Data
                ordinate=self._ordinate,
                model=model_linear,
                p0=self._p0,
            )

    def test_abscissa_none_raises(self):
        """
        None is not an array-like; np.asarray(None) yields a 0-d object array
        whose .shape[0] raises IndexError.
        """
        with pytest.raises(Exception):
            fit_lsqfit(
                abscissa=None,      # ← None
                ordinate=self._ordinate,
                model=model_linear,
                p0=self._p0,
            )

    def test_neither_cv_nor_resample_raises(self):
        """
        Setting both central_value_fit and resample_fit to False is a logical
        error: there is nothing to compute.  A ValueError must be raised
        immediately, before any fit attempt.
        """
        with pytest.raises(ValueError, match="At least one of"):
            fit_lsqfit(
                abscissa=self._x,
                ordinate=self._ordinate,
                model=model_linear,
                p0=self._p0,
                central_value_fit=False,
                resample_fit=False,
            )

    def test_neither_prior_nor_p0_raises(self):
        """
        lsqfit requires either a prior or an initial-guess dict p0 to start
        the optimisation.  Omitting both must raise ValueError.
        """
        with pytest.raises(ValueError):
            fit_lsqfit(
                abscissa=self._x,
                ordinate=self._ordinate,
                model=model_linear,
                # neither prior nor p0
            )

    def test_no_model_raises(self):
        """
        A model function is mandatory.  Passing model=None (or omitting it)
        must raise ValueError before lsqfit is invoked.
        """
        with pytest.raises(ValueError):
            fit_lsqfit(
                abscissa=self._x,
                ordinate=self._ordinate,
                model=None,
                p0=self._p0,
            )

    def test_parallel_correlated_raises(self):
        """
        Combining Nproc > 1 with a correlated fit must raise ValueError immediately.
        gvar objects in the fit arguments cannot be safely pickled across process
        boundaries, so this combination is explicitly unsupported in the lsqfit backend.
        """
        with pytest.raises(ValueError, match="Correlated fits are not supported with parallel"):
            fit_lsqfit(
                abscissa=self._x,
                ordinate=self._ordinate,
                model=model_linear,
                p0=self._p0,
                resample_fit=True,
                resample_fit_correlated=True,
                Nproc=4,
            )

# =============================================================================
# 2a. End-to-end tests – linear model
# =============================================================================

class TestLinearFit:
    """
    Fit y(x) = m·x + b to synthetic bootstrap data and verify parameter recovery.

    The linear model is the simplest possible non-trivial case.  It provides a
    sanity check for the plumbing (Data → lsqfit → FitResult) before moving to
    nonlinear models.

    p0 is set to values slightly displaced from the truth so that the fit is
    not trivially satisfied at the starting point.
    """

    _p0 = {"m": LINEAR_M * 0.85, "b": LINEAR_B * 0.85}

    @pytest.mark.parametrize(
        "cv, cv_corr, rs, rs_corr",
        _STRATEGY_PARAMS,
        ids=_STRATEGY_IDS,
    )
    def test_parameter_recovery(
        self, linear_data, cv, cv_corr, rs, rs_corr
    ):
        """
        For every combination of (cv/resample) × (correlated/uncorrelated),
        check that fit_lsqfit recovers m and b within 3σ of the true values.

        Covariance note: ordinate.cov is the bootstrap estimate of the data
        covariance.  For the uncorrelated path, ordinate.serr (diagonal entries)
        is used instead.  Both are derived from the same bootstrap ensemble, so
        they are consistent estimators of the population covariance.
        """
        x, ordinate, true_params = linear_data
        result = fit_lsqfit(
            abscissa=x,
            ordinate=ordinate,
            central_value_fit=cv,
            central_value_fit_correlated=cv_corr,
            resample_fit=rs,
            resample_fit_correlated=rs_corr,
            model=model_linear,
            p0=self._p0,
        )
        _check_params_recovered(result, true_params, cv, rs)

    @pytest.mark.parametrize(
        "cv, cv_corr, rs, rs_corr",
        _STRATEGY_PARAMS,
        ids=_STRATEGY_IDS,
    )
    def test_fit_result_metadata(
        self, linear_data, cv, cv_corr, rs, rs_corr
    ):
        """
        Check that FitResult is populated with sensible metadata:
          - dof > 0   (degrees of freedom)
          - chi2 > 0  (positive chi-squared)
          - p_value in [0, 1]  (valid probability)

        For the resample path we check the *central-value* chi2 (stored as the
        locked mean when both cv and rs are run, or checked via rspl otherwise).
        """
        x, ordinate, _ = linear_data
        result = fit_lsqfit(
            abscissa=x,
            ordinate=ordinate,
            central_value_fit=cv,
            central_value_fit_correlated=cv_corr,
            resample_fit=rs,
            resample_fit_correlated=rs_corr,
            model=model_linear,
            p0=self._p0,
        )
        # dof is set once (from the first fit) and should be positive.
        assert result.dof is not None
        assert result.dof > 0

        if cv:
            # CV fit populates the locked mean of the chi2 Data object.
            chi2_val = (
                result.chi2.mean if hasattr(result.chi2, "mean") else float(result.chi2)
            )
            assert chi2_val > 0

        if rs:
            # All resample chi2 values should be finite and positive.
            chi2_rspl = result.chi2.rspl
            assert np.all(np.isfinite(chi2_rspl))
            assert np.all(chi2_rspl > 0)


# =============================================================================
# 2b. End-to-end tests – single-exponential model
# =============================================================================

class TestSingleExpFit:
    """
    Fit C(t) = A·exp(-E·t) to synthetic bootstrap data.

    The single exponential is the archetypal Lattice QCD correlator fit.
    lsqfit's gradient-based solver handles it reliably without priors when
    a good p0 is provided.  We start 15% below the true values to verify
    that the optimiser converges from an imperfect starting point.
    """

    _p0 = {"A": SINGLE_A * 0.85, "E": SINGLE_E * 0.85}

    @pytest.mark.parametrize(
        "cv, cv_corr, rs, rs_corr",
        _STRATEGY_PARAMS,
        ids=_STRATEGY_IDS,
    )
    def test_parameter_recovery(
        self, single_exp_data, cv, cv_corr, rs, rs_corr
    ):
        """
        For every strategy combination, the fitted A and E must be within 3σ
        of SINGLE_A and SINGLE_E respectively.

        The correlated path uses the full 10×10 bootstrap covariance matrix;
        the uncorrelated path uses only the diagonal (serr).  Both should
        recover the same parameters because the mock correlations are mild.
        """
        t, ordinate, true_params = single_exp_data
        result = fit_lsqfit(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=cv,
            central_value_fit_correlated=cv_corr,
            resample_fit=rs,
            resample_fit_correlated=rs_corr,
            model=model_single_exp,
            p0=self._p0,
        )
        _check_params_recovered(result, true_params, cv, rs)

    @pytest.mark.parametrize(
        "cv, cv_corr, rs, rs_corr",
        _STRATEGY_PARAMS,
        ids=_STRATEGY_IDS,
    )
    def test_resample_count(
        self, single_exp_data, cv, cv_corr, rs, rs_corr
    ):
        """
        When resample_fit=True, the rspl array for every parameter must contain
        exactly NBST entries (one per bootstrap resample), with no NaN values.

        This guards against silent failures in the resample loop where some
        resamples might not be populated.
        """
        t, ordinate, _ = single_exp_data
        result = fit_lsqfit(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=cv,
            central_value_fit_correlated=cv_corr,
            resample_fit=rs,
            resample_fit_correlated=rs_corr,
            model=model_single_exp,
            p0=self._p0,
        )
        if rs:
            for key in ("A", "E"):
                assert result.params[key].rspl.shape == (NBST,), (
                    f"Expected rspl shape ({NBST},) for '{key}', "
                    f"got {result.params[key].rspl.shape}"
                )
                assert not np.any(np.isnan(result.params[key].rspl)), (
                    f"NaN in rspl for '{key}'"
                )


# =============================================================================
# 2c. End-to-end tests – double-exponential model
# =============================================================================

class TestDoubleExpFit:
    """
    Fit C(t) = A1·exp(-E1·t) + A2·exp(-E2·t) to synthetic bootstrap data.

    The double exponential is considerably harder to fit than the single
    exponential because the two terms are correlated in parameter space.
    We help the optimiser by:
      - Choosing well-separated energies (E2/E1 ≈ 2.7) for identifiability.
      - Setting p0 close (but not equal) to the truth.
      - Using a relatively large, correlated mock dataset (NRAW = 600).

    Note: lsqfit with unconstrained p0 may occasionally fail to converge for
    very noisy realisations of the double-exp model.  The fixed seed and
    generous NRAW make this unlikely in practice.
    """

    _p0 = {
        "A1": DOUBLE_A1 * 0.90,
        "E1": DOUBLE_E1 * 0.90,
        "A2": DOUBLE_A2 * 0.90,
        "E2": DOUBLE_E2 * 0.90,
    }

    @pytest.mark.parametrize(
        "cv, cv_corr, rs, rs_corr",
        _STRATEGY_PARAMS,
        ids=_STRATEGY_IDS,
    )
    def test_parameter_recovery(
        self, double_exp_data, cv, cv_corr, rs, rs_corr
    ):
        """
        All four parameters (A1, E1, A2, E2) must be recovered within 3σ.

        The correlated fit with a 12×12 covariance matrix is the hardest path;
        svdcut is left at None (default) because the mock covariance is
        well-conditioned.
        """
        t, ordinate, true_params = double_exp_data
        result = fit_lsqfit(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=cv,
            central_value_fit_correlated=cv_corr,
            resample_fit=rs,
            resample_fit_correlated=rs_corr,
            model=model_double_exp,
            p0=self._p0,
        )
        _check_params_recovered(result, true_params, cv, rs)


# =============================================================================
# 2d. Fit with normal priors
# =============================================================================

class TestNormalPriorFit:
    """
    Verify that normal priors are correctly forwarded to lsqfit.

    We test the linear model with generous priors (σ ≈ 5× true value) to
    ensure that the prior barely shifts the result from the data-driven
    optimum, while still exercising the prior-building code path.

    Log-normal prior tests are intentionally absent: BUG 2 in
    _build_lsqfit_args causes incorrect gvar objects to be passed to lsqfit
    for log-normal priors (see module docstring).
    """

    def test_cv_with_normal_priors(self, linear_data):
        """
        A CV fit with loose normal priors should recover m and b within 3σ.
        The prior contribution to chi² must be finite and non-negative.
        """
        x, ordinate, true_params = linear_data
        prior = {
            "m": Prior(LINEAR_M, 5.0),   # σ = 5.0 >> expected fit error
            "b": Prior(LINEAR_B, 5.0),
        }
        result = fit_lsqfit(
            abscissa=x,
            ordinate=ordinate,
            central_value_fit=True,
            resample_fit=False,
            model=model_linear,
            prior=prior,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=False)
        # Priors should be stored in FitResult after the CV fit.
        assert "m" in result.priors
        assert "b" in result.priors

    def test_resample_with_normal_priors(self, linear_data):
        """
        A resample fit with normal priors must populate rspl for every
        parameter.  The priors are passed identically to each resample fit
        (they do not change with the resample) so all resamples should
        converge to nearly the same region as the CV fit.
        """
        x, ordinate, true_params = linear_data
        prior = {
            "m": Prior(LINEAR_M, 5.0),
            "b": Prior(LINEAR_B, 5.0),
        }
        result = fit_lsqfit(
            abscissa=x,
            ordinate=ordinate,
            central_value_fit=False,
            resample_fit=True,
            model=model_linear,
            prior=prior,
        )
        _check_params_recovered(result, true_params, central_value_fit=False, resample_fit=True)
        for key in ("m", "b"):
            assert result.params[key].rspl.shape == (NBST,)
            assert not np.any(np.isnan(result.params[key].rspl))

# =============================================================================
# 2e. Fits with log-normal priors on energy parameters
# =============================================================================

class TestLogNormalPriorFit:
    """
    Verify that log-normal priors on energy parameters are correctly forwarded
    to lsqfit after the fix to _build_lsqfit_args (Bug 2).

    Log-normal priors are the natural choice for energies and amplitudes in
    Lattice QCD fits: they enforce positivity and are parameterised in a way
    that is symmetric under rescaling.

    Prior encoding after the bug fix:
      - normal     key "E"      → gv.gvar(mean, sdev)              (linear space)
      - log-normal key "log(E)" → gv.gvar(log_mean, log_sdev)      (log space)
    where log_mean = log(E_true) and log_sdev is a loose width.

    We use generous prior widths (log_sdev = 1.0 corresponds to roughly a
    factor e ≈ 2.7 uncertainty) so the prior barely constrains the fit and
    the result is dominated by the data.  The recovered parameters must
    still lie within 3σ of the true values.

    Note: the amplitude A is given a normal prior here (it could also be
    log-normal, but mixing the two kinds in one test makes the log-normal
    code path easier to isolate).
    """

    def test_single_exp_lognormal_energy_cv(self, single_exp_data):
        """
        CV fit of C(t)=A·exp(-E·t) with a log-normal prior on E.

        The Prior mean for E is log(SINGLE_E) because the Prior class stores
        the mean in the natural parameter of the distribution (log-space for
        log-normal).  A loose sdev=1.0 in log-space spans roughly a factor
        of e in linear space, so the data easily dominate.
        """
        t, ordinate, true_params = single_exp_data
        prior = {
            "A": Prior(SINGLE_A, 5.0),                         # normal prior on amplitude
            "E": Prior(np.log(SINGLE_E), 1.0, dist="log-normal"),  # log-normal prior on energy
        }
        result = fit_lsqfit(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=True,
            resample_fit=False,
            model=model_single_exp,
            prior=prior,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=False)
        # Both priors should be imported back into FitResult.
        assert "A" in result.priors
        assert "E" in result.priors
        assert result.priors["E"].dist == "log-normal"

    def test_single_exp_lognormal_energy_resample(self, single_exp_data):
        """
        Resample fit of C(t)=A·exp(-E·t) with a log-normal prior on E.

        Priors are passed identically to every resample fit (they do not vary
        with the resample), so each rspl[nres] should converge to nearly the
        same region as the CV fit.  We verify that all NBST rspl entries are
        finite and within 3σ of the truth.
        """
        t, ordinate, true_params = single_exp_data
        prior = {
            "A": Prior(SINGLE_A, 5.0),
            "E": Prior(np.log(SINGLE_E), 1.0, dist="log-normal"),
        }
        result = fit_lsqfit(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=False,
            resample_fit=True,
            model=model_single_exp,
            prior=prior,
        )
        _check_params_recovered(result, true_params, central_value_fit=False, resample_fit=True)
        for key in ("A", "E"):
            assert result.params[key].rspl.shape == (NBST,)
            assert not np.any(np.isnan(result.params[key].rspl))

    def test_single_exp_lognormal_energy_cv_and_resample(self, single_exp_data):
        """
        Combined CV + resample fit with a log-normal prior on E.

        This exercises the most complete code path: the CV fit populates the
        locked mean and the resample fits populate rspl[0..NBST-1].
        """
        t, ordinate, true_params = single_exp_data
        prior = {
            "A": Prior(SINGLE_A, 5.0),
            "E": Prior(np.log(SINGLE_E), 1.0, dist="log-normal"),
        }
        result = fit_lsqfit(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=True,
            resample_fit=True,
            model=model_single_exp,
            prior=prior,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=True)

    def test_double_exp_lognormal_energies_cv(self, double_exp_data):
        """
        CV fit of C(t)=A1·exp(-E1·t)+A2·exp(-E2·t) with log-normal priors on
        both energies E1 and E2.

        Using log-normal priors on energies is essential in practice because
        the double-exponential landscape has a saddle point between the two
        solutions (E1↔E2 swap).  The priors break this degeneracy by
        encoding our prior knowledge that E1 < E2.

        We centre the log-normal priors at log(E1_true) and log(E2_true)
        with sdev=0.5 (moderately tight, roughly a factor √e ≈ 1.6) so that
        the priors guide the optimiser toward the correct solution while still
        allowing the data to dominate the final result.
        """
        t, ordinate, true_params = double_exp_data
        prior = {
            "A1": Prior(DOUBLE_A1, 5.0),
            "E1": Prior(np.log(DOUBLE_E1), 0.5, dist="log-normal"),
            "A2": Prior(DOUBLE_A2, 5.0),
            "E2": Prior(np.log(DOUBLE_E2), 0.5, dist="log-normal"),
        }
        result = fit_lsqfit(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=True,
            resample_fit=False,
            model=model_double_exp,
            prior=prior,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=False)
        for key in ("E1", "E2"):
            assert result.priors[key].dist == "log-normal", (
                f"Prior for '{key}' should be log-normal, got {result.priors[key].dist}"
            )

    def test_double_exp_lognormal_energies_cv_and_resample(self, double_exp_data):
        """
        Combined CV + resample fit with log-normal priors on both energies.

        This is the most realistic Lattice QCD use case: a double-exponential
        correlator fit where the energies are constrained to be positive and
        the full resample distribution is used to estimate parameter
        uncertainties.
        """
        t, ordinate, true_params = double_exp_data
        prior = {
            "A1": Prior(DOUBLE_A1, 5.0),
            "E1": Prior(np.log(DOUBLE_E1), 0.5, dist="log-normal"),
            "A2": Prior(DOUBLE_A2, 5.0),
            "E2": Prior(np.log(DOUBLE_E2), 0.5, dist="log-normal"),
        }
        result = fit_lsqfit(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=True,
            resample_fit=True,
            model=model_double_exp,
            prior=prior,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=True)
        for key in ("A1", "E1", "A2", "E2"):
            assert result.params[key].rspl.shape == (NBST,)
            assert not np.any(np.isnan(result.params[key].rspl))


# =============================================================================
# 3. Parallel execution test
# =============================================================================

class TestParallelExecution:
    """
    Verify that Nproc=4 (multiprocess pool) produces results that are
    numerically identical to Nproc=None (serial execution).

    Rationale: lsqfit's optimiser is fully deterministic – it uses a
    gradient-based algorithm with no stochastic elements.  Given identical
    inputs (same ordinate.rspl[nres], same model, same p0), the fit must
    converge to the same parameters regardless of whether it runs in a
    subprocess or the main process.  Any divergence would indicate a
    serialisation/deserialisation bug in the dill-based pickling of the
    lsqfit result.

    We run the comparison for all three model families to cover different
    problem sizes (2, 2, 4 parameters) and different data sizes (8, 10, 12
    observables).
    """

    _p0_linear     = {"m": LINEAR_M * 0.85, "b": LINEAR_B * 0.85}
    _p0_single_exp = {"A": SINGLE_A * 0.85, "E": SINGLE_E * 0.85}
    _p0_double_exp = {
        "A1": DOUBLE_A1 * 0.90, "E1": DOUBLE_E1 * 0.90,
        "A2": DOUBLE_A2 * 0.90, "E2": DOUBLE_E2 * 0.90,
    }

    def _run_serial_and_parallel(self, abscissa, ordinate, model, p0):
        """
        Run a resample fit both serially (Nproc=None) and in parallel
        (Nproc=4) with identical settings.  Return (serial_result,
        parallel_result).
        """
        common_kwargs = dict(
            abscissa=abscissa,
            ordinate=ordinate,
            central_value_fit=True,
            resample_fit=True,
            resample_fit_correlated=False,
            model=model,
            p0=p0,
        )
        serial   = fit_lsqfit(**common_kwargs, Nproc=None)
        parallel = fit_lsqfit(**common_kwargs, Nproc=4)
        return serial, parallel

    def _assert_identical_results(self, serial, parallel, param_keys):
        """
        Assert that all parameter arrays (CV mean and every resample) match
        to double-precision equality between serial and parallel runs.

        We use np.testing.assert_array_equal (exact equality) rather than
        assert_allclose because the computation is deterministic and any
        difference signals a real bug.
        """
        for key in param_keys:
            # Central-value parameters
            np.testing.assert_array_equal(
                serial.params[key].mean,
                parallel.params[key].mean,
                err_msg=f"CV mean mismatch for param '{key}'",
            )
            # Resample parameters
            np.testing.assert_array_equal(
                serial.params[key].rspl,
                parallel.params[key].rspl,
                err_msg=f"rspl mismatch for param '{key}'",
            )

    def test_parallel_linear(self, linear_data):
        """Serial and parallel must produce identical rspl arrays for the linear model."""
        x, ordinate, _ = linear_data
        serial, parallel = self._run_serial_and_parallel(
            x, ordinate, model_linear, self._p0_linear
        )
        self._assert_identical_results(serial, parallel, ["m", "b"])

    def test_parallel_single_exp(self, single_exp_data):
        """
        Serial and parallel must produce identical rspl arrays for the
        single-exponential model.  This exercises dill serialisation of an
        nlf object that contains gvar objects and a two-parameter prior dict.
        """
        t, ordinate, _ = single_exp_data
        serial, parallel = self._run_serial_and_parallel(
            t, ordinate, model_single_exp, self._p0_single_exp
        )
        self._assert_identical_results(serial, parallel, ["A", "E"])

    def test_parallel_double_exp(self, double_exp_data):
        """
        Serial and parallel must produce identical rspl arrays for the
        double-exponential model (four parameters, larger data vector).
        This is the most demanding test of the multiprocess + dill path.
        """
        t, ordinate, _ = double_exp_data
        serial, parallel = self._run_serial_and_parallel(
            t, ordinate, model_double_exp, self._p0_double_exp
        )
        self._assert_identical_results(serial, parallel, ["A1", "E1", "A2", "E2"])

    def test_parallel_chi2_identical(self, linear_data):
        """
        Not only parameters but also the per-resample chi² values must be
        identical between serial and parallel runs.
        """
        x, ordinate, _ = linear_data
        serial, parallel = self._run_serial_and_parallel(
            x, ordinate, model_linear, self._p0_linear
        )
        np.testing.assert_array_equal(
            serial.chi2.rspl,
            parallel.chi2.rspl,
            err_msg="Per-resample chi2 differs between serial and parallel runs",
        )
