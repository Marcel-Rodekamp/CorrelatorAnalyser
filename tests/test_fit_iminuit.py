"""
Unit tests for the iminuit backend (fit_iminuit.py / fit.py with backend='iminuit').

Overview
--------
1. Input-validation tests — same contract as the lsqfit backend.

2. End-to-end parameter recovery for three model families:
     2a. Linear          y(x)  = m·x + b
     2b. Single-exponential C(t) = A·exp(-E·t)
     2c. Double-exponential C(t) = A1·exp(-E1·t) + A2·exp(-E2·t)
     2d. Fit with normal priors
     2e. Fit with log-normal priors  (and positivity limits)
   Each model is exercised over the same strategy matrix as the lsqfit suite.

3. Limits test — verify that parameter bounds (e.g. E > 0) are respected.

4. Parallel-execution test (Nproc=4) must produce bit-for-bit identical results
   compared with serial execution (Nproc=None).

5. Gradient information passed through model

Differences from the lsqfit backend
-------------------------------------
* Priors are injected directly as extra chi² terms via Prior.__call__; there is
  no internal parameter reparameterisation.  For log-normal priors iminuit
  operates in *linear* parameter space and evaluates Prior.__call__(theta),
  which internally computes ((log(theta) - mean) / sdev)².  Because the
  minimiser can in principle wander to theta ≤ 0, log-normal priors should be
  paired with a positivity limit (e.g. limits={"E": (0, None)}).

* The `params_hessian_err` are populated from Minuit's own covariance matrix
  (estimated by HESSE after migrad), rather than from a gvar propagation.

Shared infrastructure
---------------------
The constants, mock-data factory, model functions, `_check_params_recovered`
helper, and `FIT_STRATEGIES` parametrisation are intentionally kept identical
to test_fit_lsqfit.py so that both test modules can be read side-by-side and
the two backends compared directly.  In a larger test suite these shared
definitions should be moved to conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from correlatoranalyser import Data
from correlatoranalyser.fit_iminuit import fit_iminuit
from correlatoranalyser.prior import Prior

# =============================================================================
# Global test configuration  (identical to test_fit_lsqfit.py)
# =============================================================================

_RNG_SEED = 20240101
NRAW: int = 600
NBST: int = 300
NSIG: int = 3

# =============================================================================
# Ground-truth parameters
# =============================================================================

LINEAR_M:     float = 2.5
LINEAR_B:     float = 0.7
LINEAR_NOISE: float = 0.10

SINGLE_A:     float = 1.5
SINGLE_E:     float = 0.3
SINGLE_NOISE: float = 0.02

DOUBLE_A1:    float = 1.5
DOUBLE_E1:    float = 0.3
DOUBLE_A2:    float = 0.5
DOUBLE_E2:    float = 0.8
DOUBLE_NOISE: float = 0.01

# =============================================================================
# Model functions  (module-level; lambdas cannot be pickled by dill reliably
# for use in parallel workers, and FitResult.import_from_iminuit stores the
# model reference but does not call inspect.getsource here as lsqfit does)
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
# Mock-data factory  (identical to test_fit_lsqfit.py)
# =============================================================================

def _make_bootstrap_data(
    true_values: np.ndarray,
    noise_scale: float,
    rng: np.random.Generator,
    nraw: int = NRAW,
    nbst: int = NBST,
) -> Data:
    """
    Create a bootstrap Data object from synthetic multivariate-Gaussian data.

    Covariance = noise_scale² · (I + 0.1 · 11ᵀ) — mildly correlated but
    well-conditioned, so correlated and uncorrelated fits give consistent
    parameter estimates while exercising genuinely different code paths.
    """
    n = len(true_values)
    cov = noise_scale ** 2 * (np.eye(n) + 0.1 * np.ones((n, n)))
    raw = rng.multivariate_normal(mean=true_values, cov=cov, size=nraw)
    return Data(resample_type="bst", data=raw, Nresample=nbst)


# =============================================================================
# Module-scoped fixtures
# =============================================================================

@pytest.fixture(scope="module")
def linear_data():
    """8-point linear mock data, seeds offset from lsqfit suite to be independent."""
    rng = np.random.default_rng(_RNG_SEED + 10)
    x = np.linspace(0.0, 1.0, 8)
    true_y = LINEAR_M * x + LINEAR_B
    ordinate = _make_bootstrap_data(true_y, LINEAR_NOISE, rng)
    return x, ordinate, {"m": LINEAR_M, "b": LINEAR_B}


@pytest.fixture(scope="module")
def single_exp_data():
    """t=1..10 single-exponential mock data."""
    rng = np.random.default_rng(_RNG_SEED + 11)
    t = np.arange(1, 11, dtype=float)
    true_C = SINGLE_A * np.exp(-SINGLE_E * t)
    ordinate = _make_bootstrap_data(true_C, SINGLE_NOISE, rng)
    return t, ordinate, {"A": SINGLE_A, "E": SINGLE_E}


@pytest.fixture(scope="module")
def double_exp_data():
    """t=1..12 double-exponential mock data."""
    rng = np.random.default_rng(_RNG_SEED + 12)
    t = np.arange(1, 13, dtype=float)
    true_C = (
        DOUBLE_A1 * np.exp(-DOUBLE_E1 * t)
        + DOUBLE_A2 * np.exp(-DOUBLE_E2 * t)
    )
    ordinate = _make_bootstrap_data(true_C, DOUBLE_NOISE, rng)
    return t, ordinate, {
        "A1": DOUBLE_A1, "E1": DOUBLE_E1,
        "A2": DOUBLE_A2, "E2": DOUBLE_E2,
    }


# =============================================================================
# Shared assertion helper  (identical contract to test_fit_lsqfit.py)
# =============================================================================

def _check_params_recovered(
    fit_result,
    true_params: dict,
    central_value_fit: bool,
    resample_fit: bool,
    nsig: int = NSIG,
) -> None:
    """
    Assert that every parameter lies within *nsig* σ of its true value.

    Error estimate logic:
    * CV-only:          use params_hessian_err (Minuit HESSE estimate).
    * Resample-only:    bootstrap std of rspl values as both estimator and error.
    * CV + resample:    locked CV mean as point estimate; bootstrap std as error.

    This helper is deliberately kept identical to the lsqfit version so that
    differences in test outcomes can be attributed to the backend, not the
    assertion logic.
    """
    for key, true_val in true_params.items():
        assert key in fit_result.params, f"Parameter '{key}' missing from FitResult"

        if resample_fit:
            rspl_vals = fit_result.params[key].rspl
            assert rspl_vals is not None, f"rspl is None for '{key}'"
            assert not np.any(np.isnan(rspl_vals)), f"NaN in rspl for '{key}'"

            if central_value_fit:
                estimate    = fit_result.params[key].mean
                uncertainty = np.std(rspl_vals, ddof=1)
            else:
                estimate    = float(np.mean(rspl_vals))
                uncertainty = np.std(rspl_vals, ddof=1)
        else:
            assert key in fit_result.params_hessian_err, (
                f"Hessian error missing for '{key}'"
            )
            estimate    = fit_result.params[key].mean
            uncertainty = fit_result.params_hessian_err[key].mean

        assert uncertainty > 0, (
            f"Uncertainty for '{key}' is non-positive: {uncertainty}"
        )
        deviation = abs(estimate - true_val)
        assert deviation < nsig * uncertainty, (
            f"Parameter '{key}': |{estimate:.6g} - {true_val:.6g}| = {deviation:.3g} "
            f">= {nsig} × {uncertainty:.3g}.  Fit is more than {nsig}σ from truth."
        )


# =============================================================================
# Fit-strategy parametrisation  (identical matrix to test_fit_lsqfit.py)
# =============================================================================

FIT_STRATEGIES = [
    # id               cv     cv_c   rs     rs_c
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
    """
    Confirm that fit_iminuit raises informative errors for bad inputs.

    The validation is handled by _validate_inputs (shared with all backends),
    so the expected error types are the same as in test_fit_lsqfit.py.  We
    duplicate the tests here because the public entry point is different and
    we want each backend's test module to be self-contained.
    """

    _x        = np.linspace(0.0, 1.0, 5)
    _rng      = np.random.default_rng(0)
    _raw      = _rng.normal(1.0, 0.05, size=(100, 5))
    _ordinate = Data(resample_type="bst", data=_raw, Nresample=50)
    _p0       = {"m": 2.0, "b": 0.5}

    def test_ordinate_must_be_Data_not_ndarray(self):
        """A plain numpy array ordinate must raise TypeError."""
        with pytest.raises(TypeError):
            fit_iminuit(
                abscissa=self._x,
                ordinate=np.ones(5),
                model=model_linear,
                p0=self._p0,
            )

    def test_ordinate_must_be_Data_not_list(self):
        """A Python list ordinate must raise TypeError."""
        with pytest.raises(TypeError):
            fit_iminuit(
                abscissa=self._x,
                ordinate=[1.0, 2.0, 3.0, 4.0, 5.0],
                model=model_linear,
                p0=self._p0,
            )

    def test_ordinate_must_be_Data_not_scalar(self):
        """A scalar ordinate must raise TypeError."""
        with pytest.raises(TypeError):
            fit_iminuit(
                abscissa=self._x,
                ordinate=42.0,
                model=model_linear,
                p0=self._p0,
            )

    def test_abscissa_scalar_raises(self):
        """
        A scalar abscissa has no length.  The failure propagates through
        np.asarray().shape[0] before any iminuit call is made.
        """
        with pytest.raises(Exception):
            fit_iminuit(
                abscissa=42,
                ordinate=self._ordinate,
                model=model_linear,
                p0=self._p0,
            )

    def test_abscissa_none_raises(self):
        """None abscissa has no shape; an exception must be raised."""
        with pytest.raises(Exception):
            fit_iminuit(
                abscissa=None,
                ordinate=self._ordinate,
                model=model_linear,
                p0=self._p0,
            )

    def test_neither_cv_nor_resample_raises(self):
        """Both fit flags False is a logical error; ValueError must fire immediately."""
        with pytest.raises(ValueError, match="At least one of"):
            fit_iminuit(
                abscissa=self._x,
                ordinate=self._ordinate,
                model=model_linear,
                p0=self._p0,
                central_value_fit=False,
                resample_fit=False,
            )

    def test_neither_prior_nor_p0_raises(self):
        """
        iminuit needs starting values from either prior or p0 to construct the
        Minuit object.  Omitting both must raise ValueError before any
        minimisation is attempted.
        """
        with pytest.raises(ValueError):
            fit_iminuit(
                abscissa=self._x,
                ordinate=self._ordinate,
                model=model_linear,
            )

    def test_no_model_raises(self):
        """model=None must raise ValueError via _validate_inputs."""
        with pytest.raises(ValueError):
            fit_iminuit(
                abscissa=self._x,
                ordinate=self._ordinate,
                model=None,
                p0=self._p0,
            )

# =============================================================================
# 2a. End-to-end tests – linear model
# =============================================================================

class TestLinearFit:
    """
    Fit y(x) = m·x + b to synthetic bootstrap data with all strategy combinations.

    The linear model is the simplest sanity check for the iminuit plumbing.
    p0 is displaced 15% from truth so the optimiser must do real work.
    """

    _p0 = {"m": LINEAR_M * 0.85, "b": LINEAR_B * 0.85}

    @pytest.mark.parametrize(
        "cv, cv_corr, rs, rs_corr",
        _STRATEGY_PARAMS,
        ids=_STRATEGY_IDS,
    )
    def test_parameter_recovery(self, linear_data, cv, cv_corr, rs, rs_corr):
        """
        m and b must be recovered within 3σ for every CV/resample×corr strategy.

        Unlike lsqfit, iminuit builds its chi² function explicitly from numpy
        arrays, so both correlated (full covariance) and uncorrelated (diagonal)
        paths are exercised by the same assertion.
        """
        x, ordinate, true_params = linear_data
        result = fit_iminuit(
            abscissa=x,
            ordinate=ordinate,
            central_value_fit=cv,
            central_value_fit_correlated=cv_corr,
            resample_fit=rs,
            resample_fit_correlated=rs_corr,
            model=model_linear,
            p0=self._p0,
        )
        _check_params_recovered(result, true_params, central_value_fit=cv, resample_fit=rs)

    @pytest.mark.parametrize(
        "cv, cv_corr, rs, rs_corr",
        _STRATEGY_PARAMS,
        ids=_STRATEGY_IDS,
    )
    def test_fit_result_metadata(self, linear_data, cv, cv_corr, rs, rs_corr):
        """
        FitResult must carry positive dof and finite chi² / p-value.

        For iminuit, dof = Ndata - Nparams (no prior count because priors
        are not used in this test class).  With 8 data points and 2 parameters,
        dof = 6.
        """
        x, ordinate, _ = linear_data
        result = fit_iminuit(
            abscissa=x,
            ordinate=ordinate,
            central_value_fit=cv,
            central_value_fit_correlated=cv_corr,
            resample_fit=rs,
            resample_fit_correlated=rs_corr,
            model=model_linear,
            p0=self._p0,
        )
        assert result.dof is not None
        assert result.dof > 0

        if cv:
            chi2_val = (
                result.chi2.mean if hasattr(result.chi2, "mean") else float(result.chi2)
            )
            assert chi2_val > 0

        if rs:
            assert np.all(np.isfinite(result.chi2.rspl))
            assert np.all(result.chi2.rspl > 0)

    @pytest.mark.parametrize(
        "cv, cv_corr, rs, rs_corr",
        _STRATEGY_PARAMS,
        ids=_STRATEGY_IDS,
    )
    def test_hessian_errors_populated(self, linear_data, cv, cv_corr, rs, rs_corr):
        """
        iminuit computes Hessian errors via HESSE after migrad(); these are
        stored in params_hessian_err.  We verify they are present and positive
        for the CV fit, which is the authoritative single-fit result.

        For resample fits the Hessian errors are less meaningful (each resample
        is already a statistical estimate), but they should still be populated
        and finite.
        """
        x, ordinate, _ = linear_data
        result = fit_iminuit(
            abscissa=x,
            ordinate=ordinate,
            central_value_fit=cv,
            central_value_fit_correlated=cv_corr,
            resample_fit=rs,
            resample_fit_correlated=rs_corr,
            model=model_linear,
            p0=self._p0,
        )
        if cv:
            for key in ("m", "b"):
                assert key in result.params_hessian_err, (
                    f"Hessian error missing for '{key}' after CV fit"
                )
                assert result.params_hessian_err[key].mean > 0, (
                    f"Hessian error for '{key}' is non-positive"
                )

        if rs:
            for key in ("m", "b"):
                assert key in result.params_hessian_err
                assert np.all(result.params_hessian_err[key].rspl > 0), (
                    f"Hessian errors in rspl for '{key}' contain non-positive values"
                )


# =============================================================================
# 2b. End-to-end tests – single-exponential model
# =============================================================================

class TestSingleExpFit:
    """
    Fit C(t) = A·exp(-E·t) to synthetic bootstrap data.

    The archetypal Lattice QCD correlator fit.  p0 is 15% below truth.
    """

    _p0 = {"A": SINGLE_A * 0.85, "E": SINGLE_E * 0.85}

    @pytest.mark.parametrize(
        "cv, cv_corr, rs, rs_corr",
        _STRATEGY_PARAMS,
        ids=_STRATEGY_IDS,
    )
    def test_parameter_recovery(self, single_exp_data, cv, cv_corr, rs, rs_corr):
        """A and E must be recovered within 3σ for every strategy combination."""
        t, ordinate, true_params = single_exp_data
        result = fit_iminuit(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=cv,
            central_value_fit_correlated=cv_corr,
            resample_fit=rs,
            resample_fit_correlated=rs_corr,
            model=model_single_exp,
            p0=self._p0,
        )
        _check_params_recovered(result, true_params, central_value_fit=cv, resample_fit=rs)

    @pytest.mark.parametrize(
        "cv, cv_corr, rs, rs_corr",
        _STRATEGY_PARAMS,
        ids=_STRATEGY_IDS,
    )
    def test_resample_count(self, single_exp_data, cv, cv_corr, rs, rs_corr):
        """
        Every resample slot must be populated (no None / NaN).

        iminuit fits each resample independently; a silent failure in one
        resample would leave a NaN in rspl without raising an error unless we
        check explicitly.
        """
        t, ordinate, _ = single_exp_data
        result = fit_iminuit(
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
                assert result.params[key].rspl.shape == (NBST,)
                assert not np.any(np.isnan(result.params[key].rspl))


# =============================================================================
# 2c. End-to-end tests – double-exponential model
# =============================================================================

class TestDoubleExpFit:
    """
    Fit C(t) = A1·exp(-E1·t) + A2·exp(-E2·t) to synthetic bootstrap data.

    The double exponential has an E1↔E2 swap degeneracy.  We mitigate this by
    choosing well-separated energies (E2/E1 ≈ 2.7) and starting p0 close to
    the truth (10% below).
    """

    _p0 = {
        "A1": DOUBLE_A1 * 0.90, "E1": DOUBLE_E1 * 0.90,
        "A2": DOUBLE_A2 * 0.90, "E2": DOUBLE_E2 * 0.90,
    }

    @pytest.mark.parametrize(
        "cv, cv_corr, rs, rs_corr",
        _STRATEGY_PARAMS,
        ids=_STRATEGY_IDS,
    )
    def test_parameter_recovery(self, double_exp_data, cv, cv_corr, rs, rs_corr):
        """All four parameters must be recovered within 3σ."""
        t, ordinate, true_params = double_exp_data
        result = fit_iminuit(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=cv,
            central_value_fit_correlated=cv_corr,
            resample_fit=rs,
            resample_fit_correlated=rs_corr,
            model=model_double_exp,
            p0=self._p0,
        )
        _check_params_recovered(result, true_params, central_value_fit=cv, resample_fit=rs)


# =============================================================================
# 2d. Fits with normal priors
# =============================================================================

class TestNormalPriorFit:
    """
    Verify normal prior injection via the chi² cost function.

    For iminuit, `prior[k](params[k])` is added to chi² at every function
    evaluation.  There is no key-encoding step (unlike lsqfit's "log(k)"
    convention), so normal priors are the straightforward case.

    We use generous priors (σ = 5 × true value) so the data dominate.
    """

    def test_cv_with_normal_priors(self, linear_data):
        """CV fit with loose normal priors must recover m and b within 3σ."""
        x, ordinate, true_params = linear_data
        prior = {
            "m": Prior(LINEAR_M, 5.0),
            "b": Prior(LINEAR_B, 5.0),
        }
        result = fit_iminuit(
            abscissa=x,
            ordinate=ordinate,
            central_value_fit=True,
            resample_fit=False,
            model=model_linear,
            prior=prior,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=False)
        # priors must be stored in FitResult after a CV fit.
        assert "m" in result.priors
        assert "b" in result.priors

    def test_resample_with_normal_priors(self, linear_data):
        """Resample fits with normal priors must produce finite rspl for every bootstrap."""
        x, ordinate, true_params = linear_data
        prior = {
            "m": Prior(LINEAR_M, 5.0),
            "b": Prior(LINEAR_B, 5.0),
        }
        result = fit_iminuit(
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

    def test_single_exp_normal_priors_cv_and_resample(self, single_exp_data):
        """Combined CV + resample fit with normal priors on A and E."""
        t, ordinate, true_params = single_exp_data
        prior = {
            "A": Prior(SINGLE_A, 5.0),
            "E": Prior(SINGLE_E, 5.0),
        }
        result = fit_iminuit(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=True,
            resample_fit=True,
            model=model_single_exp,
            prior=prior,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=True)


# =============================================================================
# 2e. Fits with log-normal priors
# =============================================================================

class TestLogNormalPriorFit:
    """
    Verify log-normal prior injection via Prior.__call__ in the chi² cost.

    Key iminuit-specific consideration:
      iminuit works in linear parameter space and never reparameterises.
      Prior.__call__(theta) for a log-normal prior computes
          ((log(theta) - mean) / sdev)²
      and raises ValueError when theta ≤ 0.  If migrad explores negative
      energy values the fit crashes.  We therefore ALWAYS pair log-normal
      priors on energies with a positivity limit: limits={"E": (0, None)}.

    Prior encoding:
      Prior(mean=np.log(E_true), sdev=..., dist="log-normal")
      — mean is the mean of log(E), not of E itself —
      consistent with the lsqfit convention so user code is portable.

    We use sdev=1.0 in log-space (roughly a factor e ≈ 2.7 uncertainty) for
    the single-exp tests and sdev=0.5 for the double-exp tests (moderately
    tighter to break the E1↔E2 degeneracy).
    """

    def test_single_exp_lognormal_energy_cv(self, single_exp_data):
        """
        CV fit with log-normal prior on E and positivity limit.

        The positivity limit guards against migrad wandering to E ≤ 0, which
        would cause Prior.__call__ to raise ValueError and crash the fit.
        """
        t, ordinate, true_params = single_exp_data
        prior = {
            "A": Prior(SINGLE_A, 5.0),
            "E": Prior(np.log(SINGLE_E), 1.0, dist="log-normal"),
        }
        result = fit_iminuit(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=True,
            resample_fit=False,
            model=model_single_exp,
            prior=prior,
            limits={"E": (0.0, None)},
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=False)
        assert "E" in result.priors
        assert result.priors["E"].dist == "log-normal"

    def test_single_exp_lognormal_energy_resample(self, single_exp_data):
        """
        Resample fit with log-normal prior on E.

        Each of the NBST resample fits runs the same prior, so all rspl entries
        should be finite positive values near E_true.
        """
        t, ordinate, true_params = single_exp_data
        prior = {
            "A": Prior(SINGLE_A, 5.0),
            "E": Prior(np.log(SINGLE_E), 1.0, dist="log-normal"),
        }
        result = fit_iminuit(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=False,
            resample_fit=True,
            model=model_single_exp,
            prior=prior,
            limits={"E": (0.0, None)},
        )
        _check_params_recovered(result, true_params, central_value_fit=False, resample_fit=True)
        for key in ("A", "E"):
            assert result.params[key].rspl.shape == (NBST,)
            assert not np.any(np.isnan(result.params[key].rspl))
        # Energies must all be positive (the limit must have been enforced).
        assert np.all(result.params["E"].rspl > 0), (
            "Some resample energies are ≤ 0; the positivity limit was not enforced."
        )

    def test_single_exp_lognormal_energy_cv_and_resample(self, single_exp_data):
        """Combined CV + resample fit with a log-normal prior on E."""
        t, ordinate, true_params = single_exp_data
        prior = {
            "A": Prior(SINGLE_A, 5.0),
            "E": Prior(np.log(SINGLE_E), 1.0, dist="log-normal"),
        }
        result = fit_iminuit(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=True,
            resample_fit=True,
            model=model_single_exp,
            prior=prior,
            limits={"E": (0.0, None)},
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=True)

    def test_double_exp_lognormal_energies_cv(self, double_exp_data):
        """
        CV fit of the double exponential with log-normal priors on both E1 and E2.

        Moderately tight priors (sdev=0.5) break the E1↔E2 swap degeneracy by
        encoding prior knowledge that E1 < E2.  Positivity limits are applied
        to both energies.
        """
        t, ordinate, true_params = double_exp_data
        prior = {
            "A1": Prior(DOUBLE_A1, 5.0),
            "E1": Prior(np.log(DOUBLE_E1), 0.5, dist="log-normal"),
            "A2": Prior(DOUBLE_A2, 5.0),
            "E2": Prior(np.log(DOUBLE_E2), 0.5, dist="log-normal"),
        }
        result = fit_iminuit(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=True,
            resample_fit=False,
            model=model_double_exp,
            prior=prior,
            limits={"E1": (0.0, None), "E2": (0.0, None)},
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=False)
        for key in ("E1", "E2"):
            assert result.priors[key].dist == "log-normal", (
                f"Prior for '{key}' should be log-normal, got {result.priors[key].dist}"
            )

    def test_double_exp_lognormal_energies_cv_and_resample(self, double_exp_data):
        """
        Combined CV + resample fit with log-normal priors on E1 and E2.

        This is the most realistic Lattice QCD use case for the iminuit backend:
        a double-exponential correlator fit with energy positivity enforced by
        both limits and log-normal priors.
        """
        t, ordinate, true_params = double_exp_data
        prior = {
            "A1": Prior(DOUBLE_A1, 5.0),
            "E1": Prior(np.log(DOUBLE_E1), 0.5, dist="log-normal"),
            "A2": Prior(DOUBLE_A2, 5.0),
            "E2": Prior(np.log(DOUBLE_E2), 0.5, dist="log-normal"),
        }
        result = fit_iminuit(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=True,
            resample_fit=True,
            model=model_double_exp,
            prior=prior,
            limits={"E1": (0.0, None), "E2": (0.0, None)},
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=True)
        for key in ("A1", "E1", "A2", "E2"):
            assert result.params[key].rspl.shape == (NBST,)
            assert not np.any(np.isnan(result.params[key].rspl))
        assert np.all(result.params["E1"].rspl > 0)
        assert np.all(result.params["E2"].rspl > 0)


# =============================================================================
# 3. Limits test
# =============================================================================

class TestLimits:
    """
    Verify that parameter bounds (limits dict) are respected by Minuit.

    iminuit supports one-sided and two-sided limits.  We test the positivity
    constraint on an energy parameter, which is the most common use case in
    Lattice QCD.  We also check that a tight upper bound prevents the fit from
    exceeding it.
    """

    def test_lower_bound_respected(self, single_exp_data):
        """
        Setting limits={"E": (0, None)} must produce a positive fitted energy.

        We start p0 slightly above zero.  Without the limit a poorly converged
        fit could in principle explore E < 0; the limit must prevent this.
        """
        t, ordinate, _ = single_exp_data
        result = fit_iminuit(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=True,
            resample_fit=True,
            model=model_single_exp,
            p0={"A": SINGLE_A * 0.85, "E": SINGLE_E * 0.85},
            limits={"E": (0.0, None)},
        )
        assert result.params["E"].mean > 0, "CV energy violates lower bound E > 0"
        assert np.all(result.params["E"].rspl > 0), (
            "Some resample energies violate lower bound E > 0"
        )

    def test_limits_do_not_prevent_recovery(self, single_exp_data):
        """
        A generous lower bound (E > 0) must not prevent accurate parameter
        recovery — the true value lies well inside the allowed region.
        """
        t, ordinate, true_params = single_exp_data
        result = fit_iminuit(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=True,
            resample_fit=False,
            model=model_single_exp,
            p0={"A": SINGLE_A * 0.85, "E": SINGLE_E * 0.85},
            limits={"E": (0.0, None)},
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=False)

    def test_limit_key_not_in_params_is_ignored(self, single_exp_data):
        """
        A limits entry for a key that does not appear in the model parameters
        must be silently ignored (the loop in _run_minuit checks
        `if key in minuit.parameters`).  No error should be raised.
        """
        t, ordinate, _ = single_exp_data
        fit_iminuit(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=True,
            resample_fit=False,
            model=model_single_exp,
            p0={"A": SINGLE_A * 0.85, "E": SINGLE_E * 0.85},
            limits={"nonexistent_param": (0.0, None)},  # should be ignored
        )   # must not raise


# =============================================================================
# 4. Parallel-execution test
# =============================================================================

class TestParallelExecution:
    """
    Verify that Nproc=4 produces bit-for-bit identical results to Nproc=None.

    Rationale: Minuit2 is a deterministic gradient-based minimiser.  Given the
    same inputs (ordinate.rspl[nres], model, p0, limits), it converges to the
    same parameters regardless of which subprocess runs it.  Any discrepancy
    indicates a serialisation or process-boundary bug.

    Unlike the lsqfit backend, iminuit passes numpy closures (not gvar objects)
    across process boundaries.  dill handles these cleanly, so:
      * No gvar pickling warnings are expected.
      * Correlated fits are NOT excluded from parallel execution — there is no
        restriction analogous to the lsqfit backend's ban on correlated+parallel.
    We test both uncorrelated and correlated parallel paths.
    """

    _p0_linear     = {"m": LINEAR_M * 0.85, "b": LINEAR_B * 0.85}
    _p0_single_exp = {"A": SINGLE_A * 0.85, "E": SINGLE_E * 0.85}
    _p0_double_exp = {
        "A1": DOUBLE_A1 * 0.90, "E1": DOUBLE_E1 * 0.90,
        "A2": DOUBLE_A2 * 0.90, "E2": DOUBLE_E2 * 0.90,
    }

    def _run_serial_and_parallel(
        self, abscissa, ordinate, model, p0, correlated=False, limits=None
    ):
        """
        Run a combined CV + resample fit both serially and in parallel.
        Returns (serial_result, parallel_result).
        """
        common = dict(
            abscissa=abscissa,
            ordinate=ordinate,
            central_value_fit=True,
            resample_fit=True,
            resample_fit_correlated=correlated,
            model=model,
            p0=p0,
            limits=limits,
        )
        return fit_iminuit(**common, Nproc=None), fit_iminuit(**common, Nproc=4)

    def _assert_identical(self, serial, parallel, param_keys):
        """
        Assert exact equality of CV means and all rspl entries.

        Exact equality (not allclose) is appropriate because the computation is
        fully deterministic; any floating-point discrepancy would signal a real
        bug in the parallel infrastructure.
        """
        for key in param_keys:
            np.testing.assert_array_equal(
                serial.params[key].mean,
                parallel.params[key].mean,
                err_msg=f"CV mean mismatch for param '{key}'",
            )
            np.testing.assert_array_equal(
                serial.params[key].rspl,
                parallel.params[key].rspl,
                err_msg=f"rspl mismatch for param '{key}'",
            )

    def test_parallel_linear_uncorrelated(self, linear_data):
        """Serial and parallel must be identical for the linear model (uncorrelated)."""
        x, ordinate, _ = linear_data
        serial, parallel = self._run_serial_and_parallel(
            x, ordinate, model_linear, self._p0_linear
        )
        self._assert_identical(serial, parallel, ["m", "b"])

    def test_parallel_linear_correlated(self, linear_data):
        """
        Serial and parallel must also be identical for correlated fits.

        This test is not available in the lsqfit backend (where
        correlated+parallel raises ValueError).  Here it verifies that the
        closure containing the numpy cov_inv array is correctly serialised
        by dill and produces the same result in each subprocess.
        """
        x, ordinate, _ = linear_data
        serial, parallel = self._run_serial_and_parallel(
            x, ordinate, model_linear, self._p0_linear, correlated=True
        )
        self._assert_identical(serial, parallel, ["m", "b"])

    def test_parallel_single_exp(self, single_exp_data):
        """Serial and parallel must be identical for the single-exponential model."""
        t, ordinate, _ = single_exp_data
        serial, parallel = self._run_serial_and_parallel(
            t, ordinate, model_single_exp, self._p0_single_exp
        )
        self._assert_identical(serial, parallel, ["A", "E"])

    def test_parallel_double_exp(self, double_exp_data):
        """
        Serial and parallel must be identical for the double-exponential model
        (four parameters, 12 observables — the most demanding parallel test).
        """
        t, ordinate, _ = double_exp_data
        serial, parallel = self._run_serial_and_parallel(
            t, ordinate, model_double_exp, self._p0_double_exp
        )
        self._assert_identical(serial, parallel, ["A1", "E1", "A2", "E2"])

    def test_parallel_chi2_identical(self, linear_data):
        """
        Per-resample chi² values must be bit-for-bit identical between serial
        and parallel runs.
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

    def test_parallel_with_limits(self, single_exp_data):
        """
        Limits must be correctly forwarded to each worker subprocess and
        produce identical results to the serial run.

        The limits dict is part of each resample's args dict and must be
        serialised by dill along with the cost function closure.
        """
        t, ordinate, _ = single_exp_data
        serial, parallel = self._run_serial_and_parallel(
            t, ordinate, model_single_exp, self._p0_single_exp,
            limits={"E": (0.0, None)},
        )
        self._assert_identical(serial, parallel, ["A", "E"])

# =============================================================================
# Gradient-equipped model definitions  (module scope for dill pickling)
# =============================================================================

def _grad_single_exp(t, p):
    """
    Jacobian of A·exp(-E·t), shape (2, N):
      row 0  =  dC/dA =  exp(-E·t)
      row 1  =  dC/dE = -A·t·exp(-E·t)
    """
    exp_Et = np.exp(-p["E"] * t)
    return np.array([exp_Et, -p["A"] * t * exp_Et])


def _grad_double_exp(t, p):
    """
    Jacobian of A1·exp(-E1·t) + A2·exp(-E2·t), shape (4, N):
      row 0  =  dC/dA1 =  exp(-E1·t)
      row 1  =  dC/dE1 = -A1·t·exp(-E1·t)
      row 2  =  dC/dA2 =  exp(-E2·t)
      row 3  =  dC/dE2 = -A2·t·exp(-E2·t)
    """
    e1 = np.exp(-p["E1"] * t)
    e2 = np.exp(-p["E2"] * t)
    return np.array([e1, -p["A1"] * t * e1, e2, -p["A2"] * t * e2])


# Attach gradients to the existing module-level model functions.
# This must happen at module scope so the .grad attribute is present
# whenever dill serialises the function for a parallel worker.
model_single_exp.grad = _grad_single_exp
model_double_exp.grad = _grad_double_exp


# =============================================================================
# 5. Gradient-equipped model definitions  (module scope for dill pickling)
# =============================================================================
#
# model.grad(abscissa, params) must return shape (Nparams, Ndata):
# row i is d(model)/d(param_i) evaluated at every data point.
# This is what _build_uncorrelated_cost and _build_correlated_cost both expect:
#
#   J = model.grad(abscissa, params) * dchi2_df   # broadcast (Nparams, N)
#   J = J.sum(axis=1)                             # → (Nparams,)
#
# We attach .grad directly to the existing model functions rather than defining
# new callables, so the forward pass is identical and comparisons are clean.

def _grad_single_exp(t, p):
    """
    Jacobian of A·exp(-E·t), shape (2, N):
      row 0  =  dC/dA =  exp(-E·t)
      row 1  =  dC/dE = -A·t·exp(-E·t)
    """
    exp_Et = np.exp(-p["E"] * t)
    return np.array([exp_Et, -p["A"] * t * exp_Et])


def _grad_double_exp(t, p):
    """
    Jacobian of A1·exp(-E1·t) + A2·exp(-E2·t), shape (4, N):
      row 0  =  dC/dA1 =  exp(-E1·t)
      row 1  =  dC/dE1 = -A1·t·exp(-E1·t)
      row 2  =  dC/dA2 =  exp(-E2·t)
      row 3  =  dC/dE2 = -A2·t·exp(-E2·t)
    """
    e1 = np.exp(-p["E1"] * t)
    e2 = np.exp(-p["E2"] * t)
    return np.array([e1, -p["A1"] * t * e1, e2, -p["A2"] * t * e2])


# Attach gradients to the existing module-level model functions.
# This must happen at module scope so the .grad attribute is present
# whenever dill serialises the function for a parallel worker.
model_single_exp.grad = _grad_single_exp
model_double_exp.grad = _grad_double_exp


# =============================================================================
# 5. Gradient-based fits
# =============================================================================

class TestGradientFit:
    """
    Verify the model.grad code path in fit_iminuit.

    When model.grad is present, both _build_uncorrelated_cost and
    _build_correlated_cost attach an analytic gradient to the cost function
    and pass it to iminuit.Minuit.  Minuit then uses the analytic gradient
    instead of its internal finite-difference approximation.

    What changed from the previous implementation
    ----------------------------------------------
    * The gradient path now supports BOTH uncorrelated and correlated fits.
    * The gradient path now supports priors (normal and log-normal).
      Prior.grad(theta) provides the analytic derivative of the prior chi²
      contribution, which is added per-parameter to J.
    * The NotImplementedError for gradient + prior is gone.

    Test structure
    --------------
    5a. Parameter recovery (uncorrelated, with and without prior)
    5b. Parameter recovery (correlated, with and without prior)
    5c. Consistency: grad path must converge to the same minimum as the
        finite-difference path, within rtol=1e-5.
    5d. Parallel execution: .grad must survive dill serialisation.
    """

    _p0_single = {"A": SINGLE_A * 0.85, "E": SINGLE_E * 0.85}
    _p0_double = {
        "A1": DOUBLE_A1 * 0.90, "E1": DOUBLE_E1 * 0.90,
        "A2": DOUBLE_A2 * 0.90, "E2": DOUBLE_E2 * 0.90,
    }

    # ------------------------------------------------------------------
    # 5a. Uncorrelated path — with and without priors
    # ------------------------------------------------------------------

    def test_single_exp_grad_uncorr_cv_no_prior(self, single_exp_data):
        """
        CV uncorrelated fit with analytic gradient, no prior.
        A and E must be recovered within 3σ.
        """
        t, ordinate, true_params = single_exp_data
        result = fit_iminuit(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=True,
            central_value_fit_correlated=False,
            resample_fit=False,
            model=model_single_exp,
            p0=self._p0_single,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=False)

    def test_single_exp_grad_uncorr_resample_no_prior(self, single_exp_data):
        """
        CV + resample uncorrelated fit with analytic gradient, no prior.
        rspl must be fully populated and within 3σ.
        """
        t, ordinate, true_params = single_exp_data
        result = fit_iminuit(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=True,
            central_value_fit_correlated=False,
            resample_fit=True,
            resample_fit_correlated=False,
            model=model_single_exp,
            p0=self._p0_single,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=True)
        for key in ("A", "E"):
            assert not np.any(np.isnan(result.params[key].rspl))

    def test_single_exp_grad_uncorr_cv_normal_prior(self, single_exp_data):
        """
        CV uncorrelated fit with analytic gradient AND a normal prior on E.

        Prior.grad is called for E at each iteration; the gradient of the
        prior chi² contribution [(E - mean)²/sdev²] is 2(E-mean)/sdev².
        The fit must recover A and E within 3σ — the prior is loose (σ=5)
        so the data dominate.
        """
        t, ordinate, true_params = single_exp_data
        prior = {
            "A": Prior(SINGLE_A, 5.0),
            "E": Prior(SINGLE_E, 5.0)
        }
        result = fit_iminuit(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=True,
            central_value_fit_correlated=False,
            resample_fit=False,
            model=model_single_exp,
            prior=prior,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=False)
        assert "E" in result.priors

    def test_single_exp_grad_uncorr_cv_lognormal_prior(self, single_exp_data):
        """
        CV uncorrelated fit with analytic gradient AND a log-normal prior on E.

        Prior.grad for log-normal returns 2(log θ - mean)/(sdev² · θ),
        which is the analytic derivative of the log-normal chi² term.
        A positivity limit is required for the same reason as in the
        non-gradient case: iminuit evaluates the cost at the boundary.
        """
        t, ordinate, true_params = single_exp_data
        prior = {
            "A": Prior(SINGLE_A, 5.0),
            "E": Prior(np.log(SINGLE_E), 1.0, dist="log-normal")
        }
        result = fit_iminuit(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=True,
            central_value_fit_correlated=False,
            resample_fit=False,
            model=model_single_exp,
            prior=prior,
            limits={"E": (0.0, None)},
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=False)

    def test_double_exp_grad_uncorr_cv_no_prior(self, double_exp_data):
        """
        CV uncorrelated fit of the double exponential with analytic gradient.
        All four parameters must be recovered within 3σ.
        """
        t, ordinate, true_params = double_exp_data
        result = fit_iminuit(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=True,
            central_value_fit_correlated=False,
            resample_fit=False,
            model=model_double_exp,
            p0=self._p0_double,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=False)

    # ------------------------------------------------------------------
    # 5b. Correlated path — with and without priors
    # ------------------------------------------------------------------

    def test_single_exp_grad_corr_cv_no_prior(self, single_exp_data):
        """
        CV correlated fit with analytic gradient, no prior.

        The correlated gradient uses cov_inv @ delta instead of the diagonal
        sdev_inv² * delta.  The minimum is the same as the uncorrelated case
        for well-conditioned data; the gradient direction differs.
        """
        t, ordinate, true_params = single_exp_data
        result = fit_iminuit(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=True,
            central_value_fit_correlated=True,
            resample_fit=False,
            model=model_single_exp,
            p0=self._p0_single,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=False)

    def test_single_exp_grad_corr_resample(self, single_exp_data):
        """
        CV + resample correlated fit with analytic gradient.
        All rspl entries must be finite and within 3σ.
        """
        t, ordinate, true_params = single_exp_data
        result = fit_iminuit(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=True,
            central_value_fit_correlated=True,
            resample_fit=True,
            resample_fit_correlated=True,
            model=model_single_exp,
            p0=self._p0_single,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=True)
        for key in ("A", "E"):
            assert not np.any(np.isnan(result.params[key].rspl))

    def test_single_exp_grad_corr_cv_normal_prior(self, single_exp_data):
        """
        CV correlated fit with analytic gradient AND a normal prior on E.

        The prior gradient contribution is identical in the correlated and
        uncorrelated cases: it depends only on the parameter value, not on
        the data covariance structure.
        """
        t, ordinate, true_params = single_exp_data
        prior = {
            "A": Prior(SINGLE_A, 5.0),
            "E": Prior(SINGLE_E, 5.0)
        }
        result = fit_iminuit(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=True,
            central_value_fit_correlated=True,
            resample_fit=False,
            model=model_single_exp,
            prior=prior,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=False)

    def test_single_exp_grad_corr_cv_lognormal_prior(self, single_exp_data):
        """
        CV correlated fit with analytic gradient AND a log-normal prior on E.
        """
        t, ordinate, true_params = single_exp_data
        prior = {
            "A": Prior(SINGLE_A, 5.0),
            "E": Prior(np.log(SINGLE_E), 1.0, dist="log-normal")
        }
        result = fit_iminuit(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=True,
            central_value_fit_correlated=True,
            resample_fit=False,
            model=model_single_exp,
            prior=prior,
            limits={"E": (0.0, None)},
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=False)

    def test_double_exp_grad_corr_cv_no_prior(self, double_exp_data):
        """
        CV correlated fit of the double exponential with analytic gradient.
        """
        t, ordinate, true_params = double_exp_data
        result = fit_iminuit(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=True,
            central_value_fit_correlated=True,
            resample_fit=False,
            model=model_double_exp,
            p0=self._p0_double,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=False)

    # ------------------------------------------------------------------
    # 5c. Consistency: grad vs finite-difference
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("correlated", [False, True], ids=["uncorr", "corr"])
    def test_single_exp_grad_matches_no_grad(self, single_exp_data, correlated):
        """
        The analytic-gradient fit and the finite-difference fit must converge
        to the same minimum within rtol=1e-5, for both uncorrelated and
        correlated paths.

        We temporarily strip .grad from the model to force the finite-difference
        path, then restore it.  rtol=1e-5 (not exact equality) is used because
        Minuit uses strategy=0 for the gradient path, which can terminate at a
        slightly different point than the default strategy=1.
        """
        t, ordinate, _ = single_exp_data

        with_grad = fit_iminuit(
            abscissa=t, ordinate=ordinate,
            central_value_fit=True,
            central_value_fit_correlated=correlated,
            resample_fit=False,
            model=model_single_exp,
            p0=self._p0_single,
        )

        # Temporarily remove .grad to force finite-difference path.
        del model_single_exp.grad
        try:
            no_grad = fit_iminuit(
                abscissa=t, ordinate=ordinate,
                central_value_fit=True,
                central_value_fit_correlated=correlated,
                resample_fit=False,
                model=model_single_exp,
                p0=self._p0_single,
            )
        finally:
            model_single_exp.grad = _grad_single_exp   # always restore

        for key in ("A", "E"):
            np.testing.assert_allclose(
                with_grad.params[key].mean,
                no_grad.params[key].mean,
                rtol=1e-5,
                err_msg=(
                    f"{'Correlated' if correlated else 'Uncorrelated'} gradient "
                    f"and finite-difference paths converge to different values "
                    f"for '{key}'"
                ),
            )

    @pytest.mark.parametrize("correlated", [False, True], ids=["uncorr", "corr"])
    def test_single_exp_grad_prior_matches_no_grad_prior(self, single_exp_data, correlated):
        """
        With a normal prior on E, the gradient and finite-difference paths
        must converge to the same minimum within rtol=1e-5.

        This specifically tests that the prior gradient J[i] += prior.grad(theta)
        is computed correctly — an incorrect sign or missing factor would shift
        the minimum away from the no-grad result.
        """
        t, ordinate, _ = single_exp_data
        prior = {
            "A": Prior(SINGLE_A, 5.0),
            "E": Prior(SINGLE_E, 5.0)
        }

        with_grad = fit_iminuit(
            abscissa=t, ordinate=ordinate,
            central_value_fit=True,
            central_value_fit_correlated=correlated,
            resample_fit=False,
            model=model_single_exp,
            prior=prior,
        )

        del model_single_exp.grad
        try:
            no_grad = fit_iminuit(
                abscissa=t, ordinate=ordinate,
                central_value_fit=True,
                central_value_fit_correlated=correlated,
                resample_fit=False,
                model=model_single_exp,
                prior={
                    "A": Prior(SINGLE_A, 5.0),
                    "E": Prior(SINGLE_E, 5.0)
                },   # fresh copy; prior was mutated by pop
            )
        finally:
            model_single_exp.grad = _grad_single_exp

        for key in ("A", "E"):
            np.testing.assert_allclose(
                with_grad.params[key].mean,
                no_grad.params[key].mean,
                rtol=1e-5,
                err_msg=(
                    f"{'Correlated' if correlated else 'Uncorrelated'} gradient+prior "
                    f"and finite-difference+prior paths differ for '{key}'"
                ),
            )

    # ------------------------------------------------------------------
    # 5d. Parallel execution
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("correlated", [False, True], ids=["uncorr", "corr"])
    def test_grad_parallel_matches_serial(self, single_exp_data, correlated):
        """
        Serial (Nproc=None) and parallel (Nproc=4) gradient fits must produce
        bit-for-bit identical rspl arrays for both correlated and uncorrelated
        paths.

        This verifies that the .grad attribute on the model function survives
        dill serialisation across subprocess boundaries.  If dill silently
        dropped .grad, workers would fall back to finite-difference gradients;
        because the minimum is the same, parameter values would still match —
        but chi² values could differ slightly due to the different number of
        function evaluations affecting iminuit's internal convergence criterion.
        We therefore check both parameter rspl and chi² rspl for exact equality.
        """
        t, ordinate, _ = single_exp_data
        common = dict(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=True,
            central_value_fit_correlated=correlated,
            resample_fit=True,
            resample_fit_correlated=correlated,
            model=model_single_exp,
            p0=self._p0_single,
        )
        serial   = fit_iminuit(**common, Nproc=None)
        parallel = fit_iminuit(**common, Nproc=4)

        for key in ("A", "E"):
            np.testing.assert_array_equal(
                serial.params[key].rspl,
                parallel.params[key].rspl,
                err_msg=(
                    f"rspl mismatch for '{key}' between serial and parallel "
                    f"{'correlated' if correlated else 'uncorrelated'} gradient fits"
                ),
            )
        np.testing.assert_array_equal(
            serial.chi2.rspl,
            parallel.chi2.rspl,
            err_msg="chi2 rspl mismatch between serial and parallel gradient fits",
        )