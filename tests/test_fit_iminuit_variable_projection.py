"""
Unit tests for the iminuit variable-projection backend
(fit_iminuit_variable_projection.py / fit.py with backend='iminuit' and
linear_params in kwargs).

Overview
--------
1. Input-validation tests
     – Same contract as plain iminuit, plus variable-projection-specific
       guards (empty linear_params, prior on a linear parameter is dropped
       with a warning).

2. End-to-end parameter recovery:
     2a. Single-exponential  C(t) = A·exp(-E·t)
         linear_params=["A"],  nonlinear_params=["E"]
     2b. Double-exponential  C(t) = A1·exp(-E1·t) + A2·exp(-E2·t)
         linear_params=["A1","A2"],  nonlinear_params=["E1","E2"]
   Each model is exercised over the full strategy matrix
   (cv/resample) × (correlated/uncorrelated), same as the iminuit suite.

3. Normal and log-normal priors on energy parameters.

4. Parallel-execution tests (Nproc=4) must give bit-for-bit identical
   results to Nproc=None — for both uncorrelated and correlated paths.

Design notes
------------
* The linear model is omitted: variable projection only makes sense when there
  is at least one genuinely nonlinear parameter.  A pure-linear model (y=mx+b)
  would be better handled by the linear-regression backend.
* Model functions are at module scope for dill-serializability in parallel
  workers.
* Fixtures use seed offsets distinct from the plain iminuit suite so the two
  test modules exercise independent realisations of the mock data.
* `_check_params_recovered` is identical in contract to the iminuit version;
  it is duplicated here rather than imported from a conftest so each test
  module is self-contained (to be merged into conftest.py later).
* All log-normal prior tests pair the prior with a positivity limit, consistent
  with the approach established in test_fit_iminuit.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from correlatoranalyser import Data
# Import the varproj module's fit_iminuit directly to avoid ambiguity with
# the plain iminuit backend exposed under the same name via fit.py routing.
from correlatoranalyser.fit_iminuit_variable_projection import fit_iminuit as fit_varproj
from correlatoranalyser.prior import Prior

# =============================================================================
# Global test configuration  (identical values to both iminuit test suites)
# =============================================================================

_RNG_SEED = 20240101
NRAW: int  = 600
NBST: int  = 300
NSIG: int  = 3

# =============================================================================
# Ground-truth parameters
# =============================================================================

SINGLE_A:     float = 1.5
SINGLE_E:     float = 0.3
SINGLE_NOISE: float = 0.02

DOUBLE_A1:    float = 1.5
DOUBLE_E1:    float = 0.3
DOUBLE_A2:    float = 0.5
DOUBLE_E2:    float = 0.8
DOUBLE_NOISE: float = 0.01

# =============================================================================
# Model functions  (module-level for dill pickling in parallel workers)
# =============================================================================

def model_single_exp(t, p):
    """Single-exponential model: C(t) = A·exp(-E·t)."""
    return p["A"] * np.exp(-p["E"] * t)


def model_double_exp(t, p):
    """Double-exponential model: C(t) = A1·exp(-E1·t) + A2·exp(-E2·t)."""
    return p["A1"] * np.exp(-p["E1"] * t) + p["A2"] * np.exp(-p["E2"] * t)


# A minimal model with no linear parameter, used only for the validation test
# that checks the empty-linear_params guard.
def model_linear(x, p):
    """Linear model used only in validation tests."""
    return p["m"] * x + p["b"]


# =============================================================================
# Mock-data factory  (identical to both other iminuit suites)
# =============================================================================

def _make_bootstrap_data(
    true_values: np.ndarray,
    noise_scale: float,
    rng: np.random.Generator,
    nraw: int = NRAW,
    nbst: int = NBST,
) -> Data:
    """
    Bootstrap Data from a mildly correlated multivariate Gaussian.
    Covariance = noise_scale² · (I + 0.1 · 11ᵀ).
    """
    n   = len(true_values)
    cov = noise_scale ** 2 * (np.eye(n) + 0.1 * np.ones((n, n)))
    raw = rng.multivariate_normal(mean=true_values, cov=cov, size=nraw)
    return Data(resample_type="bst", data=raw, Nresample=nbst)


# =============================================================================
# Module-scoped fixtures  (seed offsets +20/+21/+22 to be independent of the
# plain iminuit fixtures at +10/+11/+12)
# =============================================================================

@pytest.fixture(scope="module")
def single_exp_data():
    """t=1..10 single-exponential mock data."""
    rng   = np.random.default_rng(_RNG_SEED + 20)
    t     = np.arange(1, 11, dtype=float)
    true_C = SINGLE_A * np.exp(-SINGLE_E * t)
    ordinate = _make_bootstrap_data(true_C, SINGLE_NOISE, rng)
    return t, ordinate, {"A": SINGLE_A, "E": SINGLE_E}


@pytest.fixture(scope="module")
def double_exp_data():
    """t=1..12 double-exponential mock data."""
    rng    = np.random.default_rng(_RNG_SEED + 21)
    t      = np.arange(1, 13, dtype=float)
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
# Shared assertion helper
# =============================================================================

def _check_params_recovered(
    fit_result,
    true_params: dict,
    central_value_fit: bool,
    resample_fit: bool,
    nsig: int = NSIG,
) -> None:
    """
    Assert every parameter lies within nsig σ of its true value.

    This is identical in contract to the helper in test_fit_iminuit.py.
    Both linear (A) and nonlinear (E) parameters are checked — variable
    projection recovers both, even though only the nonlinear parameters
    are passed to Minuit.
    """
    for key, true_val in true_params.items():
        assert key in fit_result.params, f"Parameter '{key}' missing from FitResult"

        if resample_fit:
            rspl_vals = fit_result.params[key].rspl
            assert rspl_vals is not None, f"rspl is None for '{key}'"
            assert not np.any(np.isnan(rspl_vals)), f"NaN in rspl for '{key}'"

            estimate    = fit_result.params[key].mean if central_value_fit else float(np.mean(rspl_vals))
            uncertainty = np.std(rspl_vals, ddof=1)
        else:
            # CV-only: use Hessian error.  Note that Hessian errors are only
            # available for nonlinear parameters (Minuit only tracks those);
            # for linear parameters we fall back to checking the CV value
            # against a generous absolute tolerance derived from the noise level.
            if key in fit_result.params_hessian_err:
                estimate    = fit_result.params[key].mean
                uncertainty = fit_result.params_hessian_err[key].mean
            else:
                # Linear parameter: no Hessian error available.
                # Use 5 % of the true value as a generous tolerance.
                estimate    = fit_result.params[key].mean
                uncertainty = abs(true_val) * 0.05
                assert uncertainty > 0, f"Fallback uncertainty for '{key}' is zero"

        assert uncertainty > 0, f"Uncertainty for '{key}' is non-positive: {uncertainty}"
        deviation = abs(estimate - true_val)
        assert deviation < nsig * uncertainty, (
            f"Parameter '{key}': |{estimate:.6g} - {true_val:.6g}| = {deviation:.3g} "
            f">= {nsig} × {uncertainty:.3g}.  Fit is more than {nsig}σ from truth."
        )


# =============================================================================
# Fit-strategy parametrisation  (same matrix as both iminuit suites)
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
    Confirm that fit_varproj raises informative errors for bad inputs.

    The general validation (ordinate type, abscissa shape, cv/rs flags, model,
    prior/p0) is shared via _validate_inputs.  Variable-projection-specific
    guards are tested separately below.
    """

    _rng      = np.random.default_rng(1)
    _raw      = _rng.normal(1.0, 0.05, size=(100, 10))
    _ordinate = Data(resample_type="bst", data=_raw, Nresample=50)
    _t        = np.arange(1, 11, dtype=float)
    _p0       = {"A": SINGLE_A * 0.85, "E": SINGLE_E * 0.85}

    def test_ordinate_must_be_Data_not_ndarray(self):
        """Plain numpy array ordinate must raise TypeError."""
        with pytest.raises(TypeError):
            fit_varproj(
                abscissa=self._t,
                ordinate=np.ones(10),
                linear_params=["A"],
                model=model_single_exp,
                p0=self._p0,
            )

    def test_ordinate_must_be_Data_not_list(self):
        """List ordinate must raise TypeError."""
        with pytest.raises(TypeError):
            fit_varproj(
                abscissa=self._t,
                ordinate=list(range(10)),
                linear_params=["A"],
                model=model_single_exp,
                p0=self._p0,
            )

    def test_ordinate_must_be_Data_not_scalar(self):
        """Scalar ordinate must raise TypeError."""
        with pytest.raises(TypeError):
            fit_varproj(
                abscissa=self._t,
                ordinate=1.0,
                linear_params=["A"],
                model=model_single_exp,
                p0=self._p0,
            )

    def test_abscissa_scalar_raises(self):
        """Scalar abscissa has no length; some exception must be raised."""
        with pytest.raises(Exception):
            fit_varproj(
                abscissa=42,
                ordinate=self._ordinate,
                linear_params=["A"],
                model=model_single_exp,
                p0=self._p0,
            )

    def test_abscissa_none_raises(self):
        """None abscissa must raise an exception."""
        with pytest.raises(Exception):
            fit_varproj(
                abscissa=None,
                ordinate=self._ordinate,
                linear_params=["A"],
                model=model_single_exp,
                p0=self._p0,
            )

    def test_neither_cv_nor_resample_raises(self):
        """Both fit flags False must raise ValueError immediately."""
        with pytest.raises(ValueError, match="At least one of"):
            fit_varproj(
                abscissa=self._t,
                ordinate=self._ordinate,
                linear_params=["A"],
                model=model_single_exp,
                p0=self._p0,
                central_value_fit=False,
                resample_fit=False,
            )

    def test_neither_prior_nor_p0_raises(self):
        """Omitting both prior and p0 must raise ValueError."""
        with pytest.raises(ValueError):
            fit_varproj(
                abscissa=self._t,
                ordinate=self._ordinate,
                linear_params=["A"],
                model=model_single_exp,
            )

    def test_no_model_raises(self):
        """model=None must raise ValueError."""
        with pytest.raises(ValueError):
            fit_varproj(
                abscissa=self._t,
                ordinate=self._ordinate,
                linear_params=["A"],
                model=None,
                p0=self._p0,
            )

    def test_empty_linear_params_raises(self):
        """
        Variable projection requires at least one linear parameter.
        Passing an empty list must raise ValueError inside the cost-function
        builder before any Minuit object is constructed.
        """
        with pytest.raises(ValueError, match="at least one linear parameter"):
            fit_varproj(
                abscissa=self._t,
                ordinate=self._ordinate,
                linear_params=[],          # ← no linear parameters
                model=model_single_exp,
                p0=self._p0,
            )

    def test_prior_on_linear_param_is_dropped_with_warning(self, single_exp_data):
        """
        A prior on a linear parameter (A) is meaningless because A is
        determined analytically and never evaluated by the prior cost term.
        The code pops the key and prints a warning; the fit must still succeed.

        We check:
          a) no exception is raised
          b) the result contains both A and E
          c) A is NOT in result.priors (it was dropped)
        """
        t, ordinate, _ = single_exp_data
        prior = {
            "A": Prior(SINGLE_A, 5.0),     # ← will be silently dropped
            "E": Prior(SINGLE_E, 5.0),
        }
        # fit_iminuit pops from the prior dict in-place and prints; we just
        # check the outcome is correct.
        result = fit_varproj(
            abscissa=t,
            ordinate=ordinate,
            linear_params=["A"],
            model=model_single_exp,
            prior=prior,
            central_value_fit=True,
            resample_fit=False,
        )
        assert "A" not in result.priors, (
            "Prior for linear parameter 'A' should have been dropped"
        )
        assert "A" in result.params, "Linear parameter 'A' must still appear in params"


# =============================================================================
# 2a. End-to-end tests – single-exponential model
# =============================================================================

class TestSingleExpFit:
    """
    Fit C(t) = A·exp(-E·t) with A treated as a linear parameter.

    The variable-projection optimiser minimises over E only; A is recovered
    analytically at each evaluation of the cost function.  Both parameters
    must appear in FitResult.params after the fit.
    """

    _p0 = {"A": SINGLE_A * 0.85, "E": SINGLE_E * 0.85}

    @pytest.mark.parametrize(
        "cv, cv_corr, rs, rs_corr",
        _STRATEGY_PARAMS,
        ids=_STRATEGY_IDS,
    )
    def test_parameter_recovery(self, single_exp_data, cv, cv_corr, rs, rs_corr):
        """
        Both A and E must be recovered within 3σ of the truth for every
        strategy combination.

        Hessian errors are only available for the nonlinear parameter E;
        for A the fallback 5 % tolerance is used (see _check_params_recovered).
        """
        t, ordinate, true_params = single_exp_data
        result = fit_varproj(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=cv,
            central_value_fit_correlated=cv_corr,
            resample_fit=rs,
            resample_fit_correlated=rs_corr,
            linear_params=["A"],
            model=model_single_exp,
            p0=self._p0,
        )
        _check_params_recovered(result, true_params, central_value_fit=cv, resample_fit=rs)

    @pytest.mark.parametrize(
        "cv, cv_corr, rs, rs_corr",
        _STRATEGY_PARAMS,
        ids=_STRATEGY_IDS,
    )
    def test_linear_param_present_in_result(self, single_exp_data, cv, cv_corr, rs, rs_corr):
        """
        The analytically-determined linear parameter A must appear in
        FitResult.params alongside the nonlinear parameter E.

        This verifies that import_from_iminuit correctly stores the
        variable_projection dict alongside the Minuit parameters.
        """
        t, ordinate, _ = single_exp_data
        result = fit_varproj(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=cv,
            central_value_fit_correlated=cv_corr,
            resample_fit=rs,
            resample_fit_correlated=rs_corr,
            linear_params=["A"],
            model=model_single_exp,
            p0=self._p0,
        )
        assert "A" in result.params, "Linear parameter 'A' missing from FitResult.params"
        assert "E" in result.params, "Nonlinear parameter 'E' missing from FitResult.params"

    @pytest.mark.parametrize(
        "cv, cv_corr, rs, rs_corr",
        _STRATEGY_PARAMS,
        ids=_STRATEGY_IDS,
    )
    def test_resample_count_and_no_nan(self, single_exp_data, cv, cv_corr, rs, rs_corr):
        """Every resample slot for both A and E must be populated and finite."""
        t, ordinate, _ = single_exp_data
        result = fit_varproj(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=cv,
            central_value_fit_correlated=cv_corr,
            resample_fit=rs,
            resample_fit_correlated=rs_corr,
            linear_params=["A"],
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
                    f"NaN values in rspl for '{key}'"
                )

    @pytest.mark.parametrize(
        "cv, cv_corr, rs, rs_corr",
        _STRATEGY_PARAMS,
        ids=_STRATEGY_IDS,
    )
    def test_fit_quality_metadata(self, single_exp_data, cv, cv_corr, rs, rs_corr):
        """dof > 0, chi² > 0, and chi² rspl entries are finite for resample fits."""
        t, ordinate, _ = single_exp_data
        result = fit_varproj(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=cv,
            central_value_fit_correlated=cv_corr,
            resample_fit=rs,
            resample_fit_correlated=rs_corr,
            linear_params=["A"],
            model=model_single_exp,
            p0=self._p0,
        )
        assert result.dof is not None and result.dof > 0

        if cv:
            chi2_val = result.chi2.mean if hasattr(result.chi2, "mean") else float(result.chi2)
            assert chi2_val > 0

        if rs:
            assert np.all(np.isfinite(result.chi2.rspl))
            assert np.all(result.chi2.rspl > 0)


# =============================================================================
# 2b. End-to-end tests – double-exponential model
# =============================================================================

class TestDoubleExpFit:
    """
    Fit C(t) = A1·exp(-E1·t) + A2·exp(-E2·t) with A1,A2 as linear parameters.

    The variable-projection optimiser minimises over E1 and E2 only.  All four
    parameters must appear in FitResult.params.

    Well-separated energies (E2/E1 ≈ 2.7) and p0 10 % below truth to keep the
    double-exp identifiable without priors.
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
        result = fit_varproj(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=cv,
            central_value_fit_correlated=cv_corr,
            resample_fit=rs,
            resample_fit_correlated=rs_corr,
            linear_params=["A1", "A2"],
            model=model_double_exp,
            p0=self._p0,
        )
        _check_params_recovered(result, true_params, central_value_fit=cv, resample_fit=rs)

    @pytest.mark.parametrize(
        "cv, cv_corr, rs, rs_corr",
        _STRATEGY_PARAMS,
        ids=_STRATEGY_IDS,
    )
    def test_both_linear_params_present(self, double_exp_data, cv, cv_corr, rs, rs_corr):
        """
        Both analytically-eliminated amplitudes A1 and A2 must be stored in
        FitResult.params alongside E1 and E2.
        """
        t, ordinate, _ = double_exp_data
        result = fit_varproj(
            abscissa=t,
            ordinate=ordinate,
            central_value_fit=cv,
            central_value_fit_correlated=cv_corr,
            resample_fit=rs,
            resample_fit_correlated=rs_corr,
            linear_params=["A1", "A2"],
            model=model_double_exp,
            p0=self._p0,
        )
        for key in ("A1", "A2", "E1", "E2"):
            assert key in result.params, f"Parameter '{key}' missing from FitResult.params"


# =============================================================================
# 3. Normal prior fits
# =============================================================================

class TestNormalPriorFit:
    """
    Verify normal prior injection on nonlinear (energy) parameters.

    Priors on linear parameters are dropped silently (tested in validation).
    Priors on nonlinear parameters contribute a ((E - mean) / sdev)² term
    to the cost function, identical to the plain iminuit path.
    """

    def test_single_exp_normal_prior_cv(self, single_exp_data):
        """CV fit with a loose normal prior on E must recover both A and E within 3σ."""
        t, ordinate, true_params = single_exp_data
        prior = {"E": Prior(SINGLE_E, 5.0)}
        result = fit_varproj(
            abscissa=t,
            ordinate=ordinate,
            linear_params=["A"],
            model=model_single_exp,
            prior={**prior, "A": Prior(SINGLE_A, 5.0)},   # A will be dropped
            central_value_fit=True,
            resample_fit=False,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=False)
        # E prior should be stored; A prior should have been dropped.
        assert "E" in result.priors
        assert "A" not in result.priors

    def test_single_exp_normal_prior_cv_and_resample(self, single_exp_data):
        """Combined CV + resample with a normal prior on E."""
        t, ordinate, true_params = single_exp_data
        prior = {"E": Prior(SINGLE_E, 5.0)}
        result = fit_varproj(
            abscissa=t,
            ordinate=ordinate,
            linear_params=["A"],
            model=model_single_exp,
            prior=prior,
            central_value_fit=True,
            resample_fit=True,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=True)
        for key in ("A", "E"):
            assert result.params[key].rspl.shape == (NBST,)
            assert not np.any(np.isnan(result.params[key].rspl))

    def test_double_exp_normal_priors_on_energies(self, double_exp_data):
        """CV fit of double-exp with normal priors on both E1 and E2."""
        t, ordinate, true_params = double_exp_data
        prior = {
            "E1": Prior(DOUBLE_E1, 5.0),
            "E2": Prior(DOUBLE_E2, 5.0),
        }
        result = fit_varproj(
            abscissa=t,
            ordinate=ordinate,
            linear_params=["A1", "A2"],
            model=model_double_exp,
            prior=prior,
            central_value_fit=True,
            resample_fit=False,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=False)


# =============================================================================
# 4. Log-normal prior fits
# =============================================================================

class TestLogNormalPriorFit:
    """
    Verify log-normal prior injection on nonlinear energy parameters.

    The considerations are identical to test_fit_iminuit.py:
      - iminuit operates in linear parameter space.
      - Prior.__call__ returns a large finite penalty for theta ≤ 0 (after fix).
      - Positivity limits are paired with log-normal priors as belt-and-braces.

    Prior encoding:  Prior(mean=np.log(E_true), sdev=..., dist="log-normal")
    """

    def test_single_exp_lognormal_energy_cv(self, single_exp_data):
        """
        CV fit with a log-normal prior on E and a positivity limit.
        Both A and E must be recovered within 3σ.
        """
        t, ordinate, true_params = single_exp_data
        prior = {"E": Prior(np.log(SINGLE_E), 1.0, dist="log-normal")}
        result = fit_varproj(
            abscissa=t,
            ordinate=ordinate,
            linear_params=["A"],
            model=model_single_exp,
            prior=prior,
            limits={"E": (0.0, None)},
            central_value_fit=True,
            resample_fit=False,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=False)
        assert "E" in result.priors
        assert result.priors["E"].dist == "log-normal"

    def test_single_exp_lognormal_energy_resample(self, single_exp_data):
        """
        Resample fits with a log-normal prior on E.
        All rspl entries for A and E must be finite; E must be positive.
        """
        t, ordinate, true_params = single_exp_data
        prior = {"E": Prior(np.log(SINGLE_E), 1.0, dist="log-normal")}
        result = fit_varproj(
            abscissa=t,
            ordinate=ordinate,
            linear_params=["A"],
            model=model_single_exp,
            prior=prior,
            limits={"E": (0.0, None)},
            central_value_fit=False,
            resample_fit=True,
        )
        _check_params_recovered(result, true_params, central_value_fit=False, resample_fit=True)
        for key in ("A", "E"):
            assert result.params[key].rspl.shape == (NBST,)
            assert not np.any(np.isnan(result.params[key].rspl))
        assert np.all(result.params["E"].rspl > 0), (
            "Some resample energies are ≤ 0; positivity limit was not enforced."
        )

    def test_single_exp_lognormal_cv_and_resample(self, single_exp_data):
        """Combined CV + resample with a log-normal prior on E."""
        t, ordinate, true_params = single_exp_data
        prior = {"E": Prior(np.log(SINGLE_E), 1.0, dist="log-normal")}
        result = fit_varproj(
            abscissa=t,
            ordinate=ordinate,
            linear_params=["A"],
            model=model_single_exp,
            prior=prior,
            limits={"E": (0.0, None)},
            central_value_fit=True,
            resample_fit=True,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=True)

    def test_double_exp_lognormal_energies_cv(self, double_exp_data):
        """
        CV fit of the double exponential with log-normal priors on E1 and E2.

        Moderately tight priors (sdev=0.5) break the E1↔E2 swap degeneracy.
        A1 and A2 are linear parameters and must NOT receive priors.
        """
        t, ordinate, true_params = double_exp_data
        prior = {
            "E1": Prior(np.log(DOUBLE_E1), 0.3, dist="log-normal"),
            "E2": Prior(np.log(DOUBLE_E2), 0.3, dist="log-normal"),
        }
        result = fit_varproj(
            abscissa=t,
            ordinate=ordinate,
            linear_params=["A1", "A2"],
            model=model_double_exp,
            prior=prior,
            limits={"E1": (0.0, None), "E2": (0.0, None)},
            central_value_fit=True,
            resample_fit=False,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=False)
        for key in ("E1", "E2"):
            assert result.priors[key].dist == "log-normal"

    def test_double_exp_lognormal_energies_cv_and_resample(self, double_exp_data):
        """
        Combined CV + resample for the double exponential with log-normal
        priors on both energies.  The most realistic Lattice QCD use case
        for the variable-projection backend.
        """
        t, ordinate, true_params = double_exp_data
        prior = {
            "E1": Prior(np.log(DOUBLE_E1), 0.3, dist="log-normal"),
            "E2": Prior(np.log(DOUBLE_E2), 0.3, dist="log-normal"),
        }
        result = fit_varproj(
            abscissa=t,
            ordinate=ordinate,
            linear_params=["A1", "A2"],
            model=model_double_exp,
            prior=prior,
            limits={"E1": (0.0, None), "E2": (0.0, None)},
            central_value_fit=True,
            resample_fit=True,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=True)
        for key in ("A1", "A2", "E1", "E2"):
            assert result.params[key].rspl.shape == (NBST,)
            assert not np.any(np.isnan(result.params[key].rspl))
        assert np.all(result.params["E1"].rspl > 0)
        assert np.all(result.params["E2"].rspl > 0)

# =============================================================================
# 6. Gradient-based fits (variable projection)
# =============================================================================

# =============================================================================
# Gradient-equipped model definitions  (module scope for dill pickling)
#
# KEY DIFFERENCE from plain iminuit: model.grad must return shape
# (N_nonlinear_params, N_data) — only derivatives w.r.t. the nonlinear
# parameters that Minuit actually optimises over.
#
# For single-exp with linear_params=["A"], nonlinear_params=["E"]:
#   model.grad returns shape (1, N): [ dC/dE ]
#
# For double-exp with linear_params=["A1","A2"], nonlinear_params=["E1","E2"]:
#   model.grad returns shape (2, N): [ dC/dE1, dC/dE2 ]
#
# The linear parameter derivatives (dC/dA, dC/dA1, dC/dA2) are NOT included
# because those parameters are eliminated analytically and never appear in
# Minuit's parameter vector.
# =============================================================================

def _grad_single_exp_varproj(t, p):
    """
    Nonlinear-only Jacobian of A·exp(-E·t), shape (1, N):
      row 0  =  dC/dE = -A·t·exp(-E·t)

    dC/dA = exp(-E·t) is intentionally omitted because A is a linear
    parameter eliminated by the variable-projection step.
    """
    return np.array([-p["A"] * t * np.exp(-p["E"] * t)])


def _grad_double_exp_varproj(t, p):
    """
    Nonlinear-only Jacobian of A1·exp(-E1·t) + A2·exp(-E2·t), shape (2, N):
      row 0  =  dC/dE1 = -A1·t·exp(-E1·t)
      row 1  =  dC/dE2 = -A2·t·exp(-E2·t)

    dC/dA1 and dC/dA2 are omitted — A1 and A2 are linear parameters.
    """
    e1 = np.exp(-p["E1"] * t)
    e2 = np.exp(-p["E2"] * t)
    return np.array([-p["A1"] * t * e1, -p["A2"] * t * e2])


# Attach gradients at module scope so dill finds them when pickling
# for parallel workers.
model_single_exp.grad = _grad_single_exp_varproj
model_double_exp.grad = _grad_double_exp_varproj


class TestGradientFit:
    """
    Verify the model.grad code path in the variable-projection backend.

    Structural differences from TestGradientFit in test_fit_iminuit.py
    -------------------------------------------------------------------
    * model.grad returns shape (N_nonlinear, N_data) — only the nonlinear
      parameter rows.  The linear parameter rows (dC/dA etc.) are excluded
      because A is resolved analytically and never enters Minuit's gradient.
    * Priors are placed only on energy parameters (nonlinear).  Priors on
      linear parameters are silently dropped by the backend and are therefore
      not used here at all.
    * The prior gradient loop in the varproj grad closure iterates over
      nonlinear_params only, consistent with the Jacobian shape.

    Test structure
    --------------
    6a. Parameter recovery (uncorrelated, with and without prior on E)
    6b. Parameter recovery (correlated, with and without prior on E)
    6c. Consistency: grad path must converge to same minimum as
        finite-difference path, within rtol=1e-5.
    6d. Parallel execution: .grad must survive dill serialisation.
    """

    _p0_single = {"A": SINGLE_A * 0.85, "E": SINGLE_E * 0.85}
    _p0_double = {
        "A1": DOUBLE_A1 * 0.90, "E1": DOUBLE_E1 * 0.90,
        "A2": DOUBLE_A2 * 0.90, "E2": DOUBLE_E2 * 0.90,
    }

    # ------------------------------------------------------------------
    # 6a. Uncorrelated path — with and without prior on E
    # ------------------------------------------------------------------

    def test_single_exp_grad_uncorr_cv_no_prior(self, single_exp_data):
        """
        CV uncorrelated fit with analytic gradient, no prior.
        Both A (linear, resolved analytically) and E (nonlinear, optimised
        by Minuit) must be recovered within 3σ.
        """
        t, ordinate, true_params = single_exp_data
        result = fit_varproj(
            abscissa=t,
            ordinate=ordinate,
            linear_params=["A"],
            model=model_single_exp,
            p0=self._p0_single,
            central_value_fit=True,
            central_value_fit_correlated=False,
            resample_fit=False,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=False)

    def test_single_exp_grad_uncorr_resample_no_prior(self, single_exp_data):
        """
        CV + resample uncorrelated fit with analytic gradient, no prior.
        rspl for both A and E must be fully populated and within 3σ.
        """
        t, ordinate, true_params = single_exp_data
        result = fit_varproj(
            abscissa=t,
            ordinate=ordinate,
            linear_params=["A"],
            model=model_single_exp,
            p0=self._p0_single,
            central_value_fit=True,
            central_value_fit_correlated=False,
            resample_fit=True,
            resample_fit_correlated=False,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=True)
        for key in ("A", "E"):
            assert not np.any(np.isnan(result.params[key].rspl))

    def test_single_exp_grad_uncorr_cv_normal_prior(self, single_exp_data):
        """
        CV uncorrelated fit with analytic gradient AND a normal prior on E.

        Only E receives a prior — A is a linear parameter and would be
        silently dropped if included.  The prior gradient for E is
        2(E - mean)/sdev², added to J[0] (the only nonlinear row).
        """
        t, ordinate, true_params = single_exp_data
        result = fit_varproj(
            abscissa=t,
            ordinate=ordinate,
            linear_params=["A"],
            model=model_single_exp,
            prior={"E": Prior(SINGLE_E, 5.0)},
            central_value_fit=True,
            central_value_fit_correlated=False,
            resample_fit=False,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=False)
        assert "E" in result.priors

    def test_single_exp_grad_uncorr_cv_lognormal_prior(self, single_exp_data):
        """
        CV uncorrelated fit with analytic gradient AND a log-normal prior on E.

        The prior gradient for log-normal is 2(log θ - mean)/(sdev² · θ),
        added to J[0].  A positivity limit is paired with the log-normal
        prior as in the non-gradient case.
        """
        t, ordinate, true_params = single_exp_data
        result = fit_varproj(
            abscissa=t,
            ordinate=ordinate,
            linear_params=["A"],
            model=model_single_exp,
            prior={"E": Prior(np.log(SINGLE_E), 1.0, dist="log-normal")},
            limits={"E": (0.0, None)},
            central_value_fit=True,
            central_value_fit_correlated=False,
            resample_fit=False,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=False)

    def test_double_exp_grad_uncorr_cv_no_prior(self, double_exp_data):
        """
        CV uncorrelated fit of the double exponential with analytic gradient.
        model.grad returns shape (2, N) — only the E1 and E2 rows.
        All four parameters must be recovered within 3σ.
        """
        t, ordinate, true_params = double_exp_data
        result = fit_varproj(
            abscissa=t,
            ordinate=ordinate,
            linear_params=["A1", "A2"],
            model=model_double_exp,
            p0=self._p0_double,
            central_value_fit=True,
            central_value_fit_correlated=False,
            resample_fit=False,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=False)

    def test_double_exp_grad_uncorr_cv_normal_priors(self, double_exp_data):
        """
        CV uncorrelated fit of the double exponential with normal priors on
        E1 and E2.  The prior gradient adds one scalar to J[0] and J[1]
        respectively, with no contribution from the linear rows.
        """
        t, ordinate, true_params = double_exp_data
        result = fit_varproj(
            abscissa=t,
            ordinate=ordinate,
            linear_params=["A1", "A2"],
            model=model_double_exp,
            prior={
                "E1": Prior(DOUBLE_E1, 5.0),
                "E2": Prior(DOUBLE_E2, 5.0),
            },
            central_value_fit=True,
            central_value_fit_correlated=False,
            resample_fit=False,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=False)

    # ------------------------------------------------------------------
    # 6b. Correlated path — with and without prior on E
    # ------------------------------------------------------------------

    def test_single_exp_grad_corr_cv_no_prior(self, single_exp_data):
        """
        CV correlated fit with analytic gradient, no prior.

        The correlated varproj gradient uses LT @ Phi and cov_inv @ delta
        instead of the diagonal equivalents.  The minimum is the same as
        the uncorrelated case; only the gradient direction differs.
        """
        t, ordinate, true_params = single_exp_data
        result = fit_varproj(
            abscissa=t,
            ordinate=ordinate,
            linear_params=["A"],
            model=model_single_exp,
            p0=self._p0_single,
            central_value_fit=True,
            central_value_fit_correlated=True,
            resample_fit=False,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=False)

    def test_single_exp_grad_corr_resample(self, single_exp_data):
        """
        CV + resample correlated fit with analytic gradient.
        All rspl entries for A and E must be finite and within 3σ.
        """
        t, ordinate, true_params = single_exp_data
        result = fit_varproj(
            abscissa=t,
            ordinate=ordinate,
            linear_params=["A"],
            model=model_single_exp,
            p0=self._p0_single,
            central_value_fit=True,
            central_value_fit_correlated=True,
            resample_fit=True,
            resample_fit_correlated=True,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=True)
        for key in ("A", "E"):
            assert not np.any(np.isnan(result.params[key].rspl))

    def test_single_exp_grad_corr_cv_normal_prior(self, single_exp_data):
        """
        CV correlated fit with analytic gradient AND a normal prior on E.

        The prior gradient contribution is identical in the correlated and
        uncorrelated varproj cases: it depends only on the parameter value,
        not on the data covariance structure.
        """
        t, ordinate, true_params = single_exp_data
        result = fit_varproj(
            abscissa=t,
            ordinate=ordinate,
            linear_params=["A"],
            model=model_single_exp,
            prior={"E": Prior(SINGLE_E, 5.0)},
            central_value_fit=True,
            central_value_fit_correlated=True,
            resample_fit=False,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=False)

    def test_single_exp_grad_corr_cv_lognormal_prior(self, single_exp_data):
        """
        CV correlated fit with analytic gradient AND a log-normal prior on E.
        """
        t, ordinate, true_params = single_exp_data
        result = fit_varproj(
            abscissa=t,
            ordinate=ordinate,
            linear_params=["A"],
            model=model_single_exp,
            prior={"E": Prior(np.log(SINGLE_E), 1.0, dist="log-normal")},
            limits={"E": (0.0, None)},
            central_value_fit=True,
            central_value_fit_correlated=True,
            resample_fit=False,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=False)

    def test_double_exp_grad_corr_cv_no_prior(self, double_exp_data):
        """
        CV correlated fit of the double exponential with analytic gradient.
        """
        t, ordinate, true_params = double_exp_data
        result = fit_varproj(
            abscissa=t,
            ordinate=ordinate,
            linear_params=["A1", "A2"],
            model=model_double_exp,
            p0=self._p0_double,
            central_value_fit=True,
            central_value_fit_correlated=True,
            resample_fit=False,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=False)

    # ------------------------------------------------------------------
    # 6c. Consistency: grad vs finite-difference
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("correlated", [False, True], ids=["uncorr", "corr"])
    def test_single_exp_grad_matches_no_grad(self, single_exp_data, correlated):
        """
        The analytic-gradient fit and the finite-difference fit must converge
        to the same minimum within rtol=1e-5, for both uncorrelated and
        correlated varproj paths.

        We temporarily strip .grad from the model to force the
        finite-difference path, then restore it in a finally block.
        """
        t, ordinate, _ = single_exp_data

        with_grad = fit_varproj(
            abscissa=t, ordinate=ordinate,
            linear_params=["A"],
            model=model_single_exp,
            p0=self._p0_single,
            central_value_fit=True,
            central_value_fit_correlated=correlated,
            resample_fit=False,
        )

        del model_single_exp.grad
        try:
            no_grad = fit_varproj(
                abscissa=t, ordinate=ordinate,
                linear_params=["A"],
                model=model_single_exp,
                p0=self._p0_single,
                central_value_fit=True,
                central_value_fit_correlated=correlated,
                resample_fit=False,
            )
        finally:
            model_single_exp.grad = _grad_single_exp_varproj   # always restore

        for key in ("A", "E"):
            np.testing.assert_allclose(
                with_grad.params[key].mean,
                no_grad.params[key].mean,
                rtol=1e-5,
                err_msg=(
                    f"{'Correlated' if correlated else 'Uncorrelated'} varproj "
                    f"gradient and finite-difference paths converge to different "
                    f"values for '{key}'"
                ),
            )

    @pytest.mark.parametrize("correlated", [False, True], ids=["uncorr", "corr"])
    def test_single_exp_grad_prior_matches_no_grad_prior(self, single_exp_data, correlated):
        """
        With a normal prior on E, the gradient and finite-difference paths
        must converge to the same minimum within rtol=1e-5.

        This specifically tests that J[0] += prior.grad(E) is computed
        correctly in the varproj grad closure.  An incorrect sign or missing
        factor would shift the minimum.

        Note: a fresh prior dict is constructed for the no-grad branch because
        the backend may mutate the dict in-place (dropping linear-param priors).
        """
        t, ordinate, _ = single_exp_data

        with_grad = fit_varproj(
            abscissa=t, ordinate=ordinate,
            linear_params=["A"],
            model=model_single_exp,
            prior={"E": Prior(SINGLE_E, 5.0)},
            central_value_fit=True,
            central_value_fit_correlated=correlated,
            resample_fit=False,
        )

        del model_single_exp.grad
        try:
            no_grad = fit_varproj(
                abscissa=t, ordinate=ordinate,
                linear_params=["A"],
                model=model_single_exp,
                prior={"E": Prior(SINGLE_E, 5.0)},   # fresh copy
                central_value_fit=True,
                central_value_fit_correlated=correlated,
                resample_fit=False,
            )
        finally:
            model_single_exp.grad = _grad_single_exp_varproj

        for key in ("A", "E"):
            np.testing.assert_allclose(
                with_grad.params[key].mean,
                no_grad.params[key].mean,
                rtol=1e-5,
                err_msg=(
                    f"{'Correlated' if correlated else 'Uncorrelated'} varproj "
                    f"gradient+prior and finite-difference+prior paths differ "
                    f"for '{key}'"
                ),
            )

    # ------------------------------------------------------------------
    # 6d. Parallel execution
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("correlated", [False, True], ids=["uncorr", "corr"])
    def test_grad_parallel_matches_serial(self, single_exp_data, correlated):
        """
        Serial (Nproc=None) and parallel (Nproc=4) gradient fits must produce
        bit-for-bit identical rspl arrays for both correlated and uncorrelated
        varproj paths.

        Verifies that both the .grad attribute and the nonlinear-only Jacobian
        shape (1, N) survive dill serialisation across subprocess boundaries.
        We check both parameter rspl and chi² rspl for exact equality.
        """
        t, ordinate, _ = single_exp_data
        common = dict(
            abscissa=t,
            ordinate=ordinate,
            linear_params=["A"],
            model=model_single_exp,
            p0=self._p0_single,
            central_value_fit=True,
            central_value_fit_correlated=correlated,
            resample_fit=True,
            resample_fit_correlated=correlated,
        )
        serial   = fit_varproj(**common, Nproc=None)
        parallel = fit_varproj(**common, Nproc=4)

        for key in ("A", "E"):
            np.testing.assert_array_equal(
                serial.params[key].rspl,
                parallel.params[key].rspl,
                err_msg=(
                    f"rspl mismatch for '{key}' between serial and parallel "
                    f"{'correlated' if correlated else 'uncorrelated'} "
                    f"varproj gradient fits"
                ),
            )
        np.testing.assert_array_equal(
            serial.chi2.rspl,
            parallel.chi2.rspl,
            err_msg="chi2 rspl mismatch between serial and parallel varproj gradient fits",
        )

# =============================================================================
# 7. Parallel-execution tests
# =============================================================================

class TestParallelExecution:
    """
    Verify that Nproc=4 produces bit-for-bit identical results to Nproc=None.

    Rationale: the variable-projection minimiser is fully deterministic.
    Given the same per-resample inputs, it must converge to the same
    parameters and varproj dict whether it runs in a subprocess or the main
    process.

    BUG 2 note: before the fix (double _run_parallel call), parallel fits run
    each resample twice and return results from the second run.  For a
    deterministic optimiser with the same inputs the second run gives the same
    answer, so the bug does not affect correctness — but it doubles runtime.
    These tests verify correctness (serial == parallel); a runtime assertion
    would require benchmarking which is out of scope here.

    Unlike the lsqfit backend, there is no correlated + parallel restriction.
    We test both uncorrelated and correlated parallel paths.
    """

    _p0_single = {"A": SINGLE_A * 0.85, "E": SINGLE_E * 0.85}
    _p0_double = {
        "A1": DOUBLE_A1 * 0.90, "E1": DOUBLE_E1 * 0.90,
        "A2": DOUBLE_A2 * 0.90, "E2": DOUBLE_E2 * 0.90,
    }

    def _run_serial_and_parallel(
        self, abscissa, ordinate, model, p0, linear_params,
        correlated=False, limits=None
    ):
        """Run CV + resample fits serially and in parallel; return both results."""
        common = dict(
            abscissa=abscissa,
            ordinate=ordinate,
            linear_params=linear_params,
            model=model,
            p0=p0,
            central_value_fit=True,
            resample_fit=True,
            resample_fit_correlated=correlated,
            limits=limits,
        )
        return fit_varproj(**common, Nproc=None), fit_varproj(**common, Nproc=4)

    def _assert_identical(self, serial, parallel, param_keys):
        """
        Assert exact equality of CV means and all rspl entries.

        Exact equality (not allclose) is correct here: the optimiser is
        deterministic and any floating-point divergence indicates a real
        serialisation or process-boundary bug.
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

    def test_parallel_single_exp_uncorrelated(self, single_exp_data):
        """Serial and parallel must be identical for single-exp (uncorrelated)."""
        t, ordinate, _ = single_exp_data
        serial, parallel = self._run_serial_and_parallel(
            t, ordinate, model_single_exp, self._p0_single, ["A"]
        )
        self._assert_identical(serial, parallel, ["A", "E"])

    def test_parallel_single_exp_correlated(self, single_exp_data):
        """
        Serial and parallel must be identical for single-exp (correlated).

        The cost function closure captures the cov_inv numpy array; dill must
        serialise it correctly for each subprocess worker.
        """
        t, ordinate, _ = single_exp_data
        serial, parallel = self._run_serial_and_parallel(
            t, ordinate, model_single_exp, self._p0_single, ["A"], correlated=True
        )
        self._assert_identical(serial, parallel, ["A", "E"])

    def test_parallel_double_exp_uncorrelated(self, double_exp_data):
        """Serial and parallel must be identical for double-exp (uncorrelated)."""
        t, ordinate, _ = double_exp_data
        serial, parallel = self._run_serial_and_parallel(
            t, ordinate, model_double_exp, self._p0_double, ["A1", "A2"]
        )
        self._assert_identical(serial, parallel, ["A1", "A2", "E1", "E2"])

    def test_parallel_double_exp_correlated(self, double_exp_data):
        """Serial and parallel must be identical for double-exp (correlated)."""
        t, ordinate, _ = double_exp_data
        serial, parallel = self._run_serial_and_parallel(
            t, ordinate, model_double_exp, self._p0_double, ["A1", "A2"], correlated=True
        )
        self._assert_identical(serial, parallel, ["A1", "A2", "E1", "E2"])

    def test_parallel_chi2_identical(self, single_exp_data):
        """Per-resample chi² values must be identical between serial and parallel."""
        t, ordinate, _ = single_exp_data
        serial, parallel = self._run_serial_and_parallel(
            t, ordinate, model_single_exp, self._p0_single, ["A"]
        )
        np.testing.assert_array_equal(
            serial.chi2.rspl,
            parallel.chi2.rspl,
            err_msg="Per-resample chi2 differs between serial and parallel runs",
        )

    def test_parallel_with_lognormal_prior(self, single_exp_data):
        """
        Parallel fits with a log-normal prior and positivity limit must give
        bit-for-bit identical results to serial.

        This verifies that the prior dict (containing Prior objects) is
        correctly serialised by dill when distributed to subprocess workers.
        """
        t, ordinate, _ = single_exp_data
        prior = {"E": Prior(np.log(SINGLE_E), 1.0, dist="log-normal")}
        common = dict(
            abscissa=t,
            ordinate=ordinate,
            linear_params=["A"],
            model=model_single_exp,
            prior=prior,
            limits={"E": (0.0, None)},
            central_value_fit=True,
            resample_fit=True,
        )
        serial   = fit_varproj(**common, Nproc=None)
        parallel = fit_varproj(**common, Nproc=4)
        self._assert_identical(serial, parallel, ["A", "E"])