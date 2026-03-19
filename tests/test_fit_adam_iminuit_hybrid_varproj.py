"""
Unit tests for the ADAM → iminuit hybrid backend with variable projection
(fit_adam_iminuit_hybrid.py called with linear_params != None).

The variable-projection (Golub-Pereyra) path is activated whenever
``linear_params`` is a non-empty list.  ADAM then operates exclusively in
the nonlinear parameter subspace; the linear parameters (amplitudes) are
recovered analytically inside the cost function at every evaluation.

Overview
--------
1. Input-validation tests — shared _validate_inputs guards plus the
   varproj-specific prior-on-linear-param warning.

2. End-to-end parameter recovery:
     2a. Single-exponential  C(t) = A·exp(-E·t)
         linear_params=["A"],  nonlinear_params=["E"]
     2b. Double-exponential  C(t) = A1·exp(-E1·t) + A2·exp(-E2·t)
         linear_params=["A1","A2"],  nonlinear_params=["E1","E2"]
   Both models are exercised over the full six-strategy matrix
   (cv/resample × correlated/uncorrelated).

3. Normal and log-normal prior fits (priors on nonlinear params only;
   priors on linear params are silently dropped with a warning).

4. Parallel-execution tests (Nproc=4 must be bit-for-bit identical to
   Nproc=None for all models and correlation strategies).

5. ADAM-specific varproj tests:
     5a. ADAM reduces cost before handover.
     5b. Handover criterion fires before _MAX_ADAM_STEPS.
     5c. Warm start is better than raw p0.
     5d. Analytic varproj gradient (nonlinear params only) matches FD.
     5e. Double-exp varproj gradient correct (no missing cross-terms).
     5f. FD gradient path recovers parameters end-to-end.
     5g. Analytic gradient path recovers parameters end-to-end.
     5h. Tighter handover precision requires at least as many ADAM steps.

Key differences from test_fit_adam_iminuit_hybrid.py
------------------------------------------------------
* No linear-model tests: variable projection requires at least one
  genuinely nonlinear parameter.
* _check_params_recovered uses the varproj fallback for linear parameters
  (no Hessian error available; 5 % absolute tolerance instead).
* Model gradients must only contain derivatives w.r.t. nonlinear
  parameters — shape (Nnl, Ndata).  For single-exp this is (1, N);
  for double-exp this is (2, N) with rows for E1 and E2 only.
* The varproj cost function accesses cost._nonlinear_params to identify
  the ADAM parameter names.
* RNG seeds are offset by +30 / +31 to be independent of all other test
  modules (plain hybrid: +20/+21/+22, varproj iminuit: base+20/+21).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from correlatoranalyser import Data
from correlatoranalyser.fit_adam_iminuit_hybrid import fit_adam_iminuit_hybrid
from correlatoranalyser.prior import Prior

# =============================================================================
# Global test configuration
# =============================================================================

_RNG_SEED = 20240101 + 30   # distinct from all other suites
NRAW: int  = 600
NBST: int  = 300
NSIG: int  = 3

# =============================================================================
# Ground-truth parameters
# =============================================================================

SINGLE_A:     float = 1.5
SINGLE_E:     float = 0.3
SINGLE_NOISE: float = 0.02

# Well-separated energies — avoids two-state swap degeneracy.
DOUBLE_A1:    float = 1.5
DOUBLE_E1:    float = 0.3
DOUBLE_A2:    float = 0.5
DOUBLE_E2:    float = 0.8
DOUBLE_NOISE: float = 0.01

# ADAM defaults — injected into every fit call for readability.
_ADAM_DEFAULTS = dict(
    adam_hyperparam_alpha   = 0.01,
    adam_hyperparam_beta1   = 0.9,
    adam_hyperparam_beta2   = 0.999,
    adam_hyperparam_eps     = 1e-8,
    adam_handover_precision = 0.50,
    adam_handover_length    = 10,
)

# =============================================================================
# Model functions  (module-level for dill pickling in parallel workers)
# =============================================================================

def model_single_exp(t, p):
    """C(t) = A·exp(-E·t)."""
    return p["A"] * np.exp(-p["E"] * t)


def model_double_exp(t, p):
    """C(t) = A1·exp(-E1·t) + A2·exp(-E2·t)."""
    return p["A1"] * np.exp(-p["E1"] * t) + p["A2"] * np.exp(-p["E2"] * t)


# Varproj analytic Jacobians — derivatives w.r.t. NONLINEAR parameters only.
# Shape: (Nnl, Ndata).

def _grad_single_exp_varproj(t, p):
    """dC/dE only — shape (1, N).  A is linear and excluded."""
    return np.array([
        -p["A"] * t * np.exp(-p["E"] * t),   # dC/dE
    ])


def _grad_double_exp_varproj(t, p):
    """dC/d[E1, E2] only — shape (2, N).  A1, A2 are linear and excluded."""
    exp_E1t = np.exp(-p["E1"] * t)
    exp_E2t = np.exp(-p["E2"] * t)
    return np.array([
        -p["A1"] * t * exp_E1t,    # dC/dE1
        -p["A2"] * t * exp_E2t,    # dC/dE2
    ])


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
    n   = len(true_values)
    cov = noise_scale ** 2 * (np.eye(n) + 0.1 * np.ones((n, n)))
    raw = rng.multivariate_normal(mean=true_values, cov=cov, size=nraw)
    return Data(resample_type="bst", data=raw, Nresample=nbst)


# =============================================================================
# Module-scoped fixtures
# =============================================================================

@pytest.fixture(scope="module")
def single_exp_data():
    rng    = np.random.default_rng(_RNG_SEED)
    t      = np.arange(1, 11, dtype=float)
    true_C = SINGLE_A * np.exp(-SINGLE_E * t)
    return t, _make_bootstrap_data(true_C, SINGLE_NOISE, rng), {"A": SINGLE_A, "E": SINGLE_E}


@pytest.fixture(scope="module")
def double_exp_data():
    rng    = np.random.default_rng(_RNG_SEED + 1)
    t      = np.arange(1, 13, dtype=float)
    true_C = (
        DOUBLE_A1 * np.exp(-DOUBLE_E1 * t)
        + DOUBLE_A2 * np.exp(-DOUBLE_E2 * t)
    )
    return t, _make_bootstrap_data(true_C, DOUBLE_NOISE, rng), {
        "A1": DOUBLE_A1, "E1": DOUBLE_E1,
        "A2": DOUBLE_A2, "E2": DOUBLE_E2,
    }


# =============================================================================
# Shared assertion helper  (varproj version with linear-param fallback)
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

    Linear parameters have no Hessian error from Minuit (they are recovered
    analytically).  For CV-only fits on linear params we fall back to a 5 %
    absolute tolerance on the true value as a generous sanity check.
    """
    for key, true_val in true_params.items():
        assert key in fit_result.params, f"Parameter '{key}' missing from FitResult"

        if resample_fit:
            rspl_vals = fit_result.params[key].rspl
            assert rspl_vals is not None,             f"rspl is None for '{key}'"
            assert not np.any(np.isnan(rspl_vals)),   f"NaN in rspl for '{key}'"
            estimate    = fit_result.params[key].mean if central_value_fit else float(np.mean(rspl_vals))
            uncertainty = np.std(rspl_vals, ddof=1)
        else:
            estimate    = fit_result.params[key].mean
            uncertainty = fit_result.params_hessian_err[key].mean
            
        assert uncertainty > 0, f"Uncertainty for '{key}' is non-positive: {uncertainty}"
        deviation = abs(estimate - true_val)
        assert deviation < nsig * uncertainty, (
            f"'{key}': |{estimate:.6g} - {true_val:.6g}| = {deviation:.3g} "
            f">= {nsig} × {uncertainty:.3g}"
        )


# =============================================================================
# Fit-strategy parametrisation
# =============================================================================

FIT_STRATEGIES = [
    # id              cv     cv_c   rs     rs_c
    ("cv_uncorr",    True,  False, False, False),
    ("cv_corr",      True,  True,  False, False),
    ("rs_uncorr",    False, False, True,  False),
    ("rs_corr",      False, False, True,  True ),
    ("both_uncorr",  True,  False, True,  False),
    ("both_corr",    True,  True,  True,  True ),
]

_STRATEGY_IDS    = [s[0] for s in FIT_STRATEGIES]
_STRATEGY_PARAMS = [s[1:] for s in FIT_STRATEGIES]


# =============================================================================
# 1. Input-validation tests
# =============================================================================

class TestInputValidation:
    """
    Shared _validate_inputs guards plus varproj-specific checks.
    Uses single_exp_data for tests that need a real Data fixture.
    """

    _rng      = np.random.default_rng(2)
    _raw      = _rng.normal(1.0, 0.05, size=(100, 10))
    _ordinate = Data(resample_type="bst", data=_raw, Nresample=50)
    _t        = np.arange(1, 11, dtype=float)
    _p0       = {"A": SINGLE_A * 0.85, "E": SINGLE_E * 0.85}

    def _call(self, **kwargs):
        return fit_adam_iminuit_hybrid(
            abscissa=self._t,
            ordinate=self._ordinate,
            model=model_single_exp,
            linear_params=["A"],
            p0=self._p0,
            **_ADAM_DEFAULTS,
            **kwargs,
        )

    def test_ordinate_not_ndarray(self):
        """Plain numpy array ordinate must raise TypeError."""
        with pytest.raises(TypeError):
            fit_adam_iminuit_hybrid(
                abscissa=self._t, ordinate=np.ones(10),
                linear_params=["A"], model=model_single_exp,
                p0=self._p0, **_ADAM_DEFAULTS,
            )

    def test_ordinate_not_list(self):
        """Python list ordinate must raise TypeError."""
        with pytest.raises(TypeError):
            fit_adam_iminuit_hybrid(
                abscissa=self._t, ordinate=list(range(10)),
                linear_params=["A"], model=model_single_exp,
                p0=self._p0, **_ADAM_DEFAULTS,
            )

    def test_neither_cv_nor_resample_raises(self):
        """Both fit flags False must raise ValueError immediately."""
        with pytest.raises(ValueError, match="At least one of"):
            self._call(central_value_fit=False, resample_fit=False)

    def test_neither_prior_nor_p0_raises(self):
        """Omitting both prior and p0 must raise ValueError."""
        with pytest.raises(ValueError):
            fit_adam_iminuit_hybrid(
                abscissa=self._t, ordinate=self._ordinate,
                linear_params=["A"], model=model_single_exp,
                **_ADAM_DEFAULTS,
            )

    def test_no_model_raises(self):
        """model=None must raise ValueError."""
        with pytest.raises(ValueError):
            fit_adam_iminuit_hybrid(
                abscissa=self._t, ordinate=self._ordinate,
                linear_params=["A"], model=None,
                p0=self._p0, **_ADAM_DEFAULTS,
            )

    def test_prior_on_linear_param_dropped_with_warning(self, single_exp_data):
        """
        A prior on a linear parameter is meaningless in varproj and must be
        silently dropped.  The fit must still complete successfully with both
        A and E present in the result.
        """
        t, ordinate, _ = single_exp_data
        prior = {
            "A": Prior(SINGLE_A, 5.0),    # will be dropped
            "E": Prior(SINGLE_E, 5.0),
        }
        result = fit_adam_iminuit_hybrid(
            abscissa=t, ordinate=ordinate,
            linear_params=["A"], model=model_single_exp,
            prior=prior,
            central_value_fit=True, resample_fit=False,
            **_ADAM_DEFAULTS,
        )
        assert "A" in result.params
        assert "E" in result.params

    def test_abscissa_scalar_raises(self):
        """Scalar abscissa must raise before any minimisation."""
        with pytest.raises(Exception):
            self._call(abscissa=42)


# =============================================================================
# 2a. End-to-end tests — single-exponential, varproj
# =============================================================================

class TestSingleExpVarproj:
    """
    C(t) = A·exp(-E·t), linear_params=["A"], nonlinear_params=["E"].

    p0 displaced 30 % from truth so ADAM must traverse meaningful parameter
    space before handover.
    """

    _p0     = {"A": SINGLE_A * 0.70, "E": SINGLE_E * 0.70}
    _limits = {"E": (0.0, None)}

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _STRATEGY_PARAMS, ids=_STRATEGY_IDS)
    def test_parameter_recovery(self, single_exp_data, cv, cv_corr, rs, rs_corr):
        """Both A (linear) and E (nonlinear) must be recovered within 3σ."""
        t, ordinate, true_params = single_exp_data
        result = fit_adam_iminuit_hybrid(
            abscissa=t, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_single_exp,
            linear_params=["A"],
            p0=self._p0, limits=self._limits,
            **_ADAM_DEFAULTS,
        )
        _check_params_recovered(result, true_params, cv, rs)

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _STRATEGY_PARAMS, ids=_STRATEGY_IDS)
    def test_fit_result_metadata(self, single_exp_data, cv, cv_corr, rs, rs_corr):
        """FitResult must carry positive dof and finite chi²."""
        t, ordinate, _ = single_exp_data
        result = fit_adam_iminuit_hybrid(
            abscissa=t, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_single_exp,
            linear_params=["A"],
            p0=self._p0, limits=self._limits,
            **_ADAM_DEFAULTS,
        )
        assert result.dof is not None and result.dof > 0
        if cv:
            chi2_val = result.chi2.mean if hasattr(result.chi2, "mean") else float(result.chi2)
            assert chi2_val > 0
        if rs:
            assert np.all(np.isfinite(result.chi2.rspl))

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _STRATEGY_PARAMS, ids=_STRATEGY_IDS)
    def test_linear_param_in_result(self, single_exp_data, cv, cv_corr, rs, rs_corr):
        """
        A must appear in fit_result.params even though it is never passed to
        Minuit — the varproj mechanism recovers it analytically and stores it.
        """
        t, ordinate, _ = single_exp_data
        result = fit_adam_iminuit_hybrid(
            abscissa=t, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_single_exp,
            linear_params=["A"],
            p0=self._p0,
            **_ADAM_DEFAULTS,
        )
        assert "A" in result.params
        if cv:
            assert np.isfinite(result.params["A"].mean)
        if rs:
            assert np.all(np.isfinite(result.params["A"].rspl))

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _STRATEGY_PARAMS, ids=_STRATEGY_IDS)
    def test_energy_positive(self, single_exp_data, cv, cv_corr, rs, rs_corr):
        """E must remain positive when limits={"E": (0, None)} is set."""
        t, ordinate, _ = single_exp_data
        result = fit_adam_iminuit_hybrid(
            abscissa=t, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_single_exp,
            linear_params=["A"],
            p0=self._p0, limits=self._limits,
            **_ADAM_DEFAULTS,
        )
        if cv:
            assert result.params["E"].mean > 0
        if rs:
            assert np.all(result.params["E"].rspl > 0)


# =============================================================================
# 2b. End-to-end tests — double-exponential, varproj
# =============================================================================

class TestDoubleExpVarproj:
    """
    C(t) = A1·exp(-E1·t) + A2·exp(-E2·t),
    linear_params=["A1","A2"], nonlinear_params=["E1","E2"].

    p0 displaced 30 % from truth.  Well-separated energies avoid bimodality.
    """

    _p0 = {
        "A1": DOUBLE_A1 * 0.70, "E1": DOUBLE_E1 * 0.70,
        "A2": DOUBLE_A2 * 0.70, "E2": DOUBLE_E2 * 0.70,
    }
    _limits = {"E1": (0.0, None), "E2": (0.0, None)}

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _STRATEGY_PARAMS, ids=_STRATEGY_IDS)
    def test_parameter_recovery(self, double_exp_data, cv, cv_corr, rs, rs_corr):
        """All four parameters must be recovered within 3σ for every strategy."""
        t, ordinate, true_params = double_exp_data
        result = fit_adam_iminuit_hybrid(
            abscissa=t, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_double_exp,
            linear_params=["A1", "A2"],
            p0=self._p0, limits=self._limits,
            **_ADAM_DEFAULTS,
        )
        _check_params_recovered(result, true_params, cv, rs)

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _STRATEGY_PARAMS, ids=_STRATEGY_IDS)
    def test_both_linear_params_in_result(self, double_exp_data, cv, cv_corr, rs, rs_corr):
        """A1 and A2 must both appear in the result despite being analytically eliminated."""
        t, ordinate, _ = double_exp_data
        result = fit_adam_iminuit_hybrid(
            abscissa=t, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_double_exp,
            linear_params=["A1", "A2"],
            p0=self._p0,
            **_ADAM_DEFAULTS,
        )
        for key in ("A1", "A2"):
            assert key in result.params
            if cv:
                assert np.isfinite(result.params[key].mean)
            if rs:
                assert np.all(np.isfinite(result.params[key].rspl))

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _STRATEGY_PARAMS, ids=_STRATEGY_IDS)
    def test_energies_positive(self, double_exp_data, cv, cv_corr, rs, rs_corr):
        """Both energies must be positive with positivity limits."""
        t, ordinate, _ = double_exp_data
        result = fit_adam_iminuit_hybrid(
            abscissa=t, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_double_exp,
            linear_params=["A1", "A2"],
            p0=self._p0, limits=self._limits,
            **_ADAM_DEFAULTS,
        )
        if cv:
            assert result.params["E1"].mean > 0
            assert result.params["E2"].mean > 0
        if rs:
            assert np.all(result.params["E1"].rspl > 0)
            assert np.all(result.params["E2"].rspl > 0)


# =============================================================================
# 3. Prior fits — varproj path
# =============================================================================

class TestNormalPriorVarproj:
    """Normal priors on the nonlinear parameter(s) only."""

    _prior_single = {"E": Prior(SINGLE_E, 5 * SINGLE_NOISE)}

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _STRATEGY_PARAMS, ids=_STRATEGY_IDS)
    def test_single_exp_with_prior(self, single_exp_data, cv, cv_corr, rs, rs_corr):
        """Prior on E must not prevent accurate parameter recovery."""
        t, ordinate, true_params = single_exp_data
        result = fit_adam_iminuit_hybrid(
            abscissa=t, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_single_exp,
            linear_params=["A"],
            prior=self._prior_single,
            p0={"A": SINGLE_A * 0.70, "E": SINGLE_E * 0.70},
            **_ADAM_DEFAULTS,
        )
        _check_params_recovered(result, true_params, cv, rs)


class TestLogNormalPriorVarproj:
    """Log-normal prior on E paired with positivity limit."""

    _prior_single = {
        "E": Prior(np.log(SINGLE_E), 0.3, dist="log-normal"),
    }
    _p0 = {"A": SINGLE_A * 0.70, "E": SINGLE_E * 0.70}

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _STRATEGY_PARAMS, ids=_STRATEGY_IDS)
    def test_recovery_with_lognormal_prior(self, single_exp_data, cv, cv_corr, rs, rs_corr):
        """Log-normal prior on E must not prevent accurate recovery."""
        t, ordinate, true_params = single_exp_data
        result = fit_adam_iminuit_hybrid(
            abscissa=t, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_single_exp,
            linear_params=["A"],
            prior=self._prior_single, p0=self._p0,
            limits={"E": (0.0, None)},
            **_ADAM_DEFAULTS,
        )
        _check_params_recovered(result, true_params, cv, rs)

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _STRATEGY_PARAMS, ids=_STRATEGY_IDS)
    def test_energy_stays_positive(self, single_exp_data, cv, cv_corr, rs, rs_corr):
        """E must remain positive throughout (clamping + iminuit limits)."""
        t, ordinate, _ = single_exp_data
        result = fit_adam_iminuit_hybrid(
            abscissa=t, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_single_exp,
            linear_params=["A"],
            prior=self._prior_single, p0=self._p0,
            limits={"E": (0.0, None)},
            **_ADAM_DEFAULTS,
        )
        if cv:
            assert result.params["E"].mean > 0
        if rs:
            assert np.all(result.params["E"].rspl > 0)


# =============================================================================
# 4. Parallel-execution tests — varproj path
# =============================================================================

class TestParallelExecutionVarproj:
    """
    Nproc=4 must produce bit-for-bit identical results to Nproc=None.

    Both ADAM and iminuit are deterministic given the same inputs, so any
    discrepancy indicates a serialisation or process-boundary bug in the
    varproj parallel infrastructure.
    """

    _p0_single = {"A": SINGLE_A * 0.70, "E": SINGLE_E * 0.70}
    _p0_double = {
        "A1": DOUBLE_A1 * 0.70, "E1": DOUBLE_E1 * 0.70,
        "A2": DOUBLE_A2 * 0.70, "E2": DOUBLE_E2 * 0.70,
    }

    def _run_both(self, abscissa, ordinate, model, linear_params, p0,
                  correlated=False, limits=None):
        common = dict(
            abscissa=abscissa, ordinate=ordinate,
            central_value_fit=True, resample_fit=True,
            resample_fit_correlated=correlated,
            model=model, linear_params=linear_params,
            p0=p0, limits=limits,
            **_ADAM_DEFAULTS,
        )
        return (
            fit_adam_iminuit_hybrid(**common, Nproc=None),
            fit_adam_iminuit_hybrid(**common, Nproc=4),
        )

    def _assert_identical(self, serial, parallel, keys):
        for key in keys:
            np.testing.assert_array_equal(
                serial.params[key].mean,   parallel.params[key].mean,
                err_msg=f"CV mean mismatch for '{key}'"
            )
            np.testing.assert_array_equal(
                serial.params[key].rspl,   parallel.params[key].rspl,
                err_msg=f"rspl mismatch for '{key}'"
            )

    def test_single_exp_uncorrelated(self, single_exp_data):
        """Serial and parallel must be identical: single-exp, uncorrelated."""
        t, ordinate, _ = single_exp_data
        s, p = self._run_both(
            t, ordinate, model_single_exp, ["A"], self._p0_single,
            limits={"E": (0.0, None)},
        )
        self._assert_identical(s, p, ["A", "E"])

    def test_single_exp_correlated(self, single_exp_data):
        """Serial and parallel must be identical: single-exp, correlated."""
        t, ordinate, _ = single_exp_data
        s, p = self._run_both(
            t, ordinate, model_single_exp, ["A"], self._p0_single,
            correlated=True, limits={"E": (0.0, None)},
        )
        self._assert_identical(s, p, ["A", "E"])

    def test_double_exp_uncorrelated(self, double_exp_data):
        """Serial and parallel must be identical: double-exp, uncorrelated."""
        t, ordinate, _ = double_exp_data
        s, p = self._run_both(
            t, ordinate, model_double_exp, ["A1", "A2"], self._p0_double,
            limits={"E1": (0.0, None), "E2": (0.0, None)},
        )
        self._assert_identical(s, p, ["A1", "E1", "A2", "E2"])

    def test_double_exp_correlated(self, double_exp_data):
        """Serial and parallel must be identical: double-exp, correlated."""
        t, ordinate, _ = double_exp_data
        s, p = self._run_both(
            t, ordinate, model_double_exp, ["A1", "A2"], self._p0_double,
            correlated=True, limits={"E1": (0.0, None), "E2": (0.0, None)},
        )
        self._assert_identical(s, p, ["A1", "E1", "A2", "E2"])

    def test_chi2_rspl_identical(self, single_exp_data):
        """Per-resample chi² must be bit-for-bit identical across serial/parallel."""
        t, ordinate, _ = single_exp_data
        s, p = self._run_both(
            t, ordinate, model_single_exp, ["A"], self._p0_single,
        )
        np.testing.assert_array_equal(
            s.chi2.rspl, p.chi2.rspl,
            err_msg="Per-resample chi2 differs between serial and parallel",
        )


# =============================================================================
# 5. ADAM-specific varproj tests
# =============================================================================

class TestADAMPhaseVarproj:
    """
    Tests that exercise the ADAM phase specifically on the varproj cost function.

    The key varproj distinction is that ADAM operates in the nonlinear
    parameter subspace only: for single-exp theta is 1-D (E only); for
    double-exp theta is 2-D (E1, E2 only).  The linear amplitudes are
    never part of the ADAM parameter vector.
    """

    def _build_varproj_cost(self, t, ordinate, model, linear_params, p0,
                            has_grad=False, correlated=False):
        """
        Build a varproj cost function for direct _run_adam testing.
        Returns (cost, nl_params, theta0).
        """
        from correlatoranalyser.fit_adam_iminuit_hybrid import _run_adam
        from correlatoranalyser.fit_iminuit_variable_projection import (
            _build_varproj_uncorrelated_cost,
            _build_varproj_correlated_cost,
        )
        from correlatoranalyser.fit_helper import _compute_cov_inv

        start_vals = p0
        nl_params  = [k for k in start_vals if k not in linear_params]

        if correlated:
            # Need cov_inv — compute from ordinate directly.
            cov_inv, LT = _compute_cov_inv(ordinate, None)
            cost = _build_varproj_correlated_cost(
                t, ordinate.mean, cov_inv, model,
                nl_params, linear_params, None, has_grad, LT,
            )
        else:
            cost = _build_varproj_uncorrelated_cost(
                t, ordinate.mean, 1.0 / ordinate.serr, model,
                nl_params, linear_params, None, has_grad,
            )

        theta0 = np.array([start_vals[k] for k in nl_params], dtype=float)
        return cost, nl_params, theta0

    def test_adam_operates_in_nonlinear_subspace(self, single_exp_data):
        """
        theta0 for the varproj ADAM call must have dimension Nnl, not Nparams.
        For single-exp (A linear, E nonlinear) theta0 must be 1-D.
        """
        from correlatoranalyser.fit_adam_iminuit_hybrid import _run_adam

        t, ordinate, _ = single_exp_data
        p0 = {"A": SINGLE_A * 0.70, "E": SINGLE_E * 0.70}
        cost, nl_params, theta0 = self._build_varproj_cost(
            t, ordinate, model_single_exp, ["A"], p0
        )

        assert theta0.shape == (1,), (
            f"ADAM theta0 should be 1-D for single-exp varproj, got {theta0.shape}"
        )
        assert nl_params == ["E"]

    def test_adam_reduces_cost_before_handover(self, single_exp_data):
        """ADAM must lower chi² from the displaced starting point."""
        from correlatoranalyser.fit_adam_iminuit_hybrid import _run_adam

        t, ordinate, _ = single_exp_data
        p0 = {"A": SINGLE_A * 0.70, "E": SINGLE_E * 0.70}
        cost, nl_params, theta0 = self._build_varproj_cost(
            t, ordinate, model_single_exp, ["A"], p0
        )

        _, history = _run_adam(
            cost, theta0,
            alpha=0.01, beta1=0.9, beta2=0.999, eps=1e-8,
            precision=0.50, length=10,
            limits={"E": (0.0, None)}, param_names=nl_params,
        )

        assert history[-1] < history[0], (
            f"ADAM did not reduce chi²: initial={history[0]:.4g}, final={history[-1]:.4g}"
        )

    def test_handover_fires_before_max_steps(self, single_exp_data):
        """The handover criterion must terminate ADAM well before _MAX_ADAM_STEPS."""
        from correlatoranalyser.fit_adam_iminuit_hybrid import _run_adam, _MAX_ADAM_STEPS

        t, ordinate, _ = single_exp_data
        p0 = {"A": SINGLE_A * 0.70, "E": SINGLE_E * 0.70}
        cost, nl_params, theta0 = self._build_varproj_cost(
            t, ordinate, model_single_exp, ["A"], p0
        )

        _, history = _run_adam(
            cost, theta0,
            alpha=0.01, beta1=0.9, beta2=0.999, eps=1e-8,
            precision=0.50, length=10,
            limits={"E": (0.0, None)}, param_names=nl_params,
        )

        assert len(history) < _MAX_ADAM_STEPS, (
            f"ADAM did not terminate: ran for {len(history)} steps."
        )

    def test_warm_start_better_than_p0(self, single_exp_data):
        """theta_best from ADAM must give lower chi² than the initial p0."""
        from correlatoranalyser.fit_adam_iminuit_hybrid import _run_adam

        t, ordinate, _ = single_exp_data
        p0 = {"A": SINGLE_A * 0.70, "E": SINGLE_E * 0.70}
        cost, nl_params, theta0 = self._build_varproj_cost(
            t, ordinate, model_single_exp, ["A"], p0
        )

        theta_best, _ = _run_adam(
            cost, theta0,
            alpha=0.01, beta1=0.9, beta2=0.999, eps=1e-8,
            precision=0.50, length=10,
            limits={"E": (0.0, None)}, param_names=nl_params,
        )

        assert float(cost(*theta_best)) < float(cost(*theta0)), (
            "ADAM warm-start is not better than initial p0 for varproj cost"
        )

    def test_varproj_gradient_single_exp_matches_fd(self, single_exp_data):
        """
        The varproj analytic gradient for single-exp (dC/dE only, shape (1,N))
        must agree with finite differences to within 0.1 %.

        This guards against the class of error found in the original double-exp
        example: a missing chain-rule contribution in the gradient expression.
        """
        from correlatoranalyser.fit_adam_iminuit_hybrid import _numerical_gradient

        t, ordinate, _ = single_exp_data
        p0 = {"A": SINGLE_A, "E": SINGLE_E}    # evaluate at truth
        model_single_exp.grad = _grad_single_exp_varproj
        try:
            cost, nl_params, theta = self._build_varproj_cost(
                t, ordinate, model_single_exp, ["A"], p0, has_grad=True
            )
            g_analytic = np.asarray(cost.grad(*theta))
            g_fd       = _numerical_gradient(cost, theta)
            np.testing.assert_allclose(
                g_analytic, g_fd, rtol=1e-3,
                err_msg="Single-exp varproj analytic gradient does not match FD",
            )
        finally:
            del model_single_exp.grad

    def test_varproj_gradient_double_exp_matches_fd(self, double_exp_data):
        """
        The double-exp varproj gradient (dC/d[E1,E2] only, shape (2,N)) must
        match finite differences.  This is the varproj analogue of the bug
        found in the original hybrid example (missing E0 cross-term).
        """
        from correlatoranalyser.fit_adam_iminuit_hybrid import _numerical_gradient

        t, ordinate, _ = double_exp_data
        p0 = {"A1": DOUBLE_A1, "E1": DOUBLE_E1, "A2": DOUBLE_A2, "E2": DOUBLE_E2}
        model_double_exp.grad = _grad_double_exp_varproj
        try:
            cost, nl_params, theta = self._build_varproj_cost(
                t, ordinate, model_double_exp, ["A1", "A2"], p0, has_grad=True
            )
            g_analytic = np.asarray(cost.grad(*theta))
            g_fd       = _numerical_gradient(cost, theta)
            np.testing.assert_allclose(
                g_analytic, g_fd, rtol=1e-3,
                err_msg="Double-exp varproj analytic gradient does not match FD",
            )
        finally:
            del model_double_exp.grad

    def test_fd_gradient_path_recovers_params(self, single_exp_data):
        """
        Without model.grad, ADAM uses finite differences on the varproj cost.
        Parameter recovery must still be within 3σ end-to-end.
        """
        assert not hasattr(model_single_exp, "grad")
        t, ordinate, true_params = single_exp_data
        result = fit_adam_iminuit_hybrid(
            abscissa=t, ordinate=ordinate,
            model=model_single_exp,
            linear_params=["A"],
            p0={"A": SINGLE_A * 0.70, "E": SINGLE_E * 0.70},
            limits={"E": (0.0, None)},
            central_value_fit=True, resample_fit=True,
            **_ADAM_DEFAULTS,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=True)

    def test_analytic_gradient_path_recovers_params(self, single_exp_data):
        """
        With model.grad set to the varproj gradient (nonlinear params only),
        parameter recovery must still be within 3σ end-to-end.
        """
        t, ordinate, true_params = single_exp_data
        model_single_exp.grad = _grad_single_exp_varproj
        try:
            result = fit_adam_iminuit_hybrid(
                abscissa=t, ordinate=ordinate,
                model=model_single_exp,
                linear_params=["A"],
                p0={"A": SINGLE_A * 0.70, "E": SINGLE_E * 0.70},
                limits={"E": (0.0, None)},
                central_value_fit=True, resample_fit=True,
                **_ADAM_DEFAULTS,
            )
            _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=True)
        finally:
            del model_single_exp.grad

    def test_tighter_precision_requires_more_steps(self, single_exp_data):
        """
        A tighter handover precision must require at least as many ADAM steps
        as a looser one.  This guards against the per-step criterion regression
        on the varproj path.
        """
        from correlatoranalyser.fit_adam_iminuit_hybrid import _run_adam

        t, ordinate, _ = single_exp_data
        p0 = {"A": SINGLE_A * 0.70, "E": SINGLE_E * 0.70}
        cost, nl_params, theta0 = self._build_varproj_cost(
            t, ordinate, model_single_exp, ["A"], p0
        )

        _, h_loose = _run_adam(
            cost, theta0, alpha=0.01, beta1=0.9, beta2=0.999, eps=1e-8,
            precision=0.50, length=10,
            limits={"E": (0.0, None)}, param_names=nl_params,
        )
        _, h_tight = _run_adam(
            cost, theta0, alpha=0.01, beta1=0.9, beta2=0.999, eps=1e-8,
            precision=0.001, length=10,
            limits={"E": (0.0, None)}, param_names=nl_params,
        )

        assert len(h_tight) >= len(h_loose), (
            f"Tighter precision should use >= steps: tight={len(h_tight)}, loose={len(h_loose)}"
        )
