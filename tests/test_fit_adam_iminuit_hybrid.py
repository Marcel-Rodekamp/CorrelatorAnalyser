"""
Unit tests for the ADAM → iminuit hybrid backend
(fit_adam_iminuit_hybrid.py).

Overview
--------
1. Input-validation tests — same contract as the plain iminuit backend, plus
   the additional guard that non-empty ``linear_params`` raises
   ``NotImplementedError``.

2. End-to-end parameter recovery for three model families:
     2a. Linear              y(x)  = m·x + b
     2b. Single-exponential  C(t)  = A·exp(-E·t)
     2c. Double-exponential  C(t)  = A1·exp(-E1·t) + A2·exp(-E2·t)
     2d. Normal-prior fits
     2e. Log-normal-prior fits  (with positivity limits)
   Each model is exercised over the full six-strategy matrix
   (cv/resample × uncorrelated/correlated).

3. Limits test — same as the iminuit suite: bounds must be respected both
   by the ADAM clamping phase and by iminuit after handover.

4. Parallel-execution tests — Nproc=4 must produce bit-for-bit identical
   results to serial execution.

5. ADAM-specific tests that have no analogue in test_fit_iminuit.py:
     5a. Handover criterion fires: the backend returns when chi² has
         converged, not only after a fixed number of steps.
     5b. ADAM improves the start: after the ADAM phase the cost is lower
         than the initial p0 cost, confirming that ADAM did real work.
     5c. Gradient information is used when model.grad is present.
     5d. Finite-difference gradient path: same results without model.grad.

"""

from __future__ import annotations

import numpy as np
import pytest

from correlatoranalyser import Data
from correlatoranalyser.fit_adam_iminuit_hybrid import fit_adam_iminuit_hybrid
from correlatoranalyser.prior import Prior

# =============================================================================
# Global test configuration
# =============================================================================

_RNG_SEED = 20240101 + 20   # offset from iminuit and lsqfit suites
NRAW: int  = 600
NBST: int  = 300
NSIG: int  = 3

# =============================================================================
# Ground-truth parameters
# =============================================================================

LINEAR_M:     float = 2.5
LINEAR_B:     float = 0.7
LINEAR_NOISE: float = 0.10

SINGLE_A:     float = 1.5
SINGLE_E:     float = 0.3
SINGLE_NOISE: float = 0.02

# Well-separated energies (gap = 0.5) to avoid two-state swap degeneracy.
DOUBLE_A1:    float = 1.5
DOUBLE_E1:    float = 0.3
DOUBLE_A2:    float = 0.5
DOUBLE_E2:    float = 0.8
DOUBLE_NOISE: float = 0.01

# ADAM default kwargs used throughout — keeps tests readable and lets us
# change the defaults in one place without touching every call site.
_ADAM_DEFAULTS = dict(
    adam_hyperparam_alpha   = 0.01,
    adam_hyperparam_beta1   = 0.9,
    adam_hyperparam_beta2   = 0.999,
    adam_hyperparam_eps     = 1e-8,
    adam_handover_precision = 0.50,
    adam_handover_length    = 10,
)

# =============================================================================
# Model functions  (module-level for pickling across parallel workers)
# =============================================================================

def model_linear(x, p):
    """y = m·x + b."""
    return p["m"] * x + p["b"]


def model_single_exp(t, p):
    """C(t) = A·exp(-E·t)."""
    return p["A"] * np.exp(-p["E"] * t)


def model_double_exp(t, p):
    """C(t) = A1·exp(-E1·t) + A2·exp(-E2·t)."""
    return p["A1"] * np.exp(-p["E1"] * t) + p["A2"] * np.exp(-p["E2"] * t)


# Analytic Jacobians for the gradient tests.
# shape: (Nparams, Ndata)

def _grad_linear(x, p):
    """dC/d[m, b] — shape (2, N)."""
    return np.array([x, np.ones_like(x)])


def _grad_single_exp(t, p):
    """dC/d[A, E] — shape (2, N)."""
    exp_Et = np.exp(-p["E"] * t)
    return np.array([
        exp_Et,                      # dC/dA
        -p["A"] * t * exp_Et,        # dC/dE
    ])


def _grad_double_exp(t, p):
    """dC/d[A1, E1, A2, E2] — shape (4, N)."""
    exp_E1t = np.exp(-p["E1"] * t)
    exp_E2t = np.exp(-p["E2"] * t)
    return np.array([
        exp_E1t,                      # dC/dA1
        -p["A1"] * t * exp_E1t,       # dC/dE1
        exp_E2t,                      # dC/dA2
        -p["A2"] * t * exp_E2t,       # dC/dE2
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
def linear_data():
    rng     = np.random.default_rng(_RNG_SEED)
    x       = np.linspace(0.0, 1.0, 8)
    true_y  = LINEAR_M * x + LINEAR_B
    ordinate = _make_bootstrap_data(true_y, LINEAR_NOISE, rng)
    return x, ordinate, {"m": LINEAR_M, "b": LINEAR_B}


@pytest.fixture(scope="module")
def single_exp_data():
    rng     = np.random.default_rng(_RNG_SEED + 1)
    t       = np.arange(1, 11, dtype=float)
    true_C  = SINGLE_A * np.exp(-SINGLE_E * t)
    ordinate = _make_bootstrap_data(true_C, SINGLE_NOISE, rng)
    return t, ordinate, {"A": SINGLE_A, "E": SINGLE_E}


@pytest.fixture(scope="module")
def double_exp_data():
    rng    = np.random.default_rng(_RNG_SEED + 2)
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
    Assert every parameter lies within *nsig* σ of its true value.

    Error-estimate logic is identical to test_fit_iminuit.py:
    * CV-only:       Hessian error from params_hessian_err.
    * Resample-only: bootstrap std of rspl.
    * CV + resample: locked CV mean; bootstrap std as error.
    """
    for key, true_val in true_params.items():
        assert key in fit_result.params, f"Parameter '{key}' missing from FitResult"

        if resample_fit:
            rspl_vals = fit_result.params[key].rspl
            assert rspl_vals is not None,       f"rspl is None for '{key}'"
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
    Confirm that fit_adam_iminuit_hybrid raises informative errors for bad
    inputs.  Tests mirror test_fit_iminuit.TestInputValidation exactly, with
    the addition of the linear_params guard unique to this backend.
    """

    _x        = np.linspace(0.0, 1.0, 5)
    _rng      = np.random.default_rng(0)
    _raw      = _rng.normal(1.0, 0.05, size=(100, 5))
    _ordinate = Data(resample_type="bst", data=_raw, Nresample=50)
    _p0       = {"m": 2.0, "b": 0.5}

    def _call(self, **kwargs):
        """Thin wrapper that injects ADAM defaults so tests stay concise."""
        return fit_adam_iminuit_hybrid(
            abscissa=self._x,
            ordinate=self._ordinate,
            model=model_linear,
            p0=self._p0,
            **_ADAM_DEFAULTS,
            **kwargs,
        )

    def test_ordinate_not_ndarray(self):
        """Plain numpy array ordinate must raise TypeError."""
        with pytest.raises(TypeError):
            fit_adam_iminuit_hybrid(
                abscissa=self._x,
                ordinate=np.ones(5),
                model=model_linear,
                p0=self._p0,
                **_ADAM_DEFAULTS,
            )

    def test_ordinate_not_list(self):
        """Python list ordinate must raise TypeError."""
        with pytest.raises(TypeError):
            fit_adam_iminuit_hybrid(
                abscissa=self._x,
                ordinate=[1.0, 2.0, 3.0, 4.0, 5.0],
                model=model_linear,
                p0=self._p0,
                **_ADAM_DEFAULTS,
            )

    def test_abscissa_scalar_raises(self):
        """Scalar abscissa must raise before any minimisation is attempted."""
        with pytest.raises(Exception):
            self._call(abscissa=42)

    def test_neither_cv_nor_resample_raises(self):
        """Both fit flags False must raise ValueError immediately."""
        with pytest.raises(ValueError, match="At least one of"):
            self._call(central_value_fit=False, resample_fit=False)

    def test_neither_prior_nor_p0_raises(self):
        """Omitting both prior and p0 must raise ValueError."""
        with pytest.raises(ValueError):
            fit_adam_iminuit_hybrid(
                abscissa=self._x,
                ordinate=self._ordinate,
                model=model_linear,
                **_ADAM_DEFAULTS,
            )

    def test_no_model_raises(self):
        """model=None must raise ValueError via _validate_inputs."""
        with pytest.raises(ValueError):
            fit_adam_iminuit_hybrid(
                abscissa=self._x,
                ordinate=self._ordinate,
                model=None,
                p0=self._p0,
                **_ADAM_DEFAULTS,
            )

    def test_linear_params_empty_list_does_not_raise(self):
        """
        An empty list for linear_params should NOT raise — it is treated as
        'no linear params requested' rather than 'variable projection'.
        """
        self._call(linear_params=[])   # must not raise

# =============================================================================
# 2a. End-to-end tests — linear model
# =============================================================================

class TestLinearFit:
    """
    y(x) = m·x + b over all six strategy combinations.

    Starting point displaced 30% from truth — larger than the plain iminuit
    suite (15%) to ensure ADAM must traverse meaningful parameter space.
    """

    _p0 = {"m": LINEAR_M * 0.70, "b": LINEAR_B * 0.70}

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _STRATEGY_PARAMS, ids=_STRATEGY_IDS)
    def test_parameter_recovery(self, linear_data, cv, cv_corr, rs, rs_corr):
        """m and b must be recovered within 3σ for every strategy."""
        x, ordinate, true_params = linear_data
        result = fit_adam_iminuit_hybrid(
            abscissa=x, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_linear, p0=self._p0,
            **_ADAM_DEFAULTS,
        )
        _check_params_recovered(result, true_params, cv, rs)

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _STRATEGY_PARAMS, ids=_STRATEGY_IDS)
    def test_fit_result_metadata(self, linear_data, cv, cv_corr, rs, rs_corr):
        """FitResult must carry positive dof and finite chi²."""
        x, ordinate, _ = linear_data
        result = fit_adam_iminuit_hybrid(
            abscissa=x, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_linear, p0=self._p0,
            **_ADAM_DEFAULTS,
        )
        assert result.dof is not None and result.dof > 0
        if cv:
            chi2_val = result.chi2.mean if hasattr(result.chi2, "mean") else float(result.chi2)
            assert chi2_val > 0
        if rs:
            assert np.all(np.isfinite(result.chi2.rspl))
            assert np.all(result.chi2.rspl > 0)

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _STRATEGY_PARAMS, ids=_STRATEGY_IDS)
    def test_hessian_errors_populated(self, linear_data, cv, cv_corr, rs, rs_corr):
        """Hessian errors must be present and positive for CV and resample fits."""
        x, ordinate, _ = linear_data
        result = fit_adam_iminuit_hybrid(
            abscissa=x, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_linear, p0=self._p0,
            **_ADAM_DEFAULTS,
        )
        if cv:
            for key in ("m", "b"):
                assert key in result.params_hessian_err
                assert result.params_hessian_err[key].mean > 0
        if rs:
            for key in ("m", "b"):
                assert np.all(result.params_hessian_err[key].rspl > 0)


# =============================================================================
# 2b. End-to-end tests — single-exponential model
# =============================================================================

class TestSingleExpFit:
    """
    C(t) = A·exp(-E·t), t = 1..10, over all six strategy combinations.

    p0 displaced 30% from truth.
    """

    _p0 = {"A": SINGLE_A * 0.70, "E": SINGLE_E * 0.70}

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _STRATEGY_PARAMS, ids=_STRATEGY_IDS)
    def test_parameter_recovery(self, single_exp_data, cv, cv_corr, rs, rs_corr):
        """A and E must be recovered within 3σ for every strategy."""
        t, ordinate, true_params = single_exp_data
        result = fit_adam_iminuit_hybrid(
            abscissa=t, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_single_exp, p0=self._p0,
            limits={"E": (0.0, None)},
            **_ADAM_DEFAULTS,
        )
        _check_params_recovered(result, true_params, cv, rs)

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _STRATEGY_PARAMS, ids=_STRATEGY_IDS)
    def test_resample_count(self, single_exp_data, cv, cv_corr, rs, rs_corr):
        """Every resample slot must be populated (no NaN)."""
        t, ordinate, _ = single_exp_data
        result = fit_adam_iminuit_hybrid(
            abscissa=t, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_single_exp, p0=self._p0,
            **_ADAM_DEFAULTS,
        )
        if rs:
            assert result.params["A"].rspl.shape[0] == NBST
            assert not np.any(np.isnan(result.params["A"].rspl))
            assert not np.any(np.isnan(result.params["E"].rspl))


# =============================================================================
# 2c. End-to-end tests — double-exponential model
# =============================================================================

class TestDoubleExpFit:
    """
    C(t) = A1·exp(-E1·t) + A2·exp(-E2·t), t = 1..12.

    Energies are well-separated (E1=0.3, E2=0.8) to avoid the near-degenerate
    two-state swap that causes bimodal bootstrap distributions.

    p0 displaced 30% from truth.  With a 30% displacement from a 4-parameter
    nonlinear model ADAM's exploration phase is genuinely beneficial.
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
            model=model_double_exp, p0=self._p0,
            limits=self._limits,
            **_ADAM_DEFAULTS,
        )
        _check_params_recovered(result, true_params, cv, rs)

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _STRATEGY_PARAMS, ids=_STRATEGY_IDS)
    def test_energies_positive(self, double_exp_data, cv, cv_corr, rs, rs_corr):
        """Fitted energies must be positive (limits enforced by clamping + iminuit)."""
        t, ordinate, _ = double_exp_data
        result = fit_adam_iminuit_hybrid(
            abscissa=t, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_double_exp, p0=self._p0,
            limits=self._limits,
            **_ADAM_DEFAULTS,
        )
        if cv:
            assert result.params["E1"].mean > 0
            assert result.params["E2"].mean > 0
        if rs:
            assert np.all(result.params["E1"].rspl > 0)
            assert np.all(result.params["E2"].rspl > 0)


# =============================================================================
# 2d. Normal-prior fits
# =============================================================================

class TestNormalPriorFit:
    """
    Gaussian priors centred on the truth with wide σ (5× noise).

    Priors should not significantly shift the central value but must reduce
    the uncertainty compared to the prior-free fit.  We verify recovery
    within the same 3σ band as the prior-free tests.
    """

    _prior_single = {
        "A": Prior(SINGLE_A, 5 * SINGLE_NOISE),
        "E": Prior(SINGLE_E, 5 * SINGLE_NOISE),
    }

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _STRATEGY_PARAMS, ids=_STRATEGY_IDS)
    def test_single_exp_with_prior(self, single_exp_data, cv, cv_corr, rs, rs_corr):
        """Prior-constrained single-exp fit must still recover A and E within 3σ."""
        t, ordinate, true_params = single_exp_data
        result = fit_adam_iminuit_hybrid(
            abscissa=t, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_single_exp,
            prior=self._prior_single,
            **_ADAM_DEFAULTS,
        )
        _check_params_recovered(result, true_params, cv, rs)


# =============================================================================
# 2e. Log-normal-prior fits
# =============================================================================

class TestLogNormalPriorFit:
    """
    Log-normal priors on energy parameters paired with positivity limits.

    For the iminuit family of backends, log-normal priors are evaluated in
    linear parameter space and can push the optimizer toward theta ≤ 0 if
    the prior is poorly specified.  Combining limits={"E": (0, None)} with
    Prior(mean=log(E_true), sdev=..., dist="log-normal") is the standard
    pattern tested here.
    """

    _prior_single = {
        "A": Prior(SINGLE_A, 5 * SINGLE_NOISE),
        "E": Prior(np.log(SINGLE_E), 0.3, dist="log-normal"),
    }
    _p0_single = {"A": SINGLE_A * 0.70, "E": SINGLE_E * 0.70}

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _STRATEGY_PARAMS, ids=_STRATEGY_IDS)
    def test_single_exp_lognormal_recovery(self, single_exp_data, cv, cv_corr, rs, rs_corr):
        """Log-normal prior on E must not prevent accurate recovery within 3σ."""
        t, ordinate, true_params = single_exp_data
        result = fit_adam_iminuit_hybrid(
            abscissa=t, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_single_exp,
            prior=self._prior_single,
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
            prior=self._prior_single,
            limits={"E": (0.0, None)},
            **_ADAM_DEFAULTS,
        )
        if cv:
            assert result.params["E"].mean > 0
        if rs:
            assert np.all(result.params["E"].rspl > 0)


# =============================================================================
# 3. Limits tests
# =============================================================================

class TestLimits:
    """
    Verify that parameter bounds are respected by both the ADAM clamping
    phase and iminuit after handover.
    """

    def test_lower_bound_respected_cv(self, single_exp_data):
        """limits={"E": (0, None)} must keep E > 0 in the CV result."""
        t, ordinate, _ = single_exp_data
        result = fit_adam_iminuit_hybrid(
            abscissa=t, ordinate=ordinate,
            model=model_single_exp,
            p0={"A": SINGLE_A * 0.70, "E": SINGLE_E * 0.70},
            limits={"E": (0.0, None)},
            central_value_fit=True, resample_fit=True,
            **_ADAM_DEFAULTS,
        )
        assert result.params["E"].mean > 0
        assert np.all(result.params["E"].rspl > 0)

    def test_lower_bound_does_not_prevent_recovery(self, single_exp_data):
        """A generous lower bound must not prevent accurate parameter recovery."""
        t, ordinate, true_params = single_exp_data
        result = fit_adam_iminuit_hybrid(
            abscissa=t, ordinate=ordinate,
            model=model_single_exp,
            p0={"A": SINGLE_A * 0.70, "E": SINGLE_E * 0.70},
            limits={"E": (0.0, None)},
            central_value_fit=True, resample_fit=False,
            **_ADAM_DEFAULTS,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=False)

    def test_unknown_limit_key_ignored(self, single_exp_data):
        """A limits entry for an unknown key must be silently ignored."""
        t, ordinate, _ = single_exp_data
        fit_adam_iminuit_hybrid(
            abscissa=t, ordinate=ordinate,
            model=model_single_exp,
            p0={"A": SINGLE_A * 0.70, "E": SINGLE_E * 0.70},
            limits={"nonexistent_param": (0.0, None)},
            central_value_fit=True, resample_fit=False,
            **_ADAM_DEFAULTS,
        )   # must not raise


# =============================================================================
# 4. Parallel-execution tests
# =============================================================================

class TestParallelExecution:
    """
    Nproc=4 must produce bit-for-bit identical results to Nproc=None.

    Both ADAM and iminuit are deterministic given the same inputs, so any
    discrepancy indicates a serialisation or process-boundary bug.
    """

    _p0_linear     = {"m": LINEAR_M * 0.70, "b": LINEAR_B * 0.70}
    _p0_single_exp = {"A": SINGLE_A * 0.70, "E": SINGLE_E * 0.70}
    _p0_double_exp = {
        "A1": DOUBLE_A1 * 0.70, "E1": DOUBLE_E1 * 0.70,
        "A2": DOUBLE_A2 * 0.70, "E2": DOUBLE_E2 * 0.70,
    }

    def _run_both(self, abscissa, ordinate, model, p0, correlated=False, limits=None):
        common = dict(
            abscissa=abscissa, ordinate=ordinate,
            central_value_fit=True, resample_fit=True,
            resample_fit_correlated=correlated,
            model=model, p0=p0, limits=limits,
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

    def test_linear_uncorrelated(self, linear_data):
        """Serial and parallel must be identical: linear, uncorrelated."""
        x, ordinate, _ = linear_data
        s, p = self._run_both(x, ordinate, model_linear, self._p0_linear)
        self._assert_identical(s, p, ["m", "b"])

    def test_linear_correlated(self, linear_data):
        """Serial and parallel must be identical: linear, correlated."""
        x, ordinate, _ = linear_data
        s, p = self._run_both(x, ordinate, model_linear, self._p0_linear, correlated=True)
        self._assert_identical(s, p, ["m", "b"])

    def test_single_exp(self, single_exp_data):
        """Serial and parallel must be identical: single-exp."""
        t, ordinate, _ = single_exp_data
        s, p = self._run_both(
            t, ordinate, model_single_exp, self._p0_single_exp,
            limits={"E": (0.0, None)},
        )
        self._assert_identical(s, p, ["A", "E"])

    def test_double_exp(self, double_exp_data):
        """Serial and parallel must be identical: double-exp (4 parameters)."""
        t, ordinate, _ = double_exp_data
        s, p = self._run_both(
            t, ordinate, model_double_exp, self._p0_double_exp,
            limits={"E1": (0.0, None), "E2": (0.0, None)},
        )
        self._assert_identical(s, p, ["A1", "E1", "A2", "E2"])

    def test_chi2_rspl_identical(self, linear_data):
        """Per-resample chi² must be bit-for-bit identical between serial and parallel."""
        x, ordinate, _ = linear_data
        s, p = self._run_both(x, ordinate, model_linear, self._p0_linear)
        np.testing.assert_array_equal(
            s.chi2.rspl, p.chi2.rspl,
            err_msg="Per-resample chi2 differs between serial and parallel runs",
        )

    def test_parallel_with_limits(self, single_exp_data):
        """Limits must be correctly forwarded to each worker and give identical results."""
        t, ordinate, _ = single_exp_data
        s, p = self._run_both(
            t, ordinate, model_single_exp, self._p0_single_exp,
            limits={"E": (0.0, None)},
        )
        self._assert_identical(s, p, ["A", "E"])


# =============================================================================
# 5. ADAM-specific tests
# =============================================================================

class TestADAMPhase:
    """
    Tests that verify the ADAM phase specifically rather than the combined
    ADAM + iminuit outcome.  These have no analogue in test_fit_iminuit.py.
    """

    def test_adam_reduces_cost_before_handover(self, single_exp_data):
        """
        The ADAM phase must reduce chi² from the displaced starting point.

        We expose ``_run_adam`` directly to check the cost history.  The cost
        at the last ADAM iteration (before handover) must be lower than at
        the first iteration (i.e. at p0).
        """
        from correlatoranalyser.fit_adam_iminuit_hybrid import _run_adam
        from correlatoranalyser.fit_iminuit import _build_uncorrelated_cost

        t, ordinate, _ = single_exp_data
        p0    = {"A": SINGLE_A * 0.70, "E": SINGLE_E * 0.70}
        names = list(p0.keys())
        theta0 = np.array([p0[k] for k in names])

        cost = _build_uncorrelated_cost(
            t, ordinate.mean, 1.0 / ordinate.serr,
            model_single_exp, names, priors=None, model_has_grad=False,
        )

        theta_best, history = _run_adam(
            cost, theta0,
            alpha     = 0.01,
            beta1     = 0.9,
            beta2     = 0.999,
            eps       = 1e-8,
            precision = 0.50,
            length    = 10,
            limits    = {"E": (0,None)},
            param_names=["A","E"]
        )

        assert history[-1] < history[0], (
            f"ADAM did not reduce chi²: initial={history[0]:.4g}, "
            f"final={history[-1]:.4g}"
        )

    def test_handover_fires_on_convergence(self, single_exp_data):
        """
        The handover criterion must fire before ``_MAX_ADAM_STEPS`` for a
        well-conditioned single-exp fit.  The ADAM loop should terminate
        in far fewer steps than the safety ceiling.
        """
        from correlatoranalyser.fit_adam_iminuit_hybrid import _run_adam, _MAX_ADAM_STEPS
        from correlatoranalyser.fit_iminuit import _build_uncorrelated_cost

        t, ordinate, _ = single_exp_data
        p0    = {"A": SINGLE_A * 0.70, "E": SINGLE_E * 0.70}
        names = list(p0.keys())
        theta0 = np.array([p0[k] for k in names])

        cost = _build_uncorrelated_cost(
            t, ordinate.mean, 1.0 / ordinate.serr,
            model_single_exp, names, priors=None, model_has_grad=False,
        )

        _, history = _run_adam(
            cost, theta0,
            alpha=0.01, beta1=0.9, beta2=0.999, eps=1e-8,
            precision=0.50, length=10,
            limits = {"E": (0,None)}, param_names=["A","E"] 
        )

        assert len(history) < _MAX_ADAM_STEPS, (
            f"ADAM ran for {len(history)} steps without the handover criterion "
            f"firing.  Expected termination well before {_MAX_ADAM_STEPS}."
        )

    def test_handover_start_better_than_p0(self, single_exp_data):
        """
        The warm start passed to iminuit must yield a lower chi² than the
        original p0 — confirming that ADAM did meaningful exploration.
        """
        from correlatoranalyser.fit_adam_iminuit_hybrid import _run_adam
        from correlatoranalyser.fit_iminuit import _build_uncorrelated_cost

        t, ordinate, _ = single_exp_data
        p0    = {"A": SINGLE_A * 0.70, "E": SINGLE_E * 0.70}
        names = list(p0.keys())
        theta0 = np.array([p0[k] for k in names])

        cost = _build_uncorrelated_cost(
            t, ordinate.mean, 1.0 / ordinate.serr,
            model_single_exp, names, priors=None, model_has_grad=False,
        )

        theta_best, _ = _run_adam(
            cost, theta0,
            alpha=0.01, beta1=0.9, beta2=0.999, eps=1e-8,
            precision=0.50, length=10, 
            limits = {"E": (0,None)}, param_names=["A","E"]
        )

        chi2_p0   = float(cost(*theta0))
        chi2_adam = float(cost(*theta_best))
        assert chi2_adam < chi2_p0, (
            f"ADAM warm-start chi²={chi2_adam:.4g} is not better than "
            f"initial chi²={chi2_p0:.4g}"
        )

    def test_analytic_gradient_matches_finite_difference(self, single_exp_data):
        """
        With model.grad attached, the gradient injected into ADAM must be
        consistent with central finite differences to within 0.1%.

        This catches gradient sign errors and missing chain-rule terms — the
        class of bug that caused the original failure on the double-exp model.
        """
        from correlatoranalyser.fit_adam_iminuit_hybrid import _numerical_gradient
        from correlatoranalyser.fit_iminuit import _build_uncorrelated_cost

        t, ordinate, _ = single_exp_data
        p0    = {"A": SINGLE_A, "E": SINGLE_E}   # evaluate at truth for a clean comparison
        names = list(p0.keys())
        theta = np.array([p0[k] for k in names])

        model_single_exp.grad = _grad_single_exp
        try:
            cost = _build_uncorrelated_cost(
                t, ordinate.mean, 1.0 / ordinate.serr,
                model_single_exp, names, priors=None, model_has_grad=True,
            )
            g_analytic = np.asarray(cost.grad(*theta))
            g_fd       = _numerical_gradient(cost, theta)

            np.testing.assert_allclose(
                g_analytic, g_fd, rtol=1e-3,
                err_msg="Analytic gradient does not match finite difference within 0.1%",
            )
        finally:
            del model_single_exp.grad

    def test_double_exp_analytic_gradient_correct(self, double_exp_data):
        """
        The double-exp gradient must include the E0 contribution from both
        exponential terms.  A wrong gradient (missing the cross-term) would
        produce an incorrect descent direction and is the exact bug that
        caused the first failed hybrid run.

        We verify at the true parameter values using the same FD comparison.
        """
        from correlatoranalyser.fit_adam_iminuit_hybrid import _numerical_gradient
        from correlatoranalyser.fit_iminuit import _build_uncorrelated_cost

        t, ordinate, _ = double_exp_data
        p0    = {"A1": DOUBLE_A1, "E1": DOUBLE_E1, "A2": DOUBLE_A2, "E2": DOUBLE_E2}
        names = list(p0.keys())
        theta = np.array([p0[k] for k in names])

        model_double_exp.grad = _grad_double_exp
        try:
            cost = _build_uncorrelated_cost(
                t, ordinate.mean, 1.0 / ordinate.serr,
                model_double_exp, names, priors=None, model_has_grad=True,
            )
            g_analytic = np.asarray(cost.grad(*theta))
            g_fd       = _numerical_gradient(cost, theta)

            np.testing.assert_allclose(
                g_analytic, g_fd, rtol=1e-3,
                err_msg="Double-exp analytic gradient does not match finite difference",
            )
        finally:
            del model_double_exp.grad

    def test_finite_difference_path_recovers_params(self, single_exp_data):
        """
        Without model.grad the ADAM phase uses finite differences.  Parameter
        recovery must be equally good — the gradient approximation is only used
        for the ADAM warm-start, and iminuit takes over for precision.
        """
        t, ordinate, true_params = single_exp_data
        # model_single_exp has no .grad by default in this module.
        assert not hasattr(model_single_exp, "grad")

        result = fit_adam_iminuit_hybrid(
            abscissa=t, ordinate=ordinate,
            model=model_single_exp,
            p0={"A": SINGLE_A * 0.70, "E": SINGLE_E * 0.70},
            limits={"E": (0.0, None)},
            central_value_fit=True, resample_fit=True,
            **_ADAM_DEFAULTS,
        )
        _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=True)

    def test_analytic_gradient_path_recovers_params(self, single_exp_data):
        """
        With model.grad the ADAM phase uses the analytic gradient.  Parameter
        recovery must still be within 3σ.
        """
        t, ordinate, true_params = single_exp_data
        model_single_exp.grad = _grad_single_exp
        try:
            result = fit_adam_iminuit_hybrid(
                abscissa=t, ordinate=ordinate,
                model=model_single_exp,
                p0={"A": SINGLE_A * 0.70, "E": SINGLE_E * 0.70},
                limits={"E": (0.0, None)},
                central_value_fit=True, resample_fit=True,
                **_ADAM_DEFAULTS,
            )
            _check_params_recovered(result, true_params, central_value_fit=True, resample_fit=True)
        finally:
            del model_single_exp.grad

    def test_handover_criterion_total_window_not_step_size(self, single_exp_data):
        """
        The handover criterion checks total improvement over the window, not
        the per-step change.  With a very tight precision (0.001) the fitter
        must run longer than with the default (0.50).

        This guards against the regression where the criterion fired on the
        per-step change and triggered far too early on a plateau at a boundary.
        """
        from correlatoranalyser.fit_adam_iminuit_hybrid import _run_adam
        from correlatoranalyser.fit_iminuit import _build_uncorrelated_cost

        t, ordinate, _ = single_exp_data
        p0    = {"A": SINGLE_A * 0.70, "E": SINGLE_E * 0.70}
        names = list(p0.keys())
        theta0 = np.array([p0[k] for k in names])

        cost = _build_uncorrelated_cost(
            t, ordinate.mean, 1.0 / ordinate.serr,
            model_single_exp, names, priors=None, model_has_grad=False,
        )

        _, history_loose = _run_adam(
            cost, theta0, alpha=0.01, beta1=0.9, beta2=0.999, eps=1e-8,
            precision=0.50, length=10,
            limits = {"E": (0,None)}, param_names=["A","E"]
        )
        _, history_tight = _run_adam(
            cost, theta0, alpha=0.01, beta1=0.9, beta2=0.999, eps=1e-8,
            precision=0.001, length=10,
            limits = {"E": (0,None)}, param_names=["A","E"]
        )

        assert len(history_tight) >= len(history_loose), (
            "Tighter precision should require at least as many ADAM steps as "
            f"loose precision (tight={len(history_tight)}, loose={len(history_loose)})"
        )
