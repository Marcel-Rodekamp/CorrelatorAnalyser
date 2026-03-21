"""
Cross-backend consistency tests.

All backends must converge to the same minimum when given identical data and
starting conditions. 

Backends under test
-------------------
For both the linear model and the single-exponential model:

  B1  lsqfit             (Levenberg-Marquardt via lsqfit)
  B2  iminuit            (MIGRAD, plain)
  B3  iminuit + varproj  (MIGRAD on nonlinear subspace)
  B4  hybrid adam+iminuit
  B5  hybrid adam+iminuit + varproj

For a single exponential we check thc agains B1-B5

Tolerance design
----------------
MIGRAD stops when the estimated distance to minimum (EDM) satisfies

    EDM < 0.002 × tol × UP

With the default tol=0.1 and UP=1 (chi² fits) this gives EDM < 2e-4.
Two independent MIGRAD runs from different starting points can therefore
only be guaranteed to agree up to ~2e-4 in chi² and parameter values.
ATOL_IMINUIT is set just below this ceiling.

lsqfit uses Levenberg-Marquardt with its own internal convergence
criterion, which is O(1e-8). Both algorithms should converge to the same 
minimum. The inter-backend tolerance is determined by the maximimum tolarance of 
the algorithms.

Both tolerances are still far tighter than any statistical uncertainty
on the parameters (~1e-2 for the noise levels used here), so these
comparisons remain meaningful tests that a wrong-minimum failure would
violate by many orders of magnitude.


Strategy matrix
---------------
We test every combination of:
  central_value_fit × {True, False}
  resample_fit      × {True, False}
  correlated        × {True, False}

subject to:
  - at least one of central_value_fit / resample_fit must be True
  - lsqfit does not support parallel execution with correlated data (not
    tested here — this file is serial-only)
  - varproj requires at least one nonlinear parameter (not applicable to
    the pure-linear model, so varproj backends are only tested on
    single-exp)

Resample-by-resample comparison
---------------------------------
For strategies that include resample fits the comparison is made resample by resample
"""

from __future__ import annotations

import numpy as np
import pytest

from correlatoranalyser import Data
from correlatoranalyser.fit_lsqfit   import fit_lsqfit
from correlatoranalyser.fit_iminuit  import fit_iminuit
from correlatoranalyser.fit_iminuit_variable_projection import fit_iminuit as fit_varproj
from correlatoranalyser.fit_adam_iminuit_hybrid import fit_adam_iminuit_hybrid
from correlatoranalyser.thc import thc

# =============================================================================
# Tolerance constants
# =============================================================================

ATOL_IMINUIT: float = 2e-4   # iminuit-family vs iminuit-family
ATOL_LSQFIT:  float = 2e-4   # lsqfit vs iminuit (different algorithm)
ATOL_THC:     float = 5 * 0.02 # 5 * SINGLE_NOISE

# =============================================================================
# Ground-truth parameters
# =============================================================================

LINEAR_M:     float = 2.5
LINEAR_B:     float = 0.7
LINEAR_NOISE: float = 0.10

SINGLE_A:     float = 1.5
SINGLE_E:     float = 0.3
SINGLE_NOISE: float = 0.02

# Starting point: 10 % of truth (very displaced — forces real exploration)
P0_LINEAR     = {"m": LINEAR_M * 0.10, "b": LINEAR_B * 0.10}
P0_SINGLE_EXP = {"A": SINGLE_A * 0.10, "E": SINGLE_E * 0.10}

# ADAM hyperparameters (same across all hybrid tests)
_ADAM = dict(
    adam_hyperparam_alpha   = 0.01,
    adam_hyperparam_beta1   = 0.9,
    adam_hyperparam_beta2   = 0.999,
    adam_hyperparam_eps     = 1e-8,
    adam_handover_precision = 0.50,
    adam_handover_length    = 10,
)

# =============================================================================
# Mock-data factory   (one shared realisation — all backends use the same Data)
# =============================================================================

_RNG_SEED = 20240101 + 99   # distinct from all other test suites

def _make_data(true_values, noise, nbst=200) -> Data:
    n   = len(true_values)
    cov = noise ** 2 * (np.eye(n) + 0.1 * np.ones((n, n)))
    raw = np.random.default_rng(_RNG_SEED).multivariate_normal(
        mean=true_values, cov=cov, size=600
    )
    return Data(resample_type="bst", data=raw, Nresample=nbst)


# =============================================================================
# Module-level fixtures — created once and shared by all test classes
# =============================================================================

@pytest.fixture(scope="module")
def linear_ordinate():
    x      = np.linspace(0.0, 1.0, 8)
    true_y = LINEAR_M * x + LINEAR_B
    return x, _make_data(true_y, LINEAR_NOISE)


@pytest.fixture(scope="module")
def single_exp_ordinate():
    t      = np.arange(1, 11, dtype=float)
    true_C = SINGLE_A * np.exp(-SINGLE_E * t)
    return t, _make_data(true_C, SINGLE_NOISE)


@pytest.fixture(scope="module")
def thc_ordinate():
    """
    21-timeslice single-exponential data: t = 0, 1, ..., 20.
 
    T_total=21, T_eff=20 (even), n=10.  Same physics and noise as the
    existing ``single_exp_ordinate`` fixture, but starting at t=0 so that
    T_eff is even (THC requirement).  Same seed as the rest of the file.
    """
    t      = np.arange(0, 21, dtype=float)
    true_C = SINGLE_A * np.exp(-SINGLE_E * t)
    n      = len(true_C)
    cov    = SINGLE_NOISE ** 2 * (np.eye(n) + 0.1 * np.ones((n, n)))
    raw    = np.random.default_rng(_RNG_SEED).multivariate_normal(
        mean=true_C, cov=cov, size=600
    )
    return t, Data(resample_type="bst", data=raw, Nresample=200)

# =============================================================================
# Model functions (module-level for dill serialisation in parallel workers)
# =============================================================================

def _grad_linear(x, p):
    """dC/d[m, b] — shape (2, N)."""
    return np.array([x, np.ones_like(x)])

def model_linear(x, p):
    return p["m"] * x + p["b"]

model_linear.grad     = _grad_linear


def _grad_single_exp(t, p):
    """dC/d[A, E] — shape (2, N)."""
    exp_Et = np.exp(-p["E"] * t)
    return np.array([
        exp_Et,                   # dC/dA
        -p["A"] * t * exp_Et,     # dC/dE
    ])

def _grad_single_exp_varproj(t, p):
    """Nonlinear-only Jacobian shape (1, N) — for varproj and hybrid+varproj."""
    return np.array([-p["A"] * t * np.exp(-p["E"] * t) ])

def model_single_exp(t, p):
    return p["A"] * np.exp(-p["E"] * t)

model_single_exp.grad = _grad_single_exp

# =============================================================================
# Fit-strategy parametrisation
# =============================================================================

# (id, central_value_fit, cv_correlated, resample_fit, rs_correlated)
FIT_STRATEGIES = [
    ("cv_uncorr",    True,  False, False, False),
    ("cv_corr",      True,  True,  False, False),
    ("rs_uncorr",    False, False, True,  False),
    ("rs_corr",      False, False, True,  True ),
    ("both_uncorr",  True,  False, True,  False),
    ("both_corr",    True,  True,  True,  True ),
]
_IDS    = [s[0] for s in FIT_STRATEGIES]
_PARAMS = [s[1:] for s in FIT_STRATEGIES]

# Strategy matrix for the THC class
_THC_STRATEGIES = [
    # id              cv     cv_c   rs     rs_c
    ("cv_uncorr",    True,  False, False, False),
    ("cv_corr",      True,  True,  False, False),
    ("rs_uncorr",    False, False, True,  False),
    ("rs_corr",      False, False, True,  True ),
    ("both_uncorr",  True,  False, True,  False),
    ("both_corr",    True,  True,  True,  True ),
]
_THC_IDS    = [s[0] for s in _THC_STRATEGIES]
_THC_PARAMS = [s[1:] for s in _THC_STRATEGIES]


# =============================================================================
# Comparison helpers
# =============================================================================

def _assert_chi2_equal(ref, other, atol: float, label: str) -> None:
    """
    Compare chi² values (CV and per-resample) between two FitResult objects.
    """
    if isinstance(ref.chi2, Data) and isinstance(other.chi2, Data):
        if other.chi2.mean is None:
            # no CV fit
            pass
        else:
            np.testing.assert_allclose(other.chi2.mean, ref.chi2.mean, atol=atol, rtol=0, err_msg=f"{label}: CV chi2 mismatch")
    elif isinstance(ref.chi2, float) and isinstance(other.chi2, float):
        np.testing.assert_allclose(other.chi2, ref.chi2, atol=atol, rtol=0, err_msg=f"{label}: CV chi2 mismatch")
    else:
        assert False

    if ref.has_resamples and other.has_resamples:
        for k in range(ref.chi2.rspl.shape[0]):
            np.testing.assert_allclose(
                other.chi2.rspl[k], ref.chi2.rspl[k],
                atol=atol, rtol=0,
                err_msg=f"{label}: chi2 mismatch at resample {k}",
            )


def _assert_params_equal(ref, other, atol: float, label: str) -> None:
    """
    Compare every parameter (CV mean and all rspl entries) between two
    FitResult objects.  Comparison is resample-by-resample.
    """
    for key in ref.params:
        if hasattr(ref.params[key], "mean") and ref.params[key].mean is not None:
            np.testing.assert_allclose(
                other.params[key].mean, ref.params[key].mean,
                atol=atol, rtol=0,
                err_msg=f"{label}: CV mean mismatch for '{key}'",
            )

        if ref.has_resamples and other.has_resamples:
            for k in range(ref.params[key].rspl.shape[0]):
                np.testing.assert_allclose(
                    other.params[key].rspl[k], ref.params[key].rspl[k],
                    atol=atol, rtol=0,
                    err_msg=f"{label}: rspl[{k}] mismatch for '{key}'",
                )


def _compare_thc_vs_minimiser(
    thc_result,
    min_result,
    thc_key: str,
    min_key: str,
    label: str,
    cv: bool,
    rs: bool,
) -> None:
    """
    Compare a single parameter between a THC result and a minimising backend.
 
    CV means are compared directly.  For resample fits, ensemble means
    (nanmean of THC rspl vs mean of iminuit rspl) are compared.
    Per-resample comparison is intentionally omitted.
    """
    if cv:
        np.testing.assert_allclose(
            thc_result.params[thc_key].mean,
            min_result.params[min_key].mean,
            atol=ATOL_THC, rtol=0,
            err_msg=f"{label}: CV mean mismatch '{thc_key}' vs '{min_key}'",
        )
 
    if rs:
        thc_rs_mean = float(np.nanmean(thc_result.params[thc_key].rspl))
        min_rs_mean = float(np.mean(min_result.params[min_key].rspl))
        np.testing.assert_allclose(
            thc_rs_mean, min_rs_mean,
            atol=ATOL_THC, rtol=0,
            err_msg=(
                f"{label}: resample-mean mismatch '{thc_key}' vs '{min_key}': "
                f"THC={thc_rs_mean:.6g}, minimiser={min_rs_mean:.6g}"
            ),
        )


def _compare(ref, other, atol: float, label: str) -> None:
    _assert_chi2_equal(ref, other, atol, label)
    _assert_params_equal(ref, other, atol, label)


# =============================================================================
# 1. Cross-backend consistency — linear model
#    varproj is excluded (linear model has no nonlinear parameter)
# =============================================================================

class TestConsistencyLinear:
    """
    Fit y = m·x + b from all applicable backends and verify that chi² and
    parameter values agree to ATOL_IMINUIT (iminuit family) or ATOL_LSQFIT
    (lsqfit) against the plain iminuit reference.

    varproj backends are excluded because the linear model has no nonlinear
    parameter to optimise in the reduced subspace.

    p0 = 10 % of truth — a severe displacement that forces every backend to
    do real optimisation.  All should still converge to the same minimum.
    """

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _PARAMS, ids=_IDS)
    def test_iminuit_vs_lsqfit(self, linear_ordinate, cv, cv_corr, rs, rs_corr):
        """lsqfit and iminuit must converge to the same minimum (within ATOL_LSQFIT)."""
        x, ordinate = linear_ordinate

        ref = fit_iminuit(
            abscissa=x, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_linear, p0=P0_LINEAR, strategy=2,
        )
        other = fit_lsqfit(
            abscissa=x, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_linear, p0=P0_LINEAR,
        )
        _compare(ref, other, atol=ATOL_LSQFIT, label="lsqfit vs iminuit [linear]")

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _PARAMS, ids=_IDS)
    def test_iminuit_vs_hybrid(self, linear_ordinate, cv, cv_corr, rs, rs_corr):
        """ADAM+iminuit hybrid must agree with plain iminuit to ATOL_IMINUIT."""
        x, ordinate = linear_ordinate

        ref = fit_iminuit(
            abscissa=x, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_linear, p0=P0_LINEAR, strategy=2,
        )
        other = fit_adam_iminuit_hybrid(
            abscissa=x, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_linear, p0=P0_LINEAR, strategy=2,
            **_ADAM,
        )
        _compare(ref, other, atol=ATOL_IMINUIT, label="hybrid vs iminuit [linear]")


# =============================================================================
# 2. Cross-backend consistency — single-exponential model
#    All six backends are tested here, including both varproj variants
# =============================================================================

class TestConsistencySingleExp:
    """
    Fit C(t) = A·exp(-E·t) from all six backends and verify agreement.

    For the varproj backends linear_params=["A"], nonlinear_params=["E"].

    iminuit family (B2, B3, B5, B6): compared at ATOL_IMINUIT.
    lsqfit (B1):                     compared at ATOL_LSQFIT.
    ADAM internal (B4):              verified separately in TestADAMInternal.

    The positivity limit {"E": (0, None)} is set for all backends that
    support limits, matching the standard practice for exponential fits.
    """

    _limits = {"E": (0.0, None)}

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _PARAMS, ids=_IDS)
    def test_iminuit_vs_lsqfit(self, single_exp_ordinate, cv, cv_corr, rs, rs_corr):
        """lsqfit must agree with iminuit to ATOL_LSQFIT."""
        t, ordinate = single_exp_ordinate

        ref = fit_iminuit(
            abscissa=t, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_single_exp, p0=P0_SINGLE_EXP,
            limits=self._limits, strategy=2,
        )
        other = fit_lsqfit(
            abscissa=t, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_single_exp, p0=P0_SINGLE_EXP,
        )
        _compare(ref, other, atol=ATOL_LSQFIT, label="lsqfit vs iminuit [single-exp]")

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _PARAMS, ids=_IDS)
    def test_iminuit_vs_varproj(self, single_exp_ordinate, cv, cv_corr, rs, rs_corr):
        """
        iminuit and iminuit+varproj use different cost functions (full parameter
        space vs nonlinear subspace) but must converge to the same minimum.
        """
        t, ordinate = single_exp_ordinate

        ref = fit_iminuit(
            abscissa=t, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_single_exp, p0=P0_SINGLE_EXP,
            limits=self._limits, strategy=2,
        )
        model_single_exp.grad = _grad_single_exp_varproj
        try:
            other = fit_varproj(
                abscissa=t, ordinate=ordinate,
                central_value_fit=cv, central_value_fit_correlated=cv_corr,
                resample_fit=rs,      resample_fit_correlated=rs_corr,
                model=model_single_exp, linear_params=["A"],
                p0=P0_SINGLE_EXP,
                limits=self._limits, strategy=2,
            )
        finally:
            model_single_exp.grad = _grad_single_exp
        _compare(ref, other, atol=ATOL_IMINUIT, label="varproj vs iminuit [single-exp]")

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _PARAMS, ids=_IDS)
    def test_iminuit_vs_hybrid(self, single_exp_ordinate, cv, cv_corr, rs, rs_corr):
        """ADAM+iminuit hybrid must agree with plain iminuit to ATOL_IMINUIT."""
        t, ordinate = single_exp_ordinate

        ref = fit_iminuit(
            abscissa=t, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_single_exp, p0=P0_SINGLE_EXP,
            limits=self._limits, strategy=2,
        )
        other = fit_adam_iminuit_hybrid(
            abscissa=t, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_single_exp, p0=P0_SINGLE_EXP,
            limits=self._limits, strategy=2,
            **_ADAM,
        )
        _compare(ref, other, atol=ATOL_IMINUIT, label="hybrid vs iminuit [single-exp]")

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _PARAMS, ids=_IDS)
    def test_iminuit_vs_hybrid_varproj(self, single_exp_ordinate, cv, cv_corr, rs, rs_corr):
        """ADAM+iminuit+varproj hybrid must agree with plain iminuit to ATOL_IMINUIT."""
        t, ordinate = single_exp_ordinate

        ref = fit_iminuit(
            abscissa=t, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_single_exp, p0=P0_SINGLE_EXP,
            limits=self._limits, strategy=2,
        )
        model_single_exp.grad = _grad_single_exp_varproj
        try:
            other = fit_adam_iminuit_hybrid(
                abscissa=t, ordinate=ordinate,
                central_value_fit=cv, central_value_fit_correlated=cv_corr,
                resample_fit=rs,      resample_fit_correlated=rs_corr,
                model=model_single_exp, linear_params=["A"],
                p0=P0_SINGLE_EXP, limits=self._limits, strategy=2,
                **_ADAM,
            )
        finally:
            model_single_exp.grad = _grad_single_exp
        _compare(ref, other, atol=ATOL_IMINUIT, label="hybrid+varproj vs iminuit [single-exp]")

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _PARAMS, ids=_IDS)
    def test_hybrid_vs_hybrid_varproj(self, single_exp_ordinate, cv, cv_corr, rs, rs_corr):
        """
        The two hybrid variants (plain and varproj) must agree with each other.
        Both warm-start from the same ADAM phase and then run iminuit on their
        respective cost functions; the minimum must be the same.
        """
        t, ordinate = single_exp_ordinate

        plain = fit_adam_iminuit_hybrid(
            abscissa=t, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_single_exp, p0=P0_SINGLE_EXP,
            limits=self._limits, strategy=2,
            **_ADAM,
        )
        model_single_exp.grad = _grad_single_exp_varproj
        try:
            vp = fit_adam_iminuit_hybrid(
                abscissa=t, ordinate=ordinate,
                central_value_fit=cv, central_value_fit_correlated=cv_corr,
                resample_fit=rs,      resample_fit_correlated=rs_corr,
                model=model_single_exp, linear_params=["A"],
                p0=P0_SINGLE_EXP, limits=self._limits, strategy=2,
                **_ADAM,
            )
        finally:
            model_single_exp.grad = _grad_single_exp
        _compare(plain, vp, atol=ATOL_IMINUIT, label="hybrid vs hybrid+varproj [single-exp]")

# =============================================================================
# 3. Cross-backend - THC - consistency — single-exponential model
# =============================================================================

class TestConsistencyTHC:
    """
    Compare THC against iminuit for C(t) = A·exp(-E·t).
 
    THC (algebraic Hankel approach) and iminuit (chi² minimisation) are two
    statistically consistent estimators of the same physical parameters.
    Agreement is verified at ATOL_THC rather than ATOL_IMINUIT because the
    two algorithms have different per-sample behaviour.
 
    Tests
    -----
    * Energy and amplitude CV means agree within ATOL_THC.
    * Resample ensemble means agree within ATOL_THC.
    * THC parameters are invariant to the correlated/uncorrelated chi² flag
      (that flag only affects the post-hoc chi² computation, not the Hankel
      eigenvalue solution).
    * Standalone truth-recovery checks for E and A.
    """
 
    _p0     = {"A": SINGLE_A * 0.10, "E": SINGLE_E * 0.10}
    _limits = {"E": (0.0, None)}
 
    def _thc(self, ordinate, cv, cv_corr, rs, rs_corr):
        return thc(
            ordinate                     = ordinate,
            central_value_fit            = cv,
            central_value_fit_correlated = cv_corr,
            resample_fit                 = rs,
            resample_fit_correlated      = rs_corr,
            truncation_dimension         = 1,
        )
 
    def _iminuit(self, t, ordinate, cv, cv_corr, rs, rs_corr):
        return fit_iminuit(
            abscissa                     = t,
            ordinate                     = ordinate,
            central_value_fit            = cv,
            central_value_fit_correlated = cv_corr,
            resample_fit                 = rs,
            resample_fit_correlated      = rs_corr,
            model                        = model_single_exp,
            p0                           = self._p0,
            limits                       = self._limits,
        )
 
    def _lsqfit(self, t, ordinate, cv, cv_corr, rs, rs_corr):
        return fit_lsqfit(
            abscissa                     = t,
            ordinate                     = ordinate,
            central_value_fit            = cv,
            central_value_fit_correlated = cv_corr,
            resample_fit                 = rs,
            resample_fit_correlated      = rs_corr,
            model                        = model_single_exp,
            p0                           = self._p0,
        )
 
    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _THC_PARAMS, ids=_THC_IDS)
    def test_thc_vs_iminuit_energy(self, thc_ordinate, cv, cv_corr, rs, rs_corr):
        """Ground-state energy E0 (THC) must agree with E (iminuit) within ATOL_THC."""
        t, ordinate = thc_ordinate
        thc_res = self._thc(ordinate, cv, cv_corr, rs, rs_corr)
        min_res = self._iminuit(t, ordinate, cv, cv_corr, rs, rs_corr)
        _compare_thc_vs_minimiser(
            thc_res, min_res, "E0", "E",
            label=f"energy [thc vs iminuit, {cv=}, {rs=}]",
            cv=cv, rs=rs,
        )
 
    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _THC_PARAMS, ids=_THC_IDS)
    def test_thc_vs_iminuit_amplitude(self, thc_ordinate, cv, cv_corr, rs, rs_corr):
        """Amplitude A0 (THC) must agree with A (iminuit) within ATOL_THC."""
        t, ordinate = thc_ordinate
        thc_res = self._thc(ordinate, cv, cv_corr, rs, rs_corr)
        min_res = self._iminuit(t, ordinate, cv, cv_corr, rs, rs_corr)
        _compare_thc_vs_minimiser(
            thc_res, min_res, "A0", "A",
            label=f"amplitude [thc vs iminuit, {cv=}, {rs=}]",
            cv=cv, rs=rs,
        )
 
    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _THC_PARAMS, ids=_THC_IDS)
    def test_thc_params_invariant_to_chi2_weight(
        self, thc_ordinate, cv, cv_corr, rs, rs_corr
    ):
        """
        Switching between correlated and uncorrelated chi² weight in THC must
        not change the extracted parameters — only the post-hoc chi² value.
 
        The uncorrelated result is used as reference; the correlated result
        must be bit-for-bit identical in params (same Hankel solve, different W).
        """
        _, ordinate = thc_ordinate
 
        uncorr = thc(
            ordinate             = ordinate,
            central_value_fit    = cv,
            resample_fit         = rs,
            truncation_dimension = 1,
            # both correlated flags off
            central_value_fit_correlated = False,
            resample_fit_correlated      = False,
        )
        corr = self._thc(ordinate, cv, cv_corr, rs, rs_corr)
 
        for key in ("E0", "A0"):
            if cv:
                np.testing.assert_array_equal(
                    uncorr.params[key].mean, corr.params[key].mean,
                    err_msg=(
                        f"CV mean for '{key}' changed between uncorrelated and "
                        f"correlated THC — correlated flags must not affect the "
                        f"Hankel eigenvalue solve"
                    ),
                )
            if rs:
                np.testing.assert_array_equal(
                    uncorr.params[key].rspl, corr.params[key].rspl,
                    err_msg=f"rspl for '{key}' changed between uncorr and corr THC",
                )
 
    def test_thc_energy_close_to_truth(self, thc_ordinate):
        """THC CV energy must lie within ATOL_THC of SINGLE_E = 0.3."""
        _, ordinate = thc_ordinate
        result = thc(ordinate=ordinate, truncation_dimension=1)
        assert abs(result.params["E0"].mean - SINGLE_E) < ATOL_THC, (
            f"THC E0={result.params['E0'].mean:.6g} deviates more than "
            f"ATOL_THC={ATOL_THC:.3g} from true E={SINGLE_E}"
        )
 
    def test_thc_amplitude_close_to_truth(self, thc_ordinate):
        """THC CV amplitude must lie within ATOL_THC of SINGLE_A = 1.5."""
        _, ordinate = thc_ordinate
        result = thc(ordinate=ordinate, truncation_dimension=1)
        assert abs(result.params["A0"].mean - SINGLE_A) < ATOL_THC, (
            f"THC A0={result.params['A0'].mean:.6g} deviates more than "
            f"ATOL_THC={ATOL_THC:.3g} from true A={SINGLE_A}"
        )
 
    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _THC_PARAMS, ids=_THC_IDS)
    def test_thc_vs_lsqfit_energy(self, thc_ordinate, cv, cv_corr, rs, rs_corr):
        """Ground-state energy E0 (THC) must agree with E (lsqfit) within ATOL_THC."""
        t, ordinate = thc_ordinate
        thc_res = self._thc(ordinate, cv, cv_corr, rs, rs_corr)
        lsq_res = self._lsqfit(t, ordinate, cv, cv_corr, rs, rs_corr)
        _compare_thc_vs_minimiser(
            thc_res, lsq_res, "E0", "E",
            label=f"energy [thc vs lsqfit, {cv=}, {rs=}]",
            cv=cv, rs=rs,
        )
 
    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _THC_PARAMS, ids=_THC_IDS)
    def test_thc_vs_lsqfit_amplitude(self, thc_ordinate, cv, cv_corr, rs, rs_corr):
        """Amplitude A0 (THC) must agree with A (lsqfit) within ATOL_THC."""
        t, ordinate = thc_ordinate
        thc_res = self._thc(ordinate, cv, cv_corr, rs, rs_corr)
        lsq_res = self._lsqfit(t, ordinate, cv, cv_corr, rs, rs_corr)
        _compare_thc_vs_minimiser(
            thc_res, lsq_res, "A0", "A",
            label=f"amplitude [thc vs lsqfit, {cv=}, {rs=}]",
            cv=cv, rs=rs,
        )
 
    def test_thc_resample_energy_close_to_iminuit(self, thc_ordinate):
        """
        Single combined test: resample-mean energy and amplitude from THC must
        agree with both iminuit and lsqfit for the both_uncorr strategy.  This
        is the most complete single integration test that covers CV + resample
        behaviour and cross-validates all three backends simultaneously.
        """
        t, ordinate = thc_ordinate
        thc_res = thc(
            ordinate             = ordinate,
            central_value_fit    = True,
            resample_fit         = True,
            truncation_dimension = 1,
        )
        min_res = fit_iminuit(
            abscissa          = t,
            ordinate          = ordinate,
            central_value_fit = True,
            resample_fit      = True,
            model             = model_single_exp,
            p0                = self._p0,
            limits            = self._limits,
        )
        lsq_res = fit_lsqfit(
            abscissa          = t,
            ordinate          = ordinate,
            central_value_fit = True,
            resample_fit      = True,
            model             = model_single_exp,
            p0                = self._p0,
        )
 
        for key_thc, key_min, label in (
            ("E0", "E", "energy"),
            ("A0", "A", "amplitude"),
        ):
            _compare_thc_vs_minimiser(
                thc_res, min_res, key_thc, key_min,
                label=f"{label} [thc vs iminuit, both_uncorr]",
                cv=True, rs=True,
            )
            _compare_thc_vs_minimiser(
                thc_res, lsq_res, key_thc, key_min,
                label=f"{label} [thc vs lsqfit, both_uncorr]",
                cv=True, rs=True,
            )
