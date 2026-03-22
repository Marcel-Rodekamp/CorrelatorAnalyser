"""
Unit tests for analytic Hessian support in the iminuit backend
(fit_iminuit.py extended with model.hessian).

Overview
--------
The analytic Hessian is provided as ``model.hessian(abscissa, params)``
returning an array of shape ``(Nparams, Nparams, Ndata)`` where

    H[i, j, k]  =  d2 model_k / (dtheta_i dtheta_j)

When both ``model.grad`` and ``model.hessian`` are present the cost-function
builders attach ``cost.hess``, which iminuit >= 2.17 uses in ``hesse()`` to
compute the covariance matrix analytically rather than by finite differences.

Test classes
------------
1. TestPriorHess
       Analytic second derivatives of Prior.__call__ for normal and log-normal
       distributions, verified against finite differences of Prior.grad.

2. TestCostHessAttachment
       ``cost.hess`` is attached iff both model.grad and model.hessian are
       present; missing either one suppresses it.

3. TestCostHessCorrectness
       ``cost.hess(*values)`` agrees with central finite differences of
       ``cost.grad(*values)`` for single- and double-exponential models,
       both uncorrelated and correlated paths.

5. TestHessianVsGradErrors
       Hessian errors from the analytic-hessian path must match those from
       the grad-only (numerical hesse) path to within a tight tolerance,
       confirming that the analytic formulas are self-consistent.

6. TestHessianWithPriors
       Normal and log-normal priors contribute correctly to ``cost.hess``
       -- verified via finite-difference consistency and recovery tests.

7. TestCostGradAttachment
       ``cost.grad`` is attached iff model.grad is present, for both
       uncorrelated and correlated builders, regardless of whether
       model.hessian is also present.

8. TestCostGradCorrectness
       ``cost.grad(*values)`` agrees with central finite differences of
       ``cost(*values)`` for single- and double-exponential models, both
       uncorrelated and correlated paths, with and without priors, and both
       with and without model.hessian attached (hessian must not disturb
       the grad computation).
"""

from __future__ import annotations

import numpy as np
import pytest

from correlatoranalyser import Data
from correlatoranalyser.fit_iminuit import (
    fit_iminuit,
    _build_uncorrelated_cost,
    _build_correlated_cost,
)
from correlatoranalyser.prior import Prior

# =============================================================================
# Global configuration
# =============================================================================

_RNG_SEED = 20240101 + 40   # distinct from all other test suites
NRAW: int  = 600
NBST: int  = 200             # small for speed; hessian tests don't need large N
NSIG: int  = 3

SINGLE_A:     float = 1.5
SINGLE_E:     float = 0.3
SINGLE_NOISE: float = 0.02

DOUBLE_A1:    float = 1.5
DOUBLE_E1:    float = 0.3
DOUBLE_A2:    float = 0.5
DOUBLE_E2:    float = 0.8
DOUBLE_NOISE: float = 0.01

# =============================================================================
# Model functions -- module-level for dill pickling
# =============================================================================

# -- single-exponential -------------------------------------------------------

def model_single_exp(t, p):
    return p["A"] * np.exp(-p["E"] * t)


def _grad_single_exp(t, p):
    """dC/d[A, E], shape (2, N)."""
    e = np.exp(-p["E"] * t)
    return np.array([e, -p["A"] * t * e])


def _hess_single_exp(t, p):
    """
    d2C/(dtheta_i dtheta_j), shape (2, 2, N).

    Param order: [A, E].
      H[0,0] = d2C/dA2    = 0
      H[0,1] = d2C/dAdE   = -t * exp(-Et)
      H[1,0] = H[0,1]     (symmetric)
      H[1,1] = d2C/dE2    = A * t2 * exp(-Et)
    """
    e = np.exp(-p["E"] * t)
    H = np.zeros((2, 2, len(t)))
    H[0, 1] = -t * e
    H[1, 0] = H[0, 1]
    H[1, 1] = p["A"] * t ** 2 * e
    return H


model_single_exp.grad    = _grad_single_exp
model_single_exp.hessian = _hess_single_exp


# -- double-exponential -------------------------------------------------------

def model_double_exp(t, p):
    return p["A1"] * np.exp(-p["E1"] * t) + p["A2"] * np.exp(-p["E2"] * t)


def _grad_double_exp(t, p):
    """dC/d[A1, E1, A2, E2], shape (4, N)."""
    e1 = np.exp(-p["E1"] * t)
    e2 = np.exp(-p["E2"] * t)
    return np.array([e1, -p["A1"] * t * e1, e2, -p["A2"] * t * e2])


def _hess_double_exp(t, p):
    """
    d2C/(dtheta_i dtheta_j), shape (4, 4, N).

    Param order: [A1, E1, A2, E2].
    The two exponential terms are independent, so cross-blocks are zero.
      H[0,1] = H[1,0] = -t * exp(-E1*t)
      H[1,1] = A1 * t2 * exp(-E1*t)
      H[2,3] = H[3,2] = -t * exp(-E2*t)
      H[3,3] = A2 * t2 * exp(-E2*t)
    All others = 0.
    """
    e1 = np.exp(-p["E1"] * t)
    e2 = np.exp(-p["E2"] * t)
    H  = np.zeros((4, 4, len(t)))
    H[0, 1] = H[1, 0] = -t * e1
    H[1, 1] = p["A1"] * t ** 2 * e1
    H[2, 3] = H[3, 2] = -t * e2
    H[3, 3] = p["A2"] * t ** 2 * e2
    return H


model_double_exp.grad    = _grad_double_exp
model_double_exp.hessian = _hess_double_exp


# =============================================================================
# Helper: temporary attribute removal with guaranteed restore
# =============================================================================

class _temporarily_remove:
    """
    Context manager that removes a named attribute from *obj* for the duration
    of the ``with`` block and unconditionally restores it on exit, even if the
    block raises.

    Usage::

        with _temporarily_remove(model_single_exp, "hessian"):
            # model_single_exp.hessian does not exist here
            cost = build_cost(model_single_exp, ...)
            assert not hasattr(cost, "hess")
        # model_single_exp.hessian is guaranteed to be back here
    """

    def __init__(self, obj, attr: str):
        self._obj   = obj
        self._attr  = attr
        self._saved = None

    def __enter__(self):
        self._saved = getattr(self._obj, self._attr)
        delattr(self._obj, self._attr)
        return self

    def __exit__(self, *_):
        setattr(self._obj, self._attr, self._saved)


# =============================================================================
# Mock-data factory
# =============================================================================

def _make_bootstrap_data(true_values, noise, rng, nraw=NRAW, nbst=NBST):
    n   = len(true_values)
    cov = noise ** 2 * (np.eye(n) + 0.1 * np.ones((n, n)))
    raw = rng.multivariate_normal(mean=true_values, cov=cov, size=nraw)
    return Data(resample_type="bst", data=raw, Nresample=nbst)


# =============================================================================
# Fixtures
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
    true_C = DOUBLE_A1 * np.exp(-DOUBLE_E1 * t) + DOUBLE_A2 * np.exp(-DOUBLE_E2 * t)
    return t, _make_bootstrap_data(true_C, DOUBLE_NOISE, rng), {
        "A1": DOUBLE_A1, "E1": DOUBLE_E1,
        "A2": DOUBLE_A2, "E2": DOUBLE_E2,
    }


# =============================================================================
# Shared assertion helper
# =============================================================================

def _check_params_recovered(fit_result, true_params, central_value_fit, resample_fit, nsig=NSIG):
    for key, true_val in true_params.items():
        assert key in fit_result.params
        if resample_fit:
            rspl = fit_result.params[key].rspl
            assert not np.any(np.isnan(rspl))
            estimate    = fit_result.params[key].mean if central_value_fit else float(np.mean(rspl))
            uncertainty = np.std(rspl, ddof=1)
        else:
            estimate    = fit_result.params[key].mean
            uncertainty = fit_result.params_hessian_err[key].mean
        assert uncertainty > 0
        assert abs(estimate - true_val) < nsig * uncertainty, (
            f"'{key}': |{estimate:.4g} - {true_val:.4g}| >= {nsig} x {uncertainty:.4g}"
        )


# =============================================================================
# Fit-strategy matrix (same as all other iminuit test modules)
# =============================================================================

FIT_STRATEGIES = [
    ("cv_uncorr",   True,  False, False, False),
    ("cv_corr",     True,  True,  False, False),
    ("rs_uncorr",   False, False, True,  False),
    ("rs_corr",     False, False, True,  True ),
    ("both_uncorr", True,  False, True,  False),
    ("both_corr",   True,  True,  True,  True ),
]
_IDS    = [s[0] for s in FIT_STRATEGIES]
_PARAMS = [s[1:] for s in FIT_STRATEGIES]


# =============================================================================
# Standalone FD-Hessian helper (used outside class methods)
# =============================================================================

def _fd_hess_generic(cost, theta, rel_step=1e-5):
    """Central FD of cost.grad to form the Hessian matrix."""
    n = len(theta)
    H = np.zeros((n, n))
    for j in range(n):
        h       = rel_step * max(abs(theta[j]), 1.0)
        t_fwd   = theta.copy(); t_fwd[j] += h
        t_bwd   = theta.copy(); t_bwd[j] -= h
        H[:, j] = (np.asarray(cost.grad(*t_fwd)) - np.asarray(cost.grad(*t_bwd))) / (2.0 * h)
    return H


# =============================================================================
# 1. TestPriorHess
# =============================================================================

class TestPriorHess:
    """
    Verify Prior.hess against finite differences of Prior.grad.

    Normal prior:     hess = 2 / sdev^2   (constant)
    Log-normal prior: hess = 2*(1 + mean - log(theta)) / (sdev^2 * theta^2)
    """

    _eps = 1e-6

    def _fd_hess(self, prior, theta):
        """Central FD of prior.grad at theta."""
        h = self._eps * max(abs(theta), 1.0)
        return (prior.grad(theta + h) - prior.grad(theta - h)) / (2.0 * h)

    def test_normal_hess_formula(self):
        """Normal hess must equal 2/sdev^2."""
        p = Prior(mean=1.0, sdev=0.5)
        assert p.hess(2.0)  == pytest.approx(2.0 / 0.5 ** 2)
        assert p.hess(-3.0) == pytest.approx(2.0 / 0.5 ** 2)   # constant everywhere

    def test_normal_hess_matches_fd(self):
        """Normal hess must match finite-differences of grad."""
        p = Prior(mean=0.3, sdev=0.1)
        for theta in (0.1, 0.5, 1.0, 2.0):
            np.testing.assert_allclose(p.hess(theta), self._fd_hess(p, theta), rtol=1e-4)

    def test_normal_hess_positive(self):
        """Normal hess is always 2/sdev^2 > 0."""
        p = Prior(mean=0.0, sdev=2.0)
        assert p.hess(100.0) > 0

    def test_lognormal_hess_matches_fd(self):
        """Log-normal hess must match finite-differences of grad to 0.01%."""
        p = Prior(mean=np.log(0.3), sdev=0.5, dist="log-normal")
        for theta in (0.1, 0.3, 0.5, 1.0, 2.0):
            np.testing.assert_allclose(
                p.hess(theta), self._fd_hess(p, theta), rtol=1e-4,
                err_msg=f"log-normal hess mismatch at theta={theta}",
            )

    def test_lognormal_hess_zero_at_nonpositive(self):
        """For theta <= 0 hess must return 0 (consistent with grad)."""
        p = Prior(mean=np.log(0.3), sdev=0.5, dist="log-normal")
        assert p.hess(0.0)  == 0.0
        assert p.hess(-1.0) == 0.0

    def test_lognormal_hess_at_mean(self):
        """
        At theta = exp(mean) (the mode), log - mean = 0 so
        hess = 2 / (sdev^2 * theta^2).
        """
        mean, sdev = np.log(0.3), 0.5
        p     = Prior(mean=mean, sdev=sdev, dist="log-normal")
        theta = np.exp(mean)
        np.testing.assert_allclose(p.hess(theta), 2.0 / (sdev ** 2 * theta ** 2), rtol=1e-12)

    def test_lognormal_hess_can_be_negative(self):
        """
        When log(theta) - mean > 1 the curvature correction is negative --
        hess must report it rather than clamp to 0.
        """
        mean, sdev = 0.0, 0.1
        p = Prior(mean=mean, sdev=sdev, dist="log-normal")
        theta = np.exp(2.0)   # log(theta) - mean = 2 > 1
        assert p.hess(theta) < 0


# =============================================================================
# 2. TestCostHessAttachment
# =============================================================================

class TestCostHessAttachment:
    """
    Verify that ``cost.hess`` is attached / absent under the right conditions.

    Attributes are removed temporarily from the module-level model objects
    using _temporarily_remove, which guarantees restoration even on failure.
    """

    _rng   = np.random.default_rng(1)
    _raw   = _rng.normal(1.0, 0.02, size=(100, 10))
    _ord   = Data(resample_type="bst", data=_raw, Nresample=50)
    _t     = np.arange(1, 11, dtype=float)
    _y     = np.mean(_raw, axis=0)
    _sdev  = np.std(_raw, axis=0) / np.sqrt(100)

    def _build_uncorr(self, model):
        return _build_uncorrelated_cost(
            self._t, self._y, 1.0 / self._sdev,
            model, ["A", "E"],
            model_has_grad = hasattr(model, "grad"),
            model_has_hess = hasattr(model, "grad") and hasattr(model, "hessian"),
        )

    def _build_corr(self, model):
        cov_inv = np.diag(1.0 / self._sdev ** 2)
        return _build_correlated_cost(
            self._t, self._y, cov_inv,
            model, ["A", "E"],
            model_has_grad = hasattr(model, "grad"),
            model_has_hess = hasattr(model, "grad") and hasattr(model, "hessian"),
        )

    def test_hess_attached_when_both_grad_and_hessian_uncorr(self):
        """cost.hess must exist when model has both .grad and .hessian (uncorrelated)."""
        cost = self._build_uncorr(model_single_exp)
        assert hasattr(cost, "hess")

    def test_hess_attached_when_both_grad_and_hessian_corr(self):
        """cost.hess must exist when model has both .grad and .hessian (correlated)."""
        cost = self._build_corr(model_single_exp)
        assert hasattr(cost, "hess")

    def test_hess_absent_when_only_grad(self):
        """cost.hess must NOT be attached when model has .grad but not .hessian."""
        with _temporarily_remove(model_single_exp, "hessian"):
            cost = self._build_uncorr(model_single_exp)
            assert not hasattr(cost, "hess")
        assert hasattr(model_single_exp, "hessian"), "hessian not restored"

    def test_hess_absent_when_only_hessian(self):
        """cost.hess must NOT be attached when model has .hessian but not .grad."""
        with _temporarily_remove(model_single_exp, "grad"):
            cost = self._build_uncorr(model_single_exp)
            assert not hasattr(cost, "hess")
        assert hasattr(model_single_exp, "grad"), "grad not restored"

    def test_hess_absent_when_neither(self):
        """cost.hess must NOT be attached when model has neither .grad nor .hessian."""
        def bare_model(t, p):
            return p["A"] * np.exp(-p["E"] * t)
        cost = self._build_uncorr(bare_model)
        assert not hasattr(cost, "hess")

    def test_grad_still_attached_when_hessian_present(self):
        """cost.grad must still be attached alongside cost.hess."""
        cost = self._build_uncorr(model_single_exp)
        assert hasattr(cost, "grad")

    def test_hess_callable(self):
        """cost.hess must be callable and return a (Nparams, Nparams) array."""
        cost = self._build_uncorr(model_single_exp)
        H    = cost.hess(SINGLE_A, SINGLE_E)
        assert H.shape == (2, 2)
        assert np.all(np.isfinite(H))


# =============================================================================
# 3. TestCostHessCorrectness
# =============================================================================

class TestCostHessCorrectness:
    """
    ``cost.hess(*values)`` must agree with central finite differences of
    ``cost.grad(*values)`` for both uncorrelated and correlated paths.
    """

    _rng  = np.random.default_rng(2)
    _t    = np.arange(1, 11, dtype=float)
    _y    = SINGLE_A * np.exp(-SINGLE_E * _t)
    _serr = np.full(10, SINGLE_NOISE)

    def _fd_hess_from_grad(self, cost, theta, rel_step=1e-5):
        """Central FD of cost.grad to form the Hessian."""
        n = len(theta)
        H = np.zeros((n, n))
        for j in range(n):
            h       = rel_step * max(abs(theta[j]), 1.0)
            t_fwd   = theta.copy(); t_fwd[j] += h
            t_bwd   = theta.copy(); t_bwd[j] -= h
            H[:, j] = (np.asarray(cost.grad(*t_fwd)) - np.asarray(cost.grad(*t_bwd))) / (2.0 * h)
        return H

    def _build_uncorr(self, y=None, priors=None):
        y = self._y if y is None else y
        return _build_uncorrelated_cost(
            self._t, y, 1.0 / self._serr,
            model_single_exp, ["A", "E"],
            priors=priors, model_has_grad=True, model_has_hess=True,
        )

    def _build_corr(self, y=None, priors=None):
        y = self._y if y is None else y
        return _build_correlated_cost(
            self._t, y, np.diag(1.0 / self._serr ** 2),
            model_single_exp, ["A", "E"],
            priors=priors, model_has_grad=True, model_has_hess=True,
        )

    @pytest.mark.parametrize("path", ["uncorr", "corr"])
    def test_analytic_hess_vs_fd_at_truth(self, path):
        """Analytic hess must match FD of grad at the true parameters."""
        cost  = self._build_uncorr() if path == "uncorr" else self._build_corr()
        theta = np.array([SINGLE_A, SINGLE_E])
        np.testing.assert_allclose(
            np.asarray(cost.hess(*theta)), self._fd_hess_from_grad(cost, theta),
            rtol=1e-3, err_msg=f"Hess vs FD mismatch ({path}) at truth",
        )

    @pytest.mark.parametrize("path", ["uncorr", "corr"])
    def test_analytic_hess_vs_fd_with_residuals(self, path):
        """Analytic hess must match FD even when residuals are non-zero."""
        rng     = np.random.default_rng(99)
        y_noisy = self._y + rng.normal(0, SINGLE_NOISE * 5, size=self._t.shape)
        cost    = self._build_uncorr(y=y_noisy) if path == "uncorr" else self._build_corr(y=y_noisy)
        theta   = np.array([SINGLE_A * 0.8, SINGLE_E * 1.2])
        np.testing.assert_allclose(
            np.asarray(cost.hess(*theta)), self._fd_hess_from_grad(cost, theta),
            rtol=1e-3, err_msg=f"Hess vs FD mismatch ({path}) with residuals",
        )

    @pytest.mark.parametrize("path", ["uncorr", "corr"])
    def test_hess_symmetric(self, path):
        """Analytic Hessian of chi^2 must be symmetric."""
        cost  = self._build_uncorr() if path == "uncorr" else self._build_corr()
        theta = np.array([SINGLE_A, SINGLE_E])
        H     = np.asarray(cost.hess(*theta))
        np.testing.assert_allclose(H, H.T, atol=1e-14)

    def test_double_exp_hess_vs_fd_uncorr(self):
        """Analytic Hessian for the double-exponential model (uncorrelated)."""
        t    = np.arange(1, 13, dtype=float)
        y    = DOUBLE_A1 * np.exp(-DOUBLE_E1 * t) + DOUBLE_A2 * np.exp(-DOUBLE_E2 * t)
        serr = np.full(len(t), DOUBLE_NOISE)
        cost = _build_uncorrelated_cost(
            t, y, 1.0 / serr, model_double_exp, ["A1", "E1", "A2", "E2"],
            model_has_grad=True, model_has_hess=True,
        )
        theta = np.array([DOUBLE_A1, DOUBLE_E1, DOUBLE_A2, DOUBLE_E2])
        np.testing.assert_allclose(
            np.asarray(cost.hess(*theta)), _fd_hess_generic(cost, theta),
            rtol=1e-3, err_msg="Double-exp hess vs FD mismatch",
        )

    @pytest.mark.parametrize("path", ["uncorr", "corr"])
    def test_hess_with_normal_prior_vs_fd(self, path):
        """cost.hess must include the prior's 2/sdev^2 contribution on the diagonal."""
        prior = {"E": Prior(SINGLE_E, 0.5)}
        cost  = self._build_uncorr(priors=prior) if path == "uncorr" else self._build_corr(priors=prior)
        theta = np.array([SINGLE_A, SINGLE_E])
        np.testing.assert_allclose(
            np.asarray(cost.hess(*theta)), self._fd_hess_from_grad(cost, theta),
            rtol=1e-3, err_msg=f"Hess+prior vs FD mismatch ({path})",
        )

    @pytest.mark.parametrize("path", ["uncorr", "corr"])
    def test_hess_with_lognormal_prior_vs_fd(self, path):
        """Log-normal prior contribution to cost.hess must match FD."""
        prior = {"E": Prior(np.log(SINGLE_E), 0.5, dist="log-normal")}
        cost  = self._build_uncorr(priors=prior) if path == "uncorr" else self._build_corr(priors=prior)
        theta = np.array([SINGLE_A, SINGLE_E])
        np.testing.assert_allclose(
            np.asarray(cost.hess(*theta)), self._fd_hess_from_grad(cost, theta),
            rtol=1e-3, err_msg=f"Hess+log-normal prior vs FD mismatch ({path})",
        )


# =============================================================================
# 4. TestHessianParameterRecovery
# =============================================================================

class TestHessianParameterRecovery:
    """End-to-end parameter recovery with analytic Hessian over all six strategies."""

    _p0_single = {"A": SINGLE_A * 0.85, "E": SINGLE_E * 0.85}
    _p0_double = {
        "A1": DOUBLE_A1 * 0.90, "E1": DOUBLE_E1 * 0.90,
        "A2": DOUBLE_A2 * 0.90, "E2": DOUBLE_E2 * 0.90,
    }

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _PARAMS, ids=_IDS)
    def test_single_exp_recovery(self, single_exp_data, cv, cv_corr, rs, rs_corr):
        """Single-exp: A and E must be recovered within 3sigma with analytic Hessian."""
        t, ordinate, true_params = single_exp_data
        result = fit_iminuit(
            abscissa=t, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_single_exp, p0=self._p0_single,
        )
        _check_params_recovered(result, true_params, cv, rs)

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _PARAMS, ids=_IDS)
    def test_double_exp_recovery(self, double_exp_data, cv, cv_corr, rs, rs_corr):
        """Double-exp: all four params must be recovered within 3sigma."""
        t, ordinate, true_params = double_exp_data
        result = fit_iminuit(
            abscissa=t, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_double_exp, p0=self._p0_double,
        )
        _check_params_recovered(result, true_params, cv, rs)

    @pytest.mark.parametrize("cv, cv_corr, rs, rs_corr", _PARAMS, ids=_IDS)
    def test_hessian_errors_populated(self, single_exp_data, cv, cv_corr, rs, rs_corr):
        """params_hessian_err must be present and positive when analytic Hessian is used."""
        t, ordinate, _ = single_exp_data
        result = fit_iminuit(
            abscissa=t, ordinate=ordinate,
            central_value_fit=cv, central_value_fit_correlated=cv_corr,
            resample_fit=rs,      resample_fit_correlated=rs_corr,
            model=model_single_exp, p0=self._p0_single,
        )
        if cv:
            for key in ("A", "E"):
                assert result.params_hessian_err[key].mean > 0
        if rs:
            for key in ("A", "E"):
                assert np.all(result.params_hessian_err[key].rspl > 0)


# =============================================================================
# 5. TestHessianVsGradErrors
# =============================================================================

class TestHessianVsGradErrors:
    """
    The analytic-Hessian path and the grad-only (numerical hesse) path must
    produce consistent parameter errors.

    model.hessian is temporarily removed for the no-hessian run and
    unconditionally restored via _temporarily_remove.
    """

    _p0 = {"A": SINGLE_A * 0.85, "E": SINGLE_E * 0.85}

    def _run_with_hess(self, single_exp_data, cv_corr):
        t, ordinate, _ = single_exp_data
        return fit_iminuit(
            abscissa=t, ordinate=ordinate,
            central_value_fit=True, central_value_fit_correlated=cv_corr,
            resample_fit=False, model=model_single_exp, p0=self._p0,
        )

    def _run_without_hess(self, single_exp_data, cv_corr):
        t, ordinate, _ = single_exp_data
        with _temporarily_remove(model_single_exp, "hessian"):
            result = fit_iminuit(
                abscissa=t, ordinate=ordinate,
                central_value_fit=True, central_value_fit_correlated=cv_corr,
                resample_fit=False, model=model_single_exp, p0=self._p0,
            )
        return result

    @pytest.mark.parametrize("cv_corr", [False, True], ids=["uncorr", "corr"])
    def test_cv_errors_consistent_with_grad_only(self, single_exp_data, cv_corr):
        """Hessian-path errors must agree with grad-only errors within 5%."""
        r_hess = self._run_with_hess(single_exp_data, cv_corr)
        r_grad = self._run_without_hess(single_exp_data, cv_corr)
        for key in ("A", "E"):
            np.testing.assert_allclose(
                r_hess.params_hessian_err[key].mean,
                r_grad.params_hessian_err[key].mean,
                rtol=0.05,
                err_msg=f"{'Corr' if cv_corr else 'Uncorr'}: hessian err vs grad err for '{key}'",
            )

    @pytest.mark.parametrize("cv_corr", [False, True], ids=["uncorr", "corr"])
    def test_cv_params_identical_regardless_of_hessian(self, single_exp_data, cv_corr):
        """The Hessian only affects error estimation -- parameter values must be identical."""
        r_hess = self._run_with_hess(single_exp_data, cv_corr)
        r_grad = self._run_without_hess(single_exp_data, cv_corr)
        for key in ("A", "E"):
            np.testing.assert_allclose(
                r_hess.params[key].mean, r_grad.params[key].mean, rtol=1e-8,
                err_msg=f"CV param changed between hessian/no-hessian paths for '{key}'",
            )

    def test_model_hessian_restored_after_runs(self):
        """Sanity check: model_single_exp.hessian must still exist after all _run calls."""
        assert hasattr(model_single_exp, "hessian"), (
            "model_single_exp.hessian was not restored after a _temporarily_remove block"
        )


# =============================================================================
# 6. TestHessianWithPriors
# =============================================================================

class TestHessianWithPriors:
    """Priors contribute to cost.hess on the diagonal."""

    def test_single_exp_normal_prior_recovery(self, single_exp_data):
        """Normal prior on E: recovery must succeed with analytic Hessian."""
        t, ordinate, true_params = single_exp_data
        result = fit_iminuit(
            abscissa=t, ordinate=ordinate,
            central_value_fit=True, resample_fit=False,
            model=model_single_exp,
            prior={"A": Prior(SINGLE_A, 5.0), "E": Prior(SINGLE_E, 5.0)},
        )
        _check_params_recovered(result, true_params, True, False)

    def test_single_exp_lognormal_prior_recovery(self, single_exp_data):
        """Log-normal prior on E with positivity limit: recovery within 3sigma."""
        t, ordinate, true_params = single_exp_data
        result = fit_iminuit(
            abscissa=t, ordinate=ordinate,
            central_value_fit=True, resample_fit=False,
            model=model_single_exp,
            prior={"A": Prior(SINGLE_A, 5.0),
                   "E": Prior(np.log(SINGLE_E), 1.0, dist="log-normal")},
            limits={"E": (0.0, None)},
        )
        _check_params_recovered(result, true_params, True, False)

    def test_prior_contribution_visible_in_hess(self, single_exp_data):
        """
        A tight prior on E must add exactly 2/sdev^2 to H[1,1] of cost.hess.
        """
        t, ordinate, _ = single_exp_data
        y, serr = ordinate.mean, ordinate.serr

        cost_no = _build_uncorrelated_cost(
            t, y, 1.0 / serr, model_single_exp, ["A", "E"],
            model_has_grad=True, model_has_hess=True,
        )
        cost_pr = _build_uncorrelated_cost(
            t, y, 1.0 / serr, model_single_exp, ["A", "E"],
            priors={"E": Prior(SINGLE_E, 0.1)},
            model_has_grad=True, model_has_hess=True,
        )
        theta = np.array([SINGLE_A, SINGLE_E])
        H_no  = np.asarray(cost_no.hess(*theta))
        H_pr  = np.asarray(cost_pr.hess(*theta))

        assert H_pr[1, 1] > H_no[1, 1], "Prior on E should increase H[1,1]"
        np.testing.assert_allclose(H_pr[1, 1] - H_no[1, 1], 2.0 / 0.1 ** 2, rtol=1e-10)


# =============================================================================
# 7. TestCostGradAttachment
# =============================================================================

class TestCostGradAttachment:
    """
    Verify that ``cost.grad`` is attached or absent under the right conditions,
    for both uncorrelated and correlated cost builders.

    Rules:
    * model.grad present  -> cost.grad attached  (regardless of model.hessian)
    * model.grad absent   -> cost.grad absent    (even if model.hessian present)

    Attributes are removed temporarily from the module-level model objects
    using _temporarily_remove, which guarantees restoration even on failure.
    """

    _t       = np.arange(1, 11, dtype=float)
    _y       = SINGLE_A * np.exp(-SINGLE_E * _t)
    _serr    = np.full(10, SINGLE_NOISE)
    _cov_inv = np.diag(1.0 / np.full(10, SINGLE_NOISE) ** 2)

    def _build(self, model, correlated):
        has_grad = hasattr(model, "grad")
        has_hess = has_grad and hasattr(model, "hessian")
        if correlated:
            return _build_correlated_cost(
                self._t, self._y, self._cov_inv, model, ["A", "E"],
                model_has_grad=has_grad, model_has_hess=has_hess,
            )
        return _build_uncorrelated_cost(
            self._t, self._y, 1.0 / self._serr, model, ["A", "E"],
            model_has_grad=has_grad, model_has_hess=has_hess,
        )

    @pytest.mark.parametrize("correlated", [False, True], ids=["uncorr", "corr"])
    def test_grad_attached_with_grad_only(self, correlated):
        """cost.grad must be attached when model has .grad but not .hessian."""
        with _temporarily_remove(model_single_exp, "hessian"):
            assert hasattr(model_single_exp, "grad") and not hasattr(model_single_exp, "hessian")
            cost = self._build(model_single_exp, correlated)
            assert hasattr(cost, "grad")
        assert hasattr(model_single_exp, "hessian"), "hessian not restored"

    @pytest.mark.parametrize("correlated", [False, True], ids=["uncorr", "corr"])
    def test_grad_attached_with_grad_and_hessian(self, correlated):
        """cost.grad must be attached when model has both .grad and .hessian."""
        cost = self._build(model_single_exp, correlated)
        assert hasattr(cost, "grad")

    @pytest.mark.parametrize("correlated", [False, True], ids=["uncorr", "corr"])
    def test_grad_absent_without_grad(self, correlated):
        """cost.grad must NOT be attached when model lacks .grad."""
        def bare(t, p):
            return p["A"] * np.exp(-p["E"] * t)
        cost = self._build(bare, correlated)
        assert not hasattr(cost, "grad")

    @pytest.mark.parametrize("correlated", [False, True], ids=["uncorr", "corr"])
    def test_grad_absent_when_only_hessian_present(self, correlated):
        """cost.grad must NOT be attached when model has .hessian but not .grad."""
        with _temporarily_remove(model_single_exp, "grad"):
            assert not hasattr(model_single_exp, "grad") and hasattr(model_single_exp, "hessian")
            cost = self._build(model_single_exp, correlated)
            assert not hasattr(cost, "grad")
        assert hasattr(model_single_exp, "grad"), "grad not restored"

    @pytest.mark.parametrize("correlated", [False, True], ids=["uncorr", "corr"])
    def test_grad_callable_returns_correct_shape(self, correlated):
        """cost.grad must be callable and return shape (Nparams,)."""
        cost = self._build(model_single_exp, correlated)
        g    = np.asarray(cost.grad(SINGLE_A, SINGLE_E))
        assert g.shape == (2,)
        assert np.all(np.isfinite(g))

    @pytest.mark.parametrize("correlated", [False, True], ids=["uncorr", "corr"])
    def test_double_exp_grad_shape(self, correlated):
        """Double-exp cost.grad must return shape (4,)."""
        t    = np.arange(1, 13, dtype=float)
        y    = DOUBLE_A1 * np.exp(-DOUBLE_E1 * t) + DOUBLE_A2 * np.exp(-DOUBLE_E2 * t)
        serr = np.full(12, DOUBLE_NOISE)
        has_grad = hasattr(model_double_exp, "grad")
        has_hess = has_grad and hasattr(model_double_exp, "hessian")
        if correlated:
            cost = _build_correlated_cost(
                t, y, np.diag(1.0 / serr ** 2),
                model_double_exp, ["A1", "E1", "A2", "E2"],
                model_has_grad=has_grad, model_has_hess=has_hess,
            )
        else:
            cost = _build_uncorrelated_cost(
                t, y, 1.0 / serr,
                model_double_exp, ["A1", "E1", "A2", "E2"],
                model_has_grad=has_grad, model_has_hess=has_hess,
            )
        g = np.asarray(cost.grad(DOUBLE_A1, DOUBLE_E1, DOUBLE_A2, DOUBLE_E2))
        assert g.shape == (4,)
        assert np.all(np.isfinite(g))


# =============================================================================
# 8. TestCostGradCorrectness
# =============================================================================

class TestCostGradCorrectness:
    """
    ``cost.grad(*values)`` must agree with central finite differences of
    ``cost(*values)`` for all model/path/prior combinations.

    FD is taken of the scalar cost -- entirely independent of the gradient
    formula -- so this is a true correctness check.

    The ``with_hessian`` parameter in ``_build_single`` controls the
    ``model_has_hess`` flag passed to the builder; it never touches the model
    object itself, so no save/restore is needed there.
    """

    _rel_step = 1e-5

    def _fd_grad(self, cost, theta):
        """Central FD gradient of scalar cost at theta."""
        g = np.empty(len(theta))
        for i in range(len(theta)):
            h       = self._rel_step * max(abs(theta[i]), 1.0)
            t_fwd   = theta.copy(); t_fwd[i] += h
            t_bwd   = theta.copy(); t_bwd[i] -= h
            g[i]    = (cost(*t_fwd) - cost(*t_bwd)) / (2.0 * h)
        return g

    def _build_single(self, correlated, y=None, priors=None, with_hessian=True):
        """
        Build a single-exp cost function.  with_hessian only controls the
        model_has_hess builder flag -- the model object is never mutated.
        """
        t    = np.arange(1, 11, dtype=float)
        y    = (SINGLE_A * np.exp(-SINGLE_E * t)) if y is None else y
        serr = np.full(10, SINGLE_NOISE)
        # model_single_exp always has both .grad and .hessian;
        # model_has_hess=with_hessian controls whether cost.hess is attached.
        if correlated:
            return _build_correlated_cost(
                t, y, np.diag(1.0 / serr ** 2), model_single_exp, ["A", "E"],
                priors=priors, model_has_grad=True, model_has_hess=with_hessian,
            )
        return _build_uncorrelated_cost(
            t, y, 1.0 / serr, model_single_exp, ["A", "E"],
            priors=priors, model_has_grad=True, model_has_hess=with_hessian,
        )

    @pytest.mark.parametrize("correlated", [False, True], ids=["uncorr", "corr"])
    def test_single_exp_grad_vs_fd_at_truth(self, correlated):
        """Analytic grad must match FD of cost at the true parameters (< 0.1% rel err)."""
        cost  = self._build_single(correlated)
        theta = np.array([SINGLE_A, SINGLE_E])
        np.testing.assert_allclose(
            np.asarray(cost.grad(*theta)), self._fd_grad(cost, theta), atol = 1e-3, rtol=1e-3,
            err_msg=f"grad vs FD at truth ({'corr' if correlated else 'uncorr'})",
        )

    @pytest.mark.parametrize("correlated", [False, True], ids=["uncorr", "corr"])
    def test_single_exp_grad_vs_fd_displaced(self, correlated):
        """Analytic grad must match FD at a displaced parameter point."""
        cost  = self._build_single(correlated)
        theta = np.array([SINGLE_A * 0.7, SINGLE_E * 1.4])
        np.testing.assert_allclose(
            np.asarray(cost.grad(*theta)), self._fd_grad(cost, theta), rtol=1e-3,
            err_msg=f"grad vs FD displaced ({'corr' if correlated else 'uncorr'})",
        )

    @pytest.mark.parametrize("correlated", [False, True], ids=["uncorr", "corr"])
    def test_single_exp_grad_vs_fd_noisy_data(self, correlated):
        """With non-zero residuals at truth the grad must still match FD."""
        rng   = np.random.default_rng(77)
        t     = np.arange(1, 11, dtype=float)
        y     = SINGLE_A * np.exp(-SINGLE_E * t) + rng.normal(0, SINGLE_NOISE * 3, 10)
        cost  = self._build_single(correlated, y=y)
        theta = np.array([SINGLE_A, SINGLE_E])
        np.testing.assert_allclose(
            np.asarray(cost.grad(*theta)), self._fd_grad(cost, theta), rtol=1e-3,
            err_msg="grad vs FD with noisy data",
        )

    @pytest.mark.parametrize("correlated", [False, True], ids=["uncorr", "corr"])
    def test_grad_near_zero_at_noiseless_minimum(self, correlated):
        """At the noiseless minimum the gradient must be numerically ~0."""
        cost  = self._build_single(correlated)
        theta = np.array([SINGLE_A, SINGLE_E])
        g     = np.asarray(cost.grad(*theta))
        chi2  = cost(*theta)
        assert np.all(np.abs(g) < 1e-8 * max(chi2, 1.0)), (
            f"Gradient at noiseless minimum should be ~0, got {g}"
        )

    @pytest.mark.parametrize("correlated", [False, True], ids=["uncorr", "corr"])
    def test_double_exp_grad_vs_fd(self, correlated):
        """Double-exp analytic grad must match FD for all four parameters."""
        t    = np.arange(1, 13, dtype=float)
        y    = DOUBLE_A1 * np.exp(-DOUBLE_E1 * t) + DOUBLE_A2 * np.exp(-DOUBLE_E2 * t)
        serr = np.full(12, DOUBLE_NOISE)
        if correlated:
            cost = _build_correlated_cost(
                t, y, np.diag(1.0 / serr ** 2), model_double_exp,
                ["A1", "E1", "A2", "E2"], model_has_grad=True, model_has_hess=True,
            )
        else:
            cost = _build_uncorrelated_cost(
                t, y, 1.0 / serr, model_double_exp,
                ["A1", "E1", "A2", "E2"], model_has_grad=True, model_has_hess=True,
            )
        theta = np.array([DOUBLE_A1, DOUBLE_E1, DOUBLE_A2, DOUBLE_E2])
        np.testing.assert_allclose(
            np.asarray(cost.grad(*theta)), self._fd_grad(cost, theta), atol = 1e-3, rtol=1e-3,
            err_msg=f"double-exp grad vs FD ({'corr' if correlated else 'uncorr'})",
        )

    @pytest.mark.parametrize("correlated", [False, True], ids=["uncorr", "corr"])
    def test_grad_with_normal_prior_vs_fd(self, correlated):
        """Normal prior adds 2*(E-mean)/sdev^2 to grad[1]; total must match FD."""
        prior = {"E": Prior(SINGLE_E, 0.5)}
        cost  = self._build_single(correlated, priors=prior)
        theta = np.array([SINGLE_A, SINGLE_E * 1.2])
        np.testing.assert_allclose(
            np.asarray(cost.grad(*theta)), self._fd_grad(cost, theta), rtol=1e-3,
            err_msg=f"grad+normal prior vs FD ({'corr' if correlated else 'uncorr'})",
        )

    @pytest.mark.parametrize("correlated", [False, True], ids=["uncorr", "corr"])
    def test_grad_with_lognormal_prior_vs_fd(self, correlated):
        """Log-normal prior contribution to grad must match FD."""
        prior = {"E": Prior(np.log(SINGLE_E), 0.5, dist="log-normal")}
        cost  = self._build_single(correlated, priors=prior)
        theta = np.array([SINGLE_A, SINGLE_E * 1.3])
        np.testing.assert_allclose(
            np.asarray(cost.grad(*theta)), self._fd_grad(cost, theta), rtol=1e-3,
            err_msg=f"grad+lognormal prior vs FD ({'corr' if correlated else 'uncorr'})",
        )

    @pytest.mark.parametrize("correlated", [False, True], ids=["uncorr", "corr"])
    def test_grad_identical_with_and_without_hessian(self, correlated):
        """
        cost.grad must return bit-for-bit identical values regardless of whether
        model_has_hess is True or False -- the hessian closure must not alter the
        gradient closure in any way.
        """
        cost_with = self._build_single(correlated, with_hessian=True)
        cost_sans = self._build_single(correlated, with_hessian=False)
        theta = np.array([SINGLE_A, SINGLE_E])
        np.testing.assert_array_equal(
            np.asarray(cost_with.grad(*theta)),
            np.asarray(cost_sans.grad(*theta)),
            err_msg="cost.grad differs depending on model_has_hess flag",
        )

    def test_prior_gradient_contribution_is_additive(self):
        """
        Prior adds exactly [prior.grad(A), prior.grad(E)] to the data-term
        gradient -- verified to rtol=1e-12.
        """
        t     = np.arange(1, 11, dtype=float)
        y     = SINGLE_A * np.exp(-SINGLE_E * t)
        serr  = np.full(10, SINGLE_NOISE)
        prior = {"A": Prior(SINGLE_A, 1.0), "E": Prior(SINGLE_E, 0.5)}

        cost_no = _build_uncorrelated_cost(
            t, y, 1.0 / serr, model_single_exp, ["A", "E"],
            model_has_grad=True, model_has_hess=True,
        )
        cost_pr = _build_uncorrelated_cost(
            t, y, 1.0 / serr, model_single_exp, ["A", "E"],
            priors=prior, model_has_grad=True, model_has_hess=True,
        )
        theta = np.array([SINGLE_A * 1.1, SINGLE_E * 0.9])
        g_no  = np.asarray(cost_no.grad(*theta))
        g_pr  = np.asarray(cost_pr.grad(*theta))
        expected = np.array([prior["A"].grad(theta[0]), prior["E"].grad(theta[1])])
        np.testing.assert_allclose(
            g_pr - g_no, expected, rtol=1e-12,
            err_msg="Prior contribution to cost.grad is not purely additive",
        )