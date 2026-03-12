"""
Unit tests for the linear regression backend (fit_linearRegression.py /
fit.py with backend='linear regression').

Overview
--------
1. Input-validation tests — ordinate type, abscissa shape, flag logic,
   parameter_names count validation.

2. End-to-end parameter recovery:
     2a. Linear model with intercept   y(x) = m·x + b
     2b. Through-origin model          y(x) = m·x   (has_intercept=False)
   Both models are exercised over the full strategy matrix:
     (cv/resample) × (correlated/uncorrelated).

3. Custom parameter names — verifying that the name mapping between the
   design matrix and FitResult is correct.

4. Analytical error checks — Hessian errors (= closed-form propagated errors)
   must be available for ALL parameters (slope and intercept alike), not just
   nonlinear ones.  This distinguishes the linear regression backend from the
   variable-projection backend.

5. dof consistency — dof = Ndata - Nparams analytically.

Differences from the iminuit / lsqfit backends
-----------------------------------------------
* No `prior` or `p0` arguments — the solution is fully analytical.
* No `Nproc` / parallel support — the closed-form solve is O(N·p²) and
  inherently vectorisable across resamples without a process pool.
* Hessian errors are available for ALL parameters from the closed-form
  covariance C = (X^T W X)^{-1}, including the intercept — unlike
  variable projection where only nonlinear parameters have Hessian errors.
* No `limits` or `maxiter` arguments.
* The linear model test suite therefore has no parallel-execution or prior
  test classes.
"""

from __future__ import annotations

import numpy as np
import pytest

from correlatoranalyser import Data
from correlatoranalyser.fit_linear_regression import linear_regression

# =============================================================================
# Global test configuration
# =============================================================================

_RNG_SEED = 20240101
NRAW: int  = 600
NBST: int  = 300
NSIG: int  = 3

# =============================================================================
# Ground-truth parameters
# =============================================================================

# Model with intercept:  y = m·x + b
LINEAR_M:     float = 2.5
LINEAR_B:     float = 0.7
LINEAR_NOISE: float = 0.10

# Through-origin model:  y = m·x   (b = 0 by construction)
ORIGIN_M:     float = 3.2
ORIGIN_NOISE: float = 0.12

# =============================================================================
# Mock-data factory  (identical structure to all previous test modules)
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

    Covariance = noise_scale² · (I + 0.1 · 11ᵀ).  The mild correlations
    make the correlated fit path genuinely different from the uncorrelated
    one while still allowing both to recover the same true values.
    """
    n   = len(true_values)
    cov = noise_scale ** 2 * (np.eye(n) + 0.1 * np.ones((n, n)))
    raw = rng.multivariate_normal(mean=true_values, cov=cov, size=nraw)
    return Data(resample_type="bst", data=raw, Nresample=nbst)


# =============================================================================
# Module-scoped fixtures
# =============================================================================

@pytest.fixture(scope="module")
def linear_data():
    """
    8-point data for y = m·x + b with intercept.

    Seed offset +30 keeps this independent of all previous test modules.
    x-values are on [0, 1] to give a well-conditioned 8×8 covariance.
    """
    rng    = np.random.default_rng(_RNG_SEED + 30)
    x      = np.linspace(0.0, 1.0, 8)
    true_y = LINEAR_M * x + LINEAR_B
    ordinate = _make_bootstrap_data(true_y, LINEAR_NOISE, rng)
    return x, ordinate, {"m": LINEAR_M, "b": LINEAR_B}


@pytest.fixture(scope="module")
def origin_data():
    """
    8-point data for y = m·x through the origin (no intercept).

    Using b_true = 0 by construction so that a through-origin fit is
    correctly specified.  x starts at 1 (not 0) to avoid a zero column in
    the uncorrelated weight matrix while keeping the problem well-conditioned.
    """
    rng    = np.random.default_rng(_RNG_SEED + 31)
    x      = np.linspace(1.0, 2.0, 8)
    true_y = ORIGIN_M * x           # no intercept term
    ordinate = _make_bootstrap_data(true_y, ORIGIN_NOISE, rng)
    return x, ordinate, {"m": ORIGIN_M}


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
    Assert that every parameter lies within nsig σ of its true value.

    Key difference from the iminuit / varproj helpers: the linear regression
    backend produces closed-form Hessian errors for ALL parameters (slope and
    intercept), so we always have an uncertainty estimate for the CV-only path.
    No fallback 5%-tolerance is needed.
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
            # Closed-form Hessian error is available for all parameters.
            assert key in fit_result.params_hessian_err, (
                f"Hessian error missing for '{key}' after CV-only fit"
            )
            estimate    = fit_result.params[key].mean
            uncertainty = fit_result.params_hessian_err[key].mean

        assert uncertainty > 0, f"Uncertainty for '{key}' is non-positive: {uncertainty}"
        deviation = abs(estimate - true_val)
        assert deviation < nsig * uncertainty, (
            f"Parameter '{key}': |{estimate:.6g} - {true_val:.6g}| = {deviation:.3g} "
            f">= {nsig} × {uncertainty:.3g}.  Fit is more than {nsig}σ from truth."
        )


# =============================================================================
# Fit-strategy parametrisation
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
    Confirm that linear_regression raises informative errors for bad inputs.

    Unlike the iminuit/lsqfit backends, linear_regression performs its own
    validation (it does not call _validate_inputs because there is no model
    function to validate).  The expected error types are otherwise the same.
    """

    _rng      = np.random.default_rng(2)
    _raw      = _rng.normal(1.0, 0.05, size=(100, 8))
    _ordinate = Data(resample_type="bst", data=_raw, Nresample=50)
    _x        = np.linspace(0.0, 1.0, 8)

    def test_ordinate_must_be_Data_not_ndarray(self):
        """Plain numpy array ordinate must raise TypeError."""
        with pytest.raises(TypeError):
            linear_regression(
                abscissa=self._x,
                ordinate=np.ones(8),
            )

    def test_ordinate_must_be_Data_not_list(self):
        """Python list ordinate must raise TypeError."""
        with pytest.raises(TypeError):
            linear_regression(
                abscissa=self._x,
                ordinate=list(range(8)),
            )

    def test_ordinate_must_be_Data_not_scalar(self):
        """Scalar ordinate must raise TypeError."""
        with pytest.raises(TypeError):
            linear_regression(
                abscissa=self._x,
                ordinate=3.14,
            )

    def test_abscissa_scalar_raises(self):
        """Scalar abscissa has no shape[0]; some exception must propagate."""
        with pytest.raises(Exception):
            linear_regression(
                abscissa=42,
                ordinate=self._ordinate,
            )

    def test_abscissa_none_raises(self):
        """None abscissa must raise an exception."""
        with pytest.raises(Exception):
            linear_regression(
                abscissa=None,
                ordinate=self._ordinate,
            )

    def test_neither_cv_nor_resample_raises(self):
        """Both fit flags False must raise ValueError immediately."""
        with pytest.raises(ValueError, match="At least one of"):
            linear_regression(
                abscissa=self._x,
                ordinate=self._ordinate,
                central_value_fit=False,
                resample_fit=False,
            )

    def test_wrong_parameter_names_count_with_intercept(self):
        """
        has_intercept=True requires exactly 2 parameter names.
        Passing 1 or 3 must raise ValueError.
        """
        with pytest.raises(ValueError):
            linear_regression(
                abscissa=self._x,
                ordinate=self._ordinate,
                has_intercept=True,
                parameter_names=("m",),   # only 1 — should be 2
            )
        with pytest.raises(ValueError):
            linear_regression(
                abscissa=self._x,
                ordinate=self._ordinate,
                has_intercept=True,
                parameter_names=("m", "b", "extra"),   # 3 — should be 2
            )

    def test_wrong_parameter_names_count_no_intercept(self):
        """
        has_intercept=False requires exactly 1 parameter name.
        Passing 2 must raise ValueError.
        """
        with pytest.raises(ValueError):
            linear_regression(
                abscissa=self._x,
                ordinate=self._ordinate,
                has_intercept=False,
                parameter_names=("m", "b"),   # 2 — should be 1
            )

    def test_abscissa_shape_mismatch_raises(self):
        """
        Abscissa and ordinate must have the same length along the data axis.
        A 5-point abscissa with an 8-point ordinate must raise ValueError.
        """
        with pytest.raises(ValueError, match="shapes must match"):
            linear_regression(
                abscissa=np.linspace(0.0, 1.0, 5),   # 5 points
                ordinate=self._ordinate,               # 8 observables
            )


# =============================================================================
# 2a. End-to-end tests – linear model with intercept
# =============================================================================

class TestLinearWithIntercept:
    """
    Fit y = m·x + b to synthetic bootstrap data.

    This is the standard linear regression with two free parameters.  The
    closed-form weighted least-squares solution should recover the true m and b
    within 3σ for all strategy combinations.
    """

    @pytest.mark.parametrize(
        "cv, cv_corr, rs, rs_corr",
        _STRATEGY_PARAMS,
        ids=_STRATEGY_IDS,
    )
    def test_parameter_recovery(self, linear_data, cv, cv_corr, rs, rs_corr):
        """
        m and b must be recovered within 3σ for every CV/resample×corr strategy.

        The correlated path uses the full 8×8 bootstrap covariance as the
        weight matrix; the uncorrelated path uses only the diagonal.  Both
        are consistent estimators and should recover the same true values.
        """
        x, ordinate, true_params = linear_data
        result = linear_regression(
            abscissa=x,
            ordinate=ordinate,
            central_value_fit=cv,
            central_value_fit_correlated=cv_corr,
            resample_fit=rs,
            resample_fit_correlated=rs_corr,
            has_intercept=True,
        )
        _check_params_recovered(result, true_params, central_value_fit=cv, resample_fit=rs)

    @pytest.mark.parametrize(
        "cv, cv_corr, rs, rs_corr",
        _STRATEGY_PARAMS,
        ids=_STRATEGY_IDS,
    )
    def test_hessian_errors_all_params(self, linear_data, cv, cv_corr, rs, rs_corr):
        """
        Closed-form Hessian errors must be populated for BOTH m and b.

        This is a key difference from the variable-projection backend: because
        the analytical covariance C = (X^T W X)^{-1} is computed unconditionally,
        every parameter has a propagated error estimate regardless of whether it
        was the 'slope' or 'intercept'.
        """
        x, ordinate, _ = linear_data
        result = linear_regression(
            abscissa=x,
            ordinate=ordinate,
            central_value_fit=cv,
            central_value_fit_correlated=cv_corr,
            resample_fit=rs,
            resample_fit_correlated=rs_corr,
            has_intercept=True,
        )
        if cv:
            for key in ("m", "b"):
                assert key in result.params_hessian_err, (
                    f"Hessian error missing for '{key}'"
                )
                assert result.params_hessian_err[key].mean > 0, (
                    f"Hessian error for '{key}' is non-positive"
                )
        if rs:
            for key in ("m", "b"):
                assert key in result.params_hessian_err
                assert np.all(result.params_hessian_err[key].rspl > 0)

    @pytest.mark.parametrize(
        "cv, cv_corr, rs, rs_corr",
        _STRATEGY_PARAMS,
        ids=_STRATEGY_IDS,
    )
    def test_fit_quality_metadata(self, linear_data, cv, cv_corr, rs, rs_corr):
        """
        dof must equal Ndata - 2 (8 points, 2 parameters).
        chi² must be positive and finite.
        """
        x, ordinate, _ = linear_data
        result = linear_regression(
            abscissa=x,
            ordinate=ordinate,
            central_value_fit=cv,
            central_value_fit_correlated=cv_corr,
            resample_fit=rs,
            resample_fit_correlated=rs_corr,
            has_intercept=True,
        )
        assert result.dof == len(x) - 2, (
            f"Expected dof = {len(x) - 2}, got {result.dof}"
        )
        if cv:
            chi2_val = result.chi2.mean if hasattr(result.chi2, "mean") else float(result.chi2)
            assert chi2_val > 0
        if rs:
            assert np.all(np.isfinite(result.chi2.rspl))
            assert np.all(result.chi2.rspl > 0)

    @pytest.mark.parametrize(
        "cv, cv_corr, rs, rs_corr",
        _STRATEGY_PARAMS,
        ids=_STRATEGY_IDS,
    )
    def test_resample_count_and_no_nan(self, linear_data, cv, cv_corr, rs, rs_corr):
        """Every resample slot for m and b must be populated and finite."""
        x, ordinate, _ = linear_data
        result = linear_regression(
            abscissa=x,
            ordinate=ordinate,
            central_value_fit=cv,
            central_value_fit_correlated=cv_corr,
            resample_fit=rs,
            resample_fit_correlated=rs_corr,
            has_intercept=True,
        )
        if rs:
            for key in ("m", "b"):
                assert result.params[key].rspl.shape == (NBST,)
                assert not np.any(np.isnan(result.params[key].rspl))


# =============================================================================
# 2b. End-to-end tests – through-origin model (has_intercept=False)
# =============================================================================

class TestLinearNoIntercept:
    """
    Fit y = m·x through the origin (no intercept term).

    Setting has_intercept=False changes the design matrix from [x | 1] to [x],
    reducing the model to a single free parameter.  This is the correct model
    when physical constraints require the signal to vanish at x=0 (e.g. a form
    factor normalised at zero momentum transfer).

    The mock data are generated with b_true = 0 exactly so that the constrained
    fit is correctly specified.  dof = Ndata - 1 in this case.
    """

    @pytest.mark.parametrize(
        "cv, cv_corr, rs, rs_corr",
        _STRATEGY_PARAMS,
        ids=_STRATEGY_IDS,
    )
    def test_parameter_recovery(self, origin_data, cv, cv_corr, rs, rs_corr):
        """
        The slope m must be recovered within 3σ of ORIGIN_M.

        Note that fitting a through-origin model to data generated with b=0 is
        the correctly specified model.  Fitting the same data with has_intercept=True
        would give b ≈ 0 but waste a degree of freedom.
        """
        x, ordinate, true_params = origin_data
        result = linear_regression(
            abscissa=x,
            ordinate=ordinate,
            central_value_fit=cv,
            central_value_fit_correlated=cv_corr,
            resample_fit=rs,
            resample_fit_correlated=rs_corr,
            has_intercept=False,
        )
        _check_params_recovered(result, true_params, central_value_fit=cv, resample_fit=rs)

    @pytest.mark.parametrize(
        "cv, cv_corr, rs, rs_corr",
        _STRATEGY_PARAMS,
        ids=_STRATEGY_IDS,
    )
    def test_only_slope_in_result(self, origin_data, cv, cv_corr, rs, rs_corr):
        """
        With has_intercept=False there must be exactly one parameter ("m") in
        FitResult.params.  The intercept key "b" must be absent.
        """
        x, ordinate, _ = origin_data
        result = linear_regression(
            abscissa=x,
            ordinate=ordinate,
            central_value_fit=cv,
            central_value_fit_correlated=cv_corr,
            resample_fit=rs,
            resample_fit_correlated=rs_corr,
            has_intercept=False,
        )
        assert "m" in result.params, "Slope 'm' missing from FitResult.params"
        assert "b" not in result.params, (
            "'b' should not be present in a no-intercept fit"
        )

    @pytest.mark.parametrize(
        "cv, cv_corr, rs, rs_corr",
        _STRATEGY_PARAMS,
        ids=_STRATEGY_IDS,
    )
    def test_dof_no_intercept(self, origin_data, cv, cv_corr, rs, rs_corr):
        """
        dof = Ndata - 1 for the through-origin model (8 data points, 1 parameter).
        """
        x, ordinate, _ = origin_data
        result = linear_regression(
            abscissa=x,
            ordinate=ordinate,
            central_value_fit=cv,
            central_value_fit_correlated=cv_corr,
            resample_fit=rs,
            resample_fit_correlated=rs_corr,
            has_intercept=False,
        )
        assert result.dof == len(x) - 1, (
            f"Expected dof = {len(x) - 1} for no-intercept, got {result.dof}"
        )

    def test_no_intercept_gives_higher_dof_than_with_intercept(self, origin_data):
        """
        A through-origin fit must have one more degree of freedom than the same
        fit with an intercept, because one fewer parameter is estimated.

        This verifies that the design matrix dimension (and therefore the dof
        formula) is correctly controlled by has_intercept.
        """
        x, ordinate, _ = origin_data

        result_with    = linear_regression(abscissa=x, ordinate=ordinate, has_intercept=True)
        result_without = linear_regression(abscissa=x, ordinate=ordinate, has_intercept=False)

        assert result_without.dof == result_with.dof + 1, (
            f"No-intercept dof ({result_without.dof}) should be "
            f"intercept dof ({result_with.dof}) + 1"
        )

    def test_misspecified_intercept_fit_recovers_zero_intercept(self, origin_data):
        """
        Fitting the through-origin data with has_intercept=True (misspecified model)
        must recover an intercept consistent with zero within 3σ.

        This is a statistical sanity check: if the true b=0, the estimated b̂
        must be statistically indistinguishable from zero when the model is
        overparameterised.
        """
        x, ordinate, _ = origin_data
        result = linear_regression(
            abscissa=x,
            ordinate=ordinate,
            has_intercept=True,
            central_value_fit=True,
            resample_fit=False,
        )
        b_hat = result.params["b"].mean
        b_err = result.params_hessian_err["b"].mean
        assert abs(b_hat) < 3 * b_err, (
            f"Intercept b̂={b_hat:.4g} is more than 3σ={3*b_err:.4g} from zero "
            f"for data generated with b=0."
        )


# =============================================================================
# 3. Custom parameter names
# =============================================================================

class TestCustomParameterNames:
    """
    Verify that custom parameter_names are correctly forwarded to FitResult.

    The default names ("m", "b") / ("m",) are convenient but users may want
    domain-specific names such as ("slope", "intercept") or ("A", "offset").
    The design matrix ordering is [x | 1] for has_intercept=True, so the first
    name always maps to the slope and the second to the intercept.
    """

    def test_custom_names_with_intercept(self, linear_data):
        """
        Custom names ("slope", "intercept") must appear in FitResult.params
        with the correct values: "slope" ≈ LINEAR_M, "intercept" ≈ LINEAR_B.
        """
        x, ordinate, _ = linear_data
        result = linear_regression(
            abscissa=x,
            ordinate=ordinate,
            has_intercept=True,
            parameter_names=("slope", "intercept"),
            central_value_fit=True,
            resample_fit=False,
        )
        assert "slope"     in result.params, "'slope' missing from FitResult"
        assert "intercept" in result.params, "'intercept' missing from FitResult"
        assert "m"         not in result.params, "Default name 'm' should not appear"
        assert "b"         not in result.params, "Default name 'b' should not appear"

        # Verify the value mapping: first name → slope ≈ LINEAR_M
        slope_err = result.params_hessian_err["slope"].mean
        assert abs(result.params["slope"].mean - LINEAR_M) < NSIG * slope_err, (
            "Custom 'slope' parameter does not match LINEAR_M within 3σ"
        )
        intercept_err = result.params_hessian_err["intercept"].mean
        assert abs(result.params["intercept"].mean - LINEAR_B) < NSIG * intercept_err, (
            "Custom 'intercept' parameter does not match LINEAR_B within 3σ"
        )

    def test_custom_names_no_intercept(self, origin_data):
        """
        Custom name ("slope",) for a through-origin fit must appear in
        FitResult.params and must not be present under the default name "m".
        """
        x, ordinate, _ = origin_data
        result = linear_regression(
            abscissa=x,
            ordinate=ordinate,
            has_intercept=False,
            parameter_names=("slope",),
            central_value_fit=True,
            resample_fit=False,
        )
        assert "slope" in result.params, "'slope' missing from FitResult"
        assert "m"     not in result.params, "Default 'm' should not appear"

        slope_err = result.params_hessian_err["slope"].mean
        assert abs(result.params["slope"].mean - ORIGIN_M) < NSIG * slope_err

    def test_custom_names_resample(self, linear_data):
        """
        Custom names must also be present in rspl after a resample fit.
        """
        x, ordinate, _ = linear_data
        result = linear_regression(
            abscissa=x,
            ordinate=ordinate,
            has_intercept=True,
            parameter_names=("slope", "intercept"),
            central_value_fit=False,
            resample_fit=True,
        )
        for key in ("slope", "intercept"):
            assert key in result.params
            assert result.params[key].rspl.shape == (NBST,)
            assert not np.any(np.isnan(result.params[key].rspl))


# =============================================================================
# 4. S-matrix caching correctness
# =============================================================================

class TestSCaching:
    """
    Verify that the S-matrix caching optimisation gives the same result as
    recomputing S from scratch on every resample.

    The caching is an internal performance optimisation: S = (X^T W X)^{-1} X^T W
    is identical for every resample when the abscissa is a fixed numpy array.
    We compare the cached result against a reference obtained by calling
    linear_regression with a Data abscissa (which forces S to be recomputed on
    each resample because abscissa_is_fixed=False).

    We construct a Data abscissa that has constant rspl (all resamples equal to
    the fixed x array) so the results should be numerically identical apart from
    any floating-point rounding in the two inversion paths.
    """

    def test_cached_s_matches_recomputed_s(self, linear_data):
        """
        A fixed numpy abscissa (uses S-cache) must produce rspl values within
        1e-10 of a Data abscissa whose every resample equals the same x array
        (forces recomputation on every resample).

        We use np.testing.assert_allclose (not exact equality) because the two
        code paths perform the matrix inversion at different points and may
        accumulate floating-point error differently.
        """
        x, ordinate, _ = linear_data

        # Reference: fixed numpy abscissa → S is cached from resample 0.
        result_cached = linear_regression(
            abscissa=x,
            ordinate=ordinate,
            central_value_fit=True,
            resample_fit=True,
            has_intercept=True,
        )

        # Comparison: Data abscissa with all resamples = x → S recomputed each time.
        # Build a Data abscissa: rspl shape (NBST, 8), all rows = x.
        rspl_x = np.tile(x, (NBST, 1))
        abscissa_data = Data.import_resamples(
            resample_type="bst",
            rspl=rspl_x,
            mean=x,
            Nresample=NBST,
        )

        result_recomputed = linear_regression(
            abscissa=abscissa_data,
            ordinate=ordinate,
            central_value_fit=True,
            resample_fit=True,
            has_intercept=True,
        )

        for key in ("m", "b"):
            np.testing.assert_allclose(
                result_cached.params[key].rspl,
                result_recomputed.params[key].rspl,
                rtol=1e-10,
                err_msg=f"S-cache mismatch for '{key}' between fixed and Data abscissa",
            )


# =============================================================================
# 5. Correlated vs uncorrelated consistency
# =============================================================================

class TestCorrelatedVsUncorrelated:
    """
    Verify that correlated and uncorrelated fits give statistically consistent
    parameter estimates.

    Because the mock covariance has only mild off-diagonal structure
    (0.1 × noise²), both paths converge to similar parameter values.  The
    estimates must be within 2σ of each other.  This is not a strict equality
    test — the two estimators are genuinely different, but they should agree
    for mildly correlated data.
    """

    def test_cv_corr_vs_uncorr_consistent(self, linear_data):
        """
        CV correlated and uncorrelated estimates must agree within 2σ for m and b.
        """
        x, ordinate, _ = linear_data

        uncorr = linear_regression(
            abscissa=x, ordinate=ordinate,
            central_value_fit=True, central_value_fit_correlated=False,
            resample_fit=False,
        )
        corr = linear_regression(
            abscissa=x, ordinate=ordinate,
            central_value_fit=True, central_value_fit_correlated=True,
            resample_fit=False,
        )

        for key in ("m", "b"):
            diff = abs(uncorr.params[key].mean - corr.params[key].mean)
            # Use the larger of the two Hessian errors as the scale.
            scale = max(
                uncorr.params_hessian_err[key].mean,
                corr.params_hessian_err[key].mean,
            )
            assert diff < 2 * scale, (
                f"'{key}': correlated ({corr.params[key].mean:.4g}) and uncorrelated "
                f"({uncorr.params[key].mean:.4g}) CV estimates differ by more than "
                f"2σ = {2*scale:.4g}"
            )

    def test_resample_corr_vs_uncorr_consistent(self, linear_data):
        """
        Resample correlated and uncorrelated mean estimates must agree within 2σ.
        """
        x, ordinate, _ = linear_data

        uncorr = linear_regression(
            abscissa=x, ordinate=ordinate,
            central_value_fit=False,
            resample_fit=True, resample_fit_correlated=False,
        )
        corr = linear_regression(
            abscissa=x, ordinate=ordinate,
            central_value_fit=False,
            resample_fit=True, resample_fit_correlated=True,
        )

        for key in ("m", "b"):
            mean_uncorr = float(np.mean(uncorr.params[key].rspl))
            mean_corr   = float(np.mean(corr.params[key].rspl))
            std_uncorr  = float(np.std(uncorr.params[key].rspl, ddof=1))
            diff        = abs(mean_uncorr - mean_corr)
            assert diff < 2 * std_uncorr, (
                f"'{key}': correlated and uncorrelated resample means differ by "
                f"more than 2σ = {2*std_uncorr:.4g}: {mean_corr:.4g} vs {mean_uncorr:.4g}"
            )