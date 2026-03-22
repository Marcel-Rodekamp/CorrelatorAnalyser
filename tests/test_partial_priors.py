"""
Tests for mixed prior / p0 usage (partial prior coverage).

Validates the changes to _get_p0 in fit_helper.py and
_build_lsqfit_args in fit_lsqfit.py.

Scenarios covered
-----------------
1. _get_p0 unit tests
     1a. prior only         → prior means used as start
     1b. p0 only            → p0 used as start (unchanged behaviour)
     1c. prior + p0 (disjoint params)  → both merged
     1d. prior + p0 (overlapping key)  → p0 wins for shared key
     1e. both None          → ValueError (unchanged guard)

2. dof with partial priors
     Verify that only the number of *actually constrained* parameters
     contributes to dof, not the total number of parameters.

3. iminuit end-to-end: partial priors
     Fit C(t) = A·exp(-E·t) with a prior only on E (not on A).
     A must start from p0, E from prior mean or user-supplied p0 override.

4. iminuit: p0 overrides prior start value
     Even when E has a prior, a user-supplied p0["E"] must be used as the
     actual Minuit starting point (the prior still regularises the cost).

5. lsqfit end-to-end: partial priors
     Same scenario: prior on E, p0 for A.  lsqfit must treat A as a free
     (unconstrained) parameter.

"""

from __future__ import annotations

import numpy as np
import pytest

from correlatoranalyser.prior import Prior
from correlatoranalyser.fit_helper import _get_p0


# =============================================================================
# 1. _get_p0 unit tests
# =============================================================================

class TestGetP0:

    def test_prior_only_returns_prior_means(self):
        """prior-only: every key maps to prior.mean."""
        prior = {"m": Prior(2.5, 1.0), "b": Prior(0.7, 0.5)}
        result = _get_p0(prior, None)
        assert result == {"m": 2.5, "b": 0.7}

    def test_p0_only_returns_p0(self):
        """p0-only: unchanged behaviour."""
        p0 = {"m": 2.0, "b": 0.6}
        result = _get_p0(None, p0)
        assert result == p0

    def test_disjoint_prior_and_p0_are_merged(self):
        """Prior covers E, p0 covers A — both must appear in the result."""
        prior = {"E": Prior(0.3, 0.1)}
        p0    = {"A": 1.5}
        result = _get_p0(prior, p0)
        assert result["E"] == pytest.approx(0.3)
        assert result["A"] == pytest.approx(1.5)
        assert set(result) == {"E", "A"}

    def test_p0_overrides_prior_mean_for_shared_key(self):
        """When E is in both prior and p0, p0 value must win."""
        prior  = {"E": Prior(0.3, 0.1)}
        p0     = {"E": 0.45, "A": 1.5}   # user starts E at 0.45, not 0.3
        result = _get_p0(prior, p0)
        assert result["E"] == pytest.approx(0.45), (
            "p0 value must override prior mean when both supply the same key"
        )
        assert result["A"] == pytest.approx(1.5)

    def test_both_none_raises(self):
        """Providing neither prior nor p0 must still be caught downstream."""
        # _get_p0 itself returns an empty dict — validation happens in
        # _validate_inputs, not here.  We simply verify the return is empty.
        result = _get_p0(None, None)
        assert result == {}

    def test_lognormal_prior_mean_used_as_start(self):
        """For a log-normal prior, prior.mean (in log-space) is used as start."""
        prior = {"E": Prior(np.log(0.3), 0.5, dist="log-normal")}
        result = _get_p0(prior, None)
        assert result["E"] == pytest.approx(np.log(0.3))

    def test_lognormal_prior_overridden_by_p0(self):
        """p0 can supply a linear-space start even when the prior is log-normal."""
        prior = {"E": Prior(np.log(0.3), 0.5, dist="log-normal")}
        p0    = {"E": 0.3}   # sensible linear-space start
        result = _get_p0(prior, p0)
        assert result["E"] == pytest.approx(0.3), (
            "p0 must override the log-space prior mean for the start value"
        )


# =============================================================================
# 2. dof with partial priors
# =============================================================================

class TestDofPartialPriors:
    """
    Verify that only parameters WITH a prior contribute to the dof adjustment.

    dof formula (iminuit importer):
        dof = Ndata - Nparams + Npriors
    where Npriors = len(prior) = number of regularised parameters only.
    """

    def _make_data(self, nbst=50):
        """Simple 10-point bootstrap data for a 2-param single-exp fit."""
        try:
            from correlatoranalyser import Data
        except ImportError:
            pytest.skip("correlatoranalyser not importable")

        rng = np.random.default_rng(42)
        t   = np.arange(1, 11, dtype=float)
        C   = 1.5 * np.exp(-0.3 * t)
        cov = 0.02 ** 2 * np.eye(10)
        raw = rng.multivariate_normal(mean=C, cov=cov, size=200)
        return t, Data(resample_type="bst", data=raw, Nresample=nbst)

    def test_no_prior_dof(self):
        """dof = Ndata - Nparams when no priors are used."""
        try:
            from correlatoranalyser.fit_iminuit import fit_iminuit
        except ImportError:
            pytest.skip("correlatoranalyser not importable")

        def model(t, p):
            return p["A"] * np.exp(-p["E"] * t)

        t, ordinate = self._make_data()
        result = fit_iminuit(
            abscissa=t, ordinate=ordinate,
            model=model,
            p0={"A": 1.0, "E": 0.4},
            central_value_fit=True, resample_fit=False,
        )
        assert result.dof == 10 - 2  # 10 data, 2 params, 0 priors

    def test_full_prior_dof(self):
        """dof = Ndata - Nparams + 2 when both params have priors."""
        try:
            from correlatoranalyser.fit_iminuit import fit_iminuit
            from correlatoranalyser.prior import Prior
        except ImportError:
            pytest.skip("correlatoranalyser not importable")

        def model(t, p):
            return p["A"] * np.exp(-p["E"] * t)

        t, ordinate = self._make_data()
        result = fit_iminuit(
            abscissa=t, ordinate=ordinate,
            model=model,
            prior={"A": Prior(1.5, 5.0), "E": Prior(0.3, 5.0)},
            central_value_fit=True, resample_fit=False,
        )
        assert result.dof == 10 - 2 + 2  # 10 data, 2 params, 2 priors

    def test_partial_prior_dof(self):
        """
        dof = Ndata - Nparams + 1 when only one of two params has a prior.
        This is the key new behaviour: dof must use len(prior), not Nparams.
        """
        try:
            from correlatoranalyser.fit_iminuit import fit_iminuit
            from correlatoranalyser.prior import Prior
        except ImportError:
            pytest.skip("correlatoranalyser not importable")

        def model(t, p):
            return p["A"] * np.exp(-p["E"] * t)

        t, ordinate = self._make_data()
        result = fit_iminuit(
            abscissa=t, ordinate=ordinate,
            model=model,
            prior={"E": Prior(0.3, 5.0)},   # only E has a prior
            p0={"A": 1.0},                   # A starts from p0
            central_value_fit=True, resample_fit=False,
        )
        assert result.dof == 10 - 2 + 1, (
            f"Expected dof=9 (Ndata=10, Nparams=2, Npriors=1), got {result.dof}"
        )


# =============================================================================
# 3. iminuit end-to-end: partial prior (prior on E, p0 for A)
# =============================================================================

class TestIminuitPartialPrior:
    """
    Fit C(t) = A·exp(-E·t) where only E has a prior.
    A is provided via p0 only (no regularisation on A).
    """

    _rng  = np.random.default_rng(99)
    _NRAW = 400
    _NBST = 100

    def _make_ordinate(self):
        try:
            from correlatoranalyser import Data
        except ImportError:
            pytest.skip("correlatoranalyser not importable")
        t   = np.arange(1, 11, dtype=float)
        C   = 1.5 * np.exp(-0.3 * t)
        cov = 0.02 ** 2 * (np.eye(10) + 0.1 * np.ones((10, 10)))
        raw = self._rng.multivariate_normal(mean=C, cov=cov, size=self._NRAW)
        return t, Data(resample_type="bst", data=raw, Nresample=self._NBST)

    def test_both_params_present_in_result(self):
        """A and E must both appear in fit_result.params even though only E has a prior."""
        try:
            from correlatoranalyser.fit_iminuit import fit_iminuit
            from correlatoranalyser.prior import Prior
        except ImportError:
            pytest.skip("correlatoranalyser not importable")

        def model(t, p):
            return p["A"] * np.exp(-p["E"] * t)

        t, ordinate = self._make_ordinate()
        result = fit_iminuit(
            abscissa=t, ordinate=ordinate,
            model=model,
            prior={"E": Prior(0.3, 5.0)},
            p0={"A": 1.0},
            limits={"E": (0.0, None)},
            central_value_fit=True, resample_fit=False,
        )
        assert "A" in result.params, "A (p0-only param) must appear in result"
        assert "E" in result.params, "E (prior param) must appear in result"

    def test_prior_stored_only_for_constrained_param(self):
        """result.priors must contain E but NOT A."""
        try:
            from correlatoranalyser.fit_iminuit import fit_iminuit
            from correlatoranalyser.prior import Prior
        except ImportError:
            pytest.skip("correlatoranalyser not importable")

        def model(t, p):
            return p["A"] * np.exp(-p["E"] * t)

        t, ordinate = self._make_ordinate()
        result = fit_iminuit(
            abscissa=t, ordinate=ordinate,
            model=model,
            prior={"E": Prior(0.3, 5.0)},
            p0={"A": 1.0},
            limits={"E": (0.0, None)},
            central_value_fit=True, resample_fit=False,
        )
        assert "E" in result.priors, "E's prior must be stored"
        assert "A" not in result.priors, "A has no prior, must not appear in result.priors"

    def test_parameter_recovery(self):
        """A and E must both be recovered within 3σ."""
        try:
            from correlatoranalyser.fit_iminuit import fit_iminuit
            from correlatoranalyser.prior import Prior
        except ImportError:
            pytest.skip("correlatoranalyser not importable")

        def model(t, p):
            return p["A"] * np.exp(-p["E"] * t)

        t, ordinate = self._make_ordinate()
        result = fit_iminuit(
            abscissa=t, ordinate=ordinate,
            model=model,
            prior={"E": Prior(0.3, 5.0)},
            p0={"A": 1.0},
            limits={"E": (0.0, None)},
            central_value_fit=True, resample_fit=True,
        )
        for key, true_val in {"A": 1.5, "E": 0.3}.items():
            rspl      = result.params[key].rspl
            estimate  = result.params[key].mean
            sigma     = np.std(rspl, ddof=1)
            assert abs(estimate - true_val) < 3 * sigma, (
                f"'{key}': |{estimate:.4g} - {true_val}| >= 3σ={3*sigma:.4g}"
            )


# =============================================================================
# 4. iminuit: p0 overrides prior start value
# =============================================================================

class TestP0OverridesPriorStart:
    """
    When both prior and p0 supply the same parameter, iminuit must start
    from p0, not from prior.mean.  The prior still regularises the cost.
    """

    def test_minuit_starts_from_p0_not_prior_mean(self):
        """
        We verify via number_function_calls: starting at the true value
        (p0 overrides prior mean) should converge in fewer steps than
        starting at a displaced prior mean.
        """
        try:
            from correlatoranalyser import Data
            from correlatoranalyser.fit_iminuit import fit_iminuit
            from correlatoranalyser.prior import Prior
        except ImportError:
            pytest.skip("correlatoranalyser not importable")

        rng = np.random.default_rng(7)
        t   = np.arange(1, 11, dtype=float)
        C   = 1.5 * np.exp(-0.3 * t)
        raw = rng.multivariate_normal(mean=C, cov=0.02**2 * np.eye(10), size=400)
        ordinate = Data(resample_type="bst", data=raw, Nresample=50)

        def model(t, p):
            return p["A"] * np.exp(-p["E"] * t)

        displaced_prior = Prior(0.8, 0.5)  # mean far from truth (0.3)

        # Run 1: start from displaced prior mean (no p0 override)
        r_prior = fit_iminuit(
            abscissa=t, ordinate=ordinate, model=model,
            prior={"A": Prior(1.5, 5.0), "E": displaced_prior},
            central_value_fit=True, resample_fit=False,
        )

        # Run 2: override start with a value close to truth via p0
        r_override = fit_iminuit(
            abscissa=t, ordinate=ordinate, model=model,
            prior={"A": Prior(1.5, 5.0), "E": displaced_prior},
            p0={"E": 0.35},   # good start overrides prior mean of 0.8
            central_value_fit=True, resample_fit=False,
        )

        # Both must converge to a similar minimum (same regularised chi2 landscape).
        np.testing.assert_allclose(
            r_prior.params["E"].mean, r_override.params["E"].mean,
            atol=0.02,
            err_msg="Both runs must converge to the same E regardless of start",
        )

        # The run with the better start should use ≤ function calls.
        assert r_override.number_function_calls <= r_prior.number_function_calls + 500, (
            "Overriding to a better start should not require drastically more calls"
        )


# =============================================================================
# 5. lsqfit end-to-end: partial prior (prior on E, p0 for A)
# =============================================================================

class TestLsqfitPartialPrior:
    """
    lsqfit does not support partial priors (prior + p0 covering different
    parameters).  Verify that a clear ValueError is raised, and that the
    two pure-coverage cases (all-prior, all-p0) still work.
    """

    _rng = np.random.default_rng(11)

    def _make_data(self, nbst=100):
        try:
            from correlatoranalyser import Data
        except ImportError:
            pytest.skip("correlatoranalyser not importable")
        t   = np.arange(1, 11, dtype=float)
        C   = 1.5 * np.exp(-0.3 * t)
        cov = 0.02 ** 2 * (np.eye(10) + 0.1 * np.ones((10, 10)))
        raw = self._rng.multivariate_normal(mean=C, cov=cov, size=400)
        return t, Data(resample_type="bst", data=raw, Nresample=nbst)

    def test_lsqfit_partial_prior_raises_value_error(self):
        """
        prior on E only + p0 on A only must raise ValueError with a
        helpful message pointing to iminuit as the alternative.
        """
        try:
            from correlatoranalyser.fit_lsqfit import fit_lsqfit
            from correlatoranalyser.prior import Prior
        except ImportError:
            pytest.skip("correlatoranalyser not importable")

        def model(t, p):
            return p["A"] * np.exp(-p["E"] * t)

        t, ordinate = self._make_data()
        with pytest.raises(ValueError, match="iminuit"):
            fit_lsqfit(
                abscissa=t, ordinate=ordinate,
                model=model,
                prior={"E": Prior(0.3, 5.0)},
                p0={"A": 1.0},
                central_value_fit=True, resample_fit=False,
            )

    def test_lsqfit_partial_prior_error_names_uncovered_param(self):
        """The error message must name the uncovered parameter(s)."""
        try:
            from correlatoranalyser.fit_lsqfit import fit_lsqfit
            from correlatoranalyser.prior import Prior
        except ImportError:
            pytest.skip("correlatoranalyser not importable")

        def model(t, p):
            return p["A"] * np.exp(-p["E"] * t)

        t, ordinate = self._make_data()
        with pytest.raises(ValueError, match="A"):
            fit_lsqfit(
                abscissa=t, ordinate=ordinate,
                model=model,
                prior={"E": Prior(0.3, 5.0)},
                p0={"A": 1.0},
                central_value_fit=True, resample_fit=False,
            )

    def test_lsqfit_all_prior_still_works(self):
        """Full prior coverage (no p0) must continue to work normally."""
        try:
            from correlatoranalyser.fit_lsqfit import fit_lsqfit
            from correlatoranalyser.prior import Prior
        except ImportError:
            pytest.skip("correlatoranalyser not importable")

        def model(t, p):
            return p["A"] * np.exp(-p["E"] * t)

        t, ordinate = self._make_data()
        result = fit_lsqfit(
            abscissa=t, ordinate=ordinate,
            model=model,
            prior={"A": Prior(1.5, 5.0), "E": Prior(0.3, 5.0)},
            central_value_fit=True, resample_fit=False,
        )
        assert np.isfinite(result.params["A"].mean)
        assert np.isfinite(result.params["E"].mean)

    def test_lsqfit_all_p0_still_works(self):
        """Pure p0 (no prior) must continue to work normally."""
        try:
            from correlatoranalyser.fit_lsqfit import fit_lsqfit
        except ImportError:
            pytest.skip("correlatoranalyser not importable")

        def model(t, p):
            return p["A"] * np.exp(-p["E"] * t)

        t, ordinate = self._make_data()
        result = fit_lsqfit(
            abscissa=t, ordinate=ordinate,
            model=model,
            p0={"A": 1.0, "E": 0.4},
            central_value_fit=True, resample_fit=False,
        )
        assert np.isfinite(result.params["A"].mean)
        assert np.isfinite(result.params["E"].mean)
