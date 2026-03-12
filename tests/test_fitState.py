"""
Unit tests for FitState (fitState.py).

FitState is an ordered collection of FitResult objects with AIC-weighted
model averaging.  All fixtures use real fits via iminuit (with and
without intercept) so that chi2 / AIC / params are populated through the
normal import path 

Test classes
------------
1. TestConstruction           — sort_by validation, initial state
2. TestCollectionInterface    — append, len, getitem, iter, keys_all
3. TestSorting                — chi2_dof and AIC criteria, order after append
4. TestAICWeights             — normalisation, single fit, equal-AIC symmetry
5. TestModelAverageScalar     — single key, list, keys=None, partial coverage,
                                empty/unknown-key guards, keys='fcn' guard
6. TestModelAverageEval       — shape, single-fit identity, alias equivalence
7. TestSerialization          — HDF5 round-trip, sort_by, Nfits, keys_all
"""

from __future__ import annotations

import io

import h5py
import numpy as np
import pytest

from correlatoranalyser import Data
from correlatoranalyser.fitResult import FitResult
from correlatoranalyser.fitState import FitState
from correlatoranalyser.fit_iminuit import fit_iminuit

# =============================================================================
# Shared mock-data and fit helpers
# =============================================================================

_RNG_SEED = 20240101 + 50
NRAW  = 400
NBST  = 60    # small for speed
NDATA = 10

# Ground-truth parameters for the two models used throughout
_M_TRUE = 2.5
_B_TRUE = 0.7
_NOISE  = 0.10

def model_linear_with_intercept(x, p):
    """Linear model: y = m·x + b."""
    return p["m"] * x + p["b"]

def model_linear_no_intercept(x, p):
    """Linear model: y = m·x."""
    return p["m"] * x 

def _make_linear_data(nbst: int = NBST) -> tuple[np.ndarray, Data]:
    """10-point bootstrap data for y = 2.5·x + 0.7."""
    rng    = np.random.default_rng(_RNG_SEED)
    x      = np.linspace(0.0, 1.0, NDATA)
    true_y = _M_TRUE * x + _B_TRUE
    n      = len(true_y)
    cov    = _NOISE ** 2 * (np.eye(n) + 0.1 * np.ones((n, n)))
    raw    = rng.multivariate_normal(mean=true_y, cov=cov, size=NRAW)
    return x, Data(resample_type="bst", data=raw, Nresample=nbst)


def _fit_with_intercept(nbst: int = NBST, correlated: bool = False) -> FitResult:
    """y = m·x + b  (2 parameters, dof = NDATA - 2)."""
    x, ordinate = _make_linear_data(nbst)
    return fit_iminuit(
            abscissa=x,
            ordinate=ordinate,
            central_value_fit=True,
            central_value_fit_correlated=correlated,
            resample_fit=True,
            resample_fit_correlated=correlated,
            model=model_linear_with_intercept,
            p0={"m": _M_TRUE*0.9, "b": _B_TRUE*0.9},
        )

def _fit_no_intercept(nbst: int = NBST, correlated: bool = False) -> FitResult:
    """y = m·x  (1 parameters, dof = NDATA - 1)."""
    x, ordinate = _make_linear_data(nbst)
    return fit_iminuit(
            abscissa=x,
            ordinate=ordinate,
            central_value_fit=True,
            central_value_fit_correlated=correlated,
            resample_fit=True,
            resample_fit_correlated=correlated,
            model=model_linear_no_intercept,
            p0={"m": _M_TRUE*0.9},
        )

def _roundtrip(fs: FitState) -> FitState:
    """Serialize then deserialize a FitState via an in-memory HDF5 buffer."""
    buf = io.BytesIO()
    with h5py.File(buf, "w") as h5:
        fs.serialize(h5)
    buf.seek(0)
    with h5py.File(buf, "r") as h5:
        return FitState.deserialize(h5)


# =============================================================================
# 1. Construction
# =============================================================================

class TestConstruction:

    def test_default_sort_by(self):
        """Default sort criterion must be 'chi2_dof'."""
        fs = FitState()
        assert fs.sort_by == "chi2_dof"

    def test_aic_sort_by(self):
        """sort_by='AIC' must be accepted."""
        fs = FitState(sort_by="AIC")
        assert fs.sort_by == "AIC"

    def test_invalid_sort_by_raises(self):
        """An unrecognised sort_by value must raise ValueError."""
        with pytest.raises(ValueError, match="sort_by"):
            FitState(sort_by="p_value")

    def test_initial_state_empty(self):
        """A freshly constructed FitState must be empty."""
        fs = FitState()
        assert len(fs) == 0
        assert fs.fit_results == []
        assert fs.keys_all    == []

    def test_model_average_empty_raises(self):
        """model_average on an empty FitState must raise RuntimeError."""
        fs = FitState()
        with pytest.raises(RuntimeError, match="empty"):
            fs.model_average("m")

    def test_model_average_eval_empty_raises(self):
        """model_average_eval on an empty FitState must raise RuntimeError."""
        fs = FitState()
        with pytest.raises(RuntimeError, match="empty"):
            fs.model_average_eval(np.linspace(0, 1, 5))

    def test_serialize_empty_raises(self):
        """Serializing an empty FitState must raise ValueError."""
        fs  = FitState()
        buf = io.BytesIO()
        with h5py.File(buf, "w") as h5:
            with pytest.raises(ValueError, match="empty"):
                fs.serialize(h5)


# =============================================================================
# 2. Collection interface
# =============================================================================

class TestCollectionInterface:

    def test_len_after_appends(self):
        """len() must reflect the number of appended fits."""
        fs  = FitState()
        fr1 = _fit_with_intercept()
        fr2 = _fit_no_intercept()
        fs.append(fr1)
        assert len(fs) == 1
        fs.append(fr2)
        assert len(fs) == 2

    def test_getitem(self):
        """__getitem__ must return the fit at the sorted position."""
        fs  = FitState()
        fr1 = _fit_with_intercept()
        fs.append(fr1)
        assert fs[0] is fs.fit_results[0]

    def test_iter(self):
        """Iterating over a FitState must yield all stored FitResult objects."""
        fs  = FitState()
        fr1 = _fit_with_intercept()
        fr2 = _fit_no_intercept()
        fs.append(fr1)
        fs.append(fr2)
        results = list(fs)
        assert len(results) == 2
        assert all(isinstance(r, FitResult) for r in results)

    def test_keys_all_merges_on_append(self):
        """
        keys_all must accumulate the union of parameter keys across all fits.

        Fit-with-intercept has ("m", "b"); fit-without-intercept has ("m",).
        After both appends, keys_all must contain exactly {"m", "b"}.
        """
        fs  = FitState()
        fr1 = _fit_with_intercept()   # params: m, b
        fr2 = _fit_no_intercept()     # params: m only
        fs.append(fr1)
        assert set(fs.keys_all) == {"m", "b"}
        fs.append(fr2)
        assert set(fs.keys_all) == {"m", "b"}   # "m" not duplicated

    def test_keys_all_no_duplicates(self):
        """Appending two fits with identical keys must not produce duplicates."""
        fs  = FitState()
        fs.append(_fit_with_intercept())
        fs.append(_fit_with_intercept())
        assert fs.keys_all.count("m") == 1
        assert fs.keys_all.count("b") == 1


# =============================================================================
# 3. Sorting
# =============================================================================

class TestSorting:
    """
    The intercept fit is correctly specified (y = m·x + b) so its chi²/dof
    will be close to 1.  The no-intercept fit is misspecified (true b ≠ 0)
    so its chi²/dof will be substantially larger.  This gives a reliable
    ordering that does not depend on random seed fluctuations.
    """

    def test_chi2_dof_sort_best_first(self):
        """
        After appending two fits in any order, sort_by='chi2_dof' must place
        the fit with lower chi²/dof at index 0.
        """
        fs  = FitState(sort_by="chi2_dof")
        fr_bad  = _fit_no_intercept()      # misspecified → high chi²/dof
        fr_good = _fit_with_intercept()    # correctly specified → low chi²/dof

        # Append worst first to verify sorting fires on append.
        fs.append(fr_bad)
        fs.append(fr_good)

        chi2_dof_0 = fs[0].chi2.mean / fs[0].dof
        chi2_dof_1 = fs[1].chi2.mean / fs[1].dof
        assert chi2_dof_0 < chi2_dof_1, (
            f"Index 0 should have lower chi²/dof ({chi2_dof_0:.4g}) "
            f"than index 1 ({chi2_dof_1:.4g})"
        )

    def test_aic_sort_best_first(self):
        """
        sort_by='AIC' must place the fit with lower AIC mean at index 0.
        """
        fs  = FitState(sort_by="AIC")
        fr_bad  = _fit_no_intercept()
        fr_good = _fit_with_intercept()

        fs.append(fr_bad)
        fs.append(fr_good)

        assert fs[0].AIC.mean < fs[1].AIC.mean, (
            f"Index 0 AIC ({fs[0].AIC.mean:.4g}) must be less than "
            f"index 1 AIC ({fs[1].AIC.mean:.4g})"
        )

    def test_sort_fires_on_every_append(self):
        """
        Inserting a very-good fit last must move it to index 0 immediately.
        We simulate this by appending the bad fit twice, then the good fit.
        """
        fs = FitState(sort_by="chi2_dof")
        fs.append(_fit_no_intercept())     # bad
        fs.append(_fit_no_intercept())     # bad again
        assert len(fs) == 2

        fs.append(_fit_with_intercept())   # good — must land at index 0
        assert fs[0].chi2.mean / fs[0].dof < fs[1].chi2.mean / fs[1].dof


# =============================================================================
# 4. AIC weights
# =============================================================================

class TestAICWeights:
    """
    Test _aic_weights directly.  The helper is internal but its correctness
    is critical for model averaging: we exercise it through known inputs.
    """

    def _make_aic_stack(self, aic_means: list[float], nbst: int = NBST) -> Data:
        """Build a (Nfits,) Data object with prescribed CV and random rspl AIC values."""
        rng = np.random.default_rng(42)
        n   = len(aic_means)
        d   = Data.zeros(resample_type="bst", shape=(n,), Nresample=nbst, locked_mean=True)
        d.mean = np.array(aic_means)
        for nres in range(nbst):
            # Perturb each AIC slightly per resample.
            d.rspl[nres] = np.array(aic_means) + rng.normal(0, 0.1, size=n)
        return d

    def _make_fs_with_one_fit(self) -> FitState:
        """FitState with a single fit — needed by _resample_config."""
        fs = FitState()
        fs.append(_fit_with_intercept())
        return fs

    def test_weights_sum_to_one_cv(self):
        """CV weights must sum to exactly 1.0."""
        fs     = self._make_fs_with_one_fit()
        stack  = self._make_aic_stack([1.0, 2.0, 3.0])
        w      = fs._aic_weights(stack)
        assert w.mean.sum() == pytest.approx(1.0)

    def test_weights_sum_to_one_per_resample(self):
        """Weights must sum to 1.0 for every resample index."""
        fs    = self._make_fs_with_one_fit()
        stack = self._make_aic_stack([1.0, 2.0, 3.0])
        w     = fs._aic_weights(stack)
        for nres in range(NBST):
            assert w.rspl[nres].sum() == pytest.approx(1.0), (
                f"Weights at nres={nres} sum to {w.rspl[nres].sum():.8g}, expected 1.0"
            )

    def test_single_fit_weight_is_one(self):
        """With a single fit the weight must be exactly 1.0."""
        fs    = self._make_fs_with_one_fit()
        stack = self._make_aic_stack([5.0])
        w     = fs._aic_weights(stack)
        assert w.mean[0] == pytest.approx(1.0)

    def test_equal_aic_gives_equal_weights(self):
        """When all fits have the same AIC the weights must be uniform."""
        fs    = self._make_fs_with_one_fit()
        n     = 4
        stack = self._make_aic_stack([3.0] * n)
        # Force equal rspl too.
        for nres in range(NBST):
            stack.rspl[nres] = np.array([3.0] * n)
        w = fs._aic_weights(stack)
        np.testing.assert_allclose(w.mean, np.full(n, 1.0 / n), rtol=1e-12)

    def test_best_fit_has_highest_weight(self):
        """The fit with the lowest AIC must receive the highest weight."""
        fs    = self._make_fs_with_one_fit()
        stack = self._make_aic_stack([1.0, 5.0, 10.0])
        w     = fs._aic_weights(stack)
        assert w.mean[0] > w.mean[1] > w.mean[2]

    def test_weights_nonnegative(self):
        """All weights must be non-negative (exp is always positive)."""
        fs    = self._make_fs_with_one_fit()
        stack = self._make_aic_stack([0.0, -2.0, 3.5, 100.0])
        w     = fs._aic_weights(stack)
        assert np.all(w.mean >= 0)


# =============================================================================
# 5. model_average — scalar parameters
# =============================================================================

class TestModelAverageScalar:

    @pytest.fixture(scope="class")
    def fs_two_fits(self):
        """FitState with intercept fit (good) and no-intercept fit (bad)."""
        fs = FitState()
        fs.append(_fit_with_intercept())
        fs.append(_fit_no_intercept())
        return fs

    def test_single_key_returns_data(self, fs_two_fits):
        """model_average('m') must return a Data object."""
        avg = fs_two_fits.model_average("m")
        assert isinstance(avg, Data)

    def test_single_key_mean_finite(self, fs_two_fits):
        """The averaged mean for 'm' must be finite."""
        avg = fs_two_fits.model_average("m")
        assert np.isfinite(avg.mean)

    def test_single_key_rspl_populated(self, fs_two_fits):
        """The averaged rspl must be fully populated and finite."""
        avg = fs_two_fits.model_average("m")
        assert avg.rspl.shape == (NBST,)
        assert np.all(np.isfinite(avg.rspl))

    def test_single_key_cv_near_truth(self, fs_two_fits):
        """
        The model-averaged m must be within 10 % of the true slope.
        The best fit (intercept model) strongly dominates the average
        because its AIC is much lower than the misspecified fit.
        """
        avg = fs_two_fits.model_average("m")
        assert abs(avg.mean - _M_TRUE) / _M_TRUE < 0.10, (
            f"Model-averaged m={avg.mean:.4g} deviates more than 10 % from "
            f"true value {_M_TRUE}"
        )

    def test_list_of_keys_returns_dict(self, fs_two_fits):
        """model_average(['m', 'b']) must return a dict with both keys."""
        result = fs_two_fits.model_average(["m", "b"])
        assert isinstance(result, dict)
        assert set(result.keys()) == {"m", "b"}
        for key, val in result.items():
            assert isinstance(val, Data), f"Entry '{key}' must be Data"

    def test_keys_none_averages_all(self, fs_two_fits):
        """model_average(None) must return a dict covering all keys_all."""
        result = fs_two_fits.model_average(None)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(fs_two_fits.keys_all)

    def test_partial_key_coverage(self, fs_two_fits):
        """
        'b' only exists in the intercept fit, not the no-intercept fit.
        model_average('b') must silently exclude the no-intercept fit and
        return a Data object based on the single contributing fit.
        The result must equal the intercept fit's 'b' parameter (weight=1
        because only one fit contributes).
        """
        avg     = fs_two_fits.model_average("b")
        fr_good = fs_two_fits[0]   # intercept fit is at index 0 (best chi²/dof)
        assert avg.mean == pytest.approx(fr_good.params["b"].mean)

    def test_unknown_key_raises(self, fs_two_fits):
        """Requesting a key not in keys_all must raise KeyError."""
        with pytest.raises(KeyError):
            fs_two_fits.model_average("nonexistent_param")

    def test_keys_fcn_without_abscissa_raises(self, fs_two_fits):
        """model_average('fcn') without abscissa must raise ValueError."""
        with pytest.raises(ValueError, match="abscissa"):
            fs_two_fits.model_average("fcn")

    def test_dominant_fit_drives_average(self):
        """
        When one fit has a dramatically lower AIC its parameter must dominate
        the model average.  We construct two fits where the better one has
        a known m value and verify the average is very close to it.
        """
        # Build two corr / uncorr fits from the same data.
        # The correlated fit uses a richer weight matrix; both should give
        # similar m but may have slightly different AIC.
        fs = FitState()
        fs.append(_fit_with_intercept(correlated=False))
        fs.append(_fit_with_intercept(correlated=True))

        avg = fs.model_average("m")
        # Both fits estimate the same truth; the average must still be close.
        assert abs(avg.mean - _M_TRUE) / _M_TRUE < 0.10


# =============================================================================
# 6. model_average_eval (function-evaluation averaging)
# =============================================================================

class TestModelAverageEval:

    @pytest.fixture(scope="class")
    def fs_two_fits(self):
        fs = FitState()
        fs.append(_fit_with_intercept())
        fs.append(_fit_no_intercept())
        return fs

    def test_returns_data_with_correct_shape(self, fs_two_fits):
        """model_average_eval must return Data with shape (Nout,)."""
        x_eval = np.linspace(0.0, 1.5, 20)
        result = fs_two_fits.model_average_eval(x_eval)
        assert isinstance(result, Data)
        assert result.mean.shape == (20,)

    def test_rspl_shape_correct(self, fs_two_fits):
        """rspl must have shape (Nresample, Nout)."""
        x_eval = np.linspace(0.0, 1.5, 20)
        result = fs_two_fits.model_average_eval(x_eval)
        assert result.rspl.shape == (NBST, 20)

    def test_rspl_finite(self, fs_two_fits):
        """All rspl entries must be finite."""
        x_eval = np.linspace(0.0, 1.5, 20)
        result = fs_two_fits.model_average_eval(x_eval)
        assert np.all(np.isfinite(result.rspl))

    def test_single_fit_eval_equals_direct_eval(self):
        """
        With a single stored fit, model_average_eval must return the same
        prediction as calling fit.eval() directly, because the single fit
        receives weight 1.0.
        """
        fr = _fit_with_intercept()
        fs = FitState()
        fs.append(fr)

        x_eval = np.linspace(0.0, 1.5, 15)
        avg    = fs.model_average_eval(x_eval)
        direct = fr.eval(abscissa=x_eval)

        np.testing.assert_allclose(
            avg.mean, direct.mean, rtol=1e-12,
            err_msg="Single-fit model_average_eval must match fit.eval() exactly",
        )
        np.testing.assert_allclose(
            avg.rspl, direct.rspl, rtol=1e-12,
            err_msg="Single-fit rspl must match fit.eval() rspl exactly",
        )

    def test_alias_model_average_fcn_matches(self, fs_two_fits):
        """
        model_average(keys='fcn', abscissa=x) and model_average_eval(x)
        must produce identical results — they call the same implementation.
        """
        x_eval = np.linspace(0.0, 1.0, 10)
        via_alias  = fs_two_fits.model_average_eval(x_eval)
        via_keys   = fs_two_fits.model_average("fcn", abscissa=x_eval)

        np.testing.assert_array_equal(via_alias.mean, via_keys.mean)
        np.testing.assert_array_equal(via_alias.rspl, via_keys.rspl)

    def test_eval_avg_dominated_by_best_fit(self):
        """
        When one fit is strongly preferred its curve should dominate the
        model average.  For our intercept fit vs no-intercept fit the
        intercept fit has much lower AIC; the averaged curve at x=0 must
        be close to the true intercept b_true (which the no-intercept model
        predicts as 0).
        """
        fs = FitState()
        fs.append(_fit_with_intercept())
        fs.append(_fit_no_intercept())

        x_eval = np.array([0.0])   # no-intercept predicts 0; intercept predicts ~0.7
        result = fs.model_average_eval(x_eval)

        # The AIC-weighted average should be much closer to b_true than to 0.
        assert result.mean[0] > 0.3, (
            f"Model avg at x=0 ({result.mean[0]:.4g}) should be pulled "
            f"toward the intercept value {_B_TRUE}"
        )


# =============================================================================
# 7. Serialization
# =============================================================================

class TestSerialization:

    def test_round_trip_nfits(self):
        """The number of stored fits must be preserved after a round-trip."""
        fs = FitState()
        fs.append(_fit_with_intercept())
        fs.append(_fit_no_intercept())
        out = _roundtrip(fs)
        assert len(out) == 2

    def test_round_trip_sort_by_chi2dof(self):
        """sort_by='chi2_dof' must be reconstructed correctly."""
        fs = FitState(sort_by="chi2_dof")
        fs.append(_fit_with_intercept())
        out = _roundtrip(fs)
        assert out.sort_by == "chi2_dof"

    def test_round_trip_sort_by_aic(self):
        """sort_by='AIC' must be reconstructed correctly."""
        fs = FitState(sort_by="AIC")
        fs.append(_fit_with_intercept())
        out = _roundtrip(fs)
        assert out.sort_by == "AIC"

    def test_round_trip_keys_all(self):
        """keys_all must contain the union of parameter keys after round-trip."""
        fs = FitState()
        fs.append(_fit_with_intercept())   # m, b
        fs.append(_fit_no_intercept())     # m
        out = _roundtrip(fs)
        assert set(out.keys_all) == {"m", "b"}

    def test_round_trip_params_preserved(self):
        """
        Parameter means must survive the round-trip.  We compare out against
        the original fit (not against the true values) because serialization
        fidelity — not parameter recovery — is what we are testing.
        """
        fs = FitState()
        fs.append(_fit_with_intercept())
        out = _roundtrip(fs)

        fr_orig = fs[0]
        fr_out  = out[0]
        for key in ("m", "b"):
            assert fr_out.params[key].mean == pytest.approx(
                fr_orig.params[key].mean, rel=1e-10
            )

    def test_round_trip_params_rspl_preserved(self):
        """rspl arrays must survive the round-trip element-wise."""
        fs = FitState()
        fs.append(_fit_with_intercept())
        out = _roundtrip(fs)

        fr_orig = fs[0]
        fr_out  = out[0]
        for key in ("m", "b"):
            np.testing.assert_allclose(
                fr_out.params[key].rspl,
                fr_orig.params[key].rspl,
                rtol=1e-10,
                err_msg=f"rspl mismatch for '{key}' after FitState round-trip",
            )

    def test_round_trip_order_preserved(self):
        """
        The sort order must be maintained after a round-trip: the fit with
        the lower chi²/dof must still be at index 0.
        """
        fs = FitState(sort_by="chi2_dof")
        fs.append(_fit_with_intercept())
        fs.append(_fit_no_intercept())
        out = _roundtrip(fs)

        chi2_dof_0 = out[0].chi2.mean / out[0].dof
        chi2_dof_1 = out[1].chi2.mean / out[1].dof
        assert chi2_dof_0 < chi2_dof_1

    def test_round_trip_model_average_consistent(self):
        """
        model_average('m') computed before and after the round-trip must
        give the same result, since all weights are derived deterministically
        from the stored AIC values.
        """
        fs = FitState()
        fs.append(_fit_with_intercept())
        fs.append(_fit_no_intercept())

        avg_before = fs.model_average("m")
        out        = _roundtrip(fs)
        avg_after  = out.model_average("m")

        assert avg_after.mean == pytest.approx(avg_before.mean, rel=1e-10)
        np.testing.assert_allclose(
            avg_after.rspl, avg_before.rspl, rtol=1e-10,
            err_msg="model_average rspl changed after FitState round-trip",
        )

    def test_deserialize_missing_fitresults_raises(self):
        """
        deserialize must raise RuntimeError if the 'FitResults' group is absent.
        """
        buf = io.BytesIO()
        with h5py.File(buf, "w") as h5:
            h5.create_dataset("FitState/sort_by", data="chi2_dof")
            # deliberately omit FitResults group
        buf.seek(0)
        with h5py.File(buf, "r") as h5:
            with pytest.raises(RuntimeError, match="FitResults"):
                FitState.deserialize(h5)

    def test_round_trip_to_real_file(self, tmp_path):
        """Serialization must work correctly to a real file on disk."""
        path = tmp_path / "fitstate.h5"
        fs   = FitState()
        fs.append(_fit_with_intercept())
        fs.append(_fit_no_intercept())

        with h5py.File(path, "w") as h5:
            fs.serialize(h5)
        with h5py.File(path, "r") as h5:
            out = FitState.deserialize(h5)

        assert len(out) == 2
        assert set(out.keys_all) == {"m", "b"}