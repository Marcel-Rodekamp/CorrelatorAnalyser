"""
Unit tests for FitResult (fitResult.py).

The end-to-end tests (test_fit_iminuit.py etc.) already exercise
import_from_* indirectly.  This module tests the remaining public API
of FitResult in isolation, building minimal FitResult objects by hand
rather than running a full fit.

Test classes
------------
1. TestConstruction         — constructor guards, has_resamples, Ndata
2. TestSerialization        — HDF5 round-trip for all field combinations
3. TestEval                 — eval() on fit and new abscissa, with/without resamples
4. TestAIC                  — AIC / AICc formula correctness and edge cases
5. TestPValue               — compute_p_value (eq. 2.18 of arXiv:2209.14188)
6. TestRepr                 — __repr__ smoke test and key field presence
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.special import gammaincc

from correlatoranalyser import Data
from correlatoranalyser.fitResult import FitResult
from correlatoranalyser.prior import Prior


# =============================================================================
# Shared helpers
# =============================================================================

NBST = 50   # small enough to keep tests fast
NDATA = 8   # number of fit data points


def _abscissa():
    return np.linspace(1.0, float(NDATA), NDATA)


def _make_param_data(mean_val: float, rspl_vals=None, nbst=NBST) -> Data:
    """Return a locked-mean Data object mimicking a fit parameter."""
    d = Data.zeros(
        resample_type="bst" if rspl_vals is not None else None,
        shape=None,
        Nresample=nbst if rspl_vals is not None else None,
        locked_mean=True
    )
    d.mean = mean_val
    if rspl_vals is not None:
        d.rspl[:] = rspl_vals
    return d


def _make_quality_data(mean_val: float, rspl_vals, nbst=NBST) -> Data:
    """Return a locked-mean Data object for chi2 / p_value / AIC."""
    d = Data.empty(resample_type="bst", shape=None, Nresample=nbst, locked_mean=True)
    d.mean = mean_val
    d.rspl[:] = rspl_vals
    return d


def _minimal_cv_result(abscissa=None) -> FitResult:
    """
    Minimal FitResult with a single CV fit (no resamples).

    Parameters: m=2.5 (slope), b=0.7 (intercept)
    Model: y = m*x + b
    chi2 = 3.2, dof = 6
    """
    fr = FitResult(abscissa=abscissa if abscissa is not None else _abscissa())
    fr.dof = NDATA - 2
    fr.params["m"] = _make_param_data(2.5)
    fr.params["b"] = _make_param_data(0.7)
    fr.params_hessian_err["m"] = _make_param_data(0.04)
    fr.params_hessian_err["b"] = _make_param_data(0.12)
    fr.chi2    = 3.2
    fr.expected_chi2 = 6
    fr.p_value = 0.78
    fr.expected_p_value = 0.81
    fr.AIC     = -7.1

    def model_linear(x, p):
        return p["m"] * x + p["b"]

    fr.fcn = model_linear
    return fr


def _minimal_resample_result(nbst=NBST) -> FitResult:
    """
    Minimal FitResult with CV + resample fits.

    Parameters: A=1.5, E=0.3 (single exponential)
    """
    rng = np.random.default_rng(99)
    A_rspl = rng.normal(1.5, 0.05, size=nbst)
    E_rspl = rng.normal(0.3, 0.01, size=nbst)

    fr = FitResult(
        abscissa=_abscissa(),
        Nresample=nbst,
        resample_type="bst",
    )
    fr.dof = NDATA - 2

    fr.params["A"] = _make_param_data(1.5, A_rspl, nbst)
    fr.params["E"] = _make_param_data(0.3, E_rspl, nbst)
    fr.params_hessian_err["A"] = _make_param_data(0.05, rng.normal(0.05, 0.002, nbst), nbst)
    fr.params_hessian_err["E"] = _make_param_data(0.01, rng.normal(0.01, 0.0005, nbst), nbst)

    chi2_rspl   = rng.chisquare(NDATA - 2, size=nbst)
    pval_rspl   = np.clip(rng.uniform(0, 1, size=nbst), 1e-6, 1.0)
    aic_rspl    = rng.normal(-5.0, 1.0, size=nbst)

    exp_pval_rspl = np.clip(rng.uniform(0, 1, size=nbst), 1e-6, 1.0)

    fr.chi2    = _make_quality_data(3.2, chi2_rspl, nbst)
    fr.p_value = _make_quality_data(0.78, pval_rspl, nbst)
    fr.expected_chi2 = _make_quality_data(NDATA-2, np.full_like(chi2_rspl, NDATA-2), nbst)
    fr.expected_p_value = _make_quality_data(0.81, exp_pval_rspl, nbst)
    fr.AIC     = _make_quality_data(-7.1, aic_rspl, nbst)

    def model_single_exp(t, p):
        return p["A"] * np.exp(-p["E"] * t)

    fr.fcn = model_single_exp
    return fr

# =============================================================================
# 1. Construction
# =============================================================================

class TestConstruction:
    """
    Test FitResult.__init__ guards, has_resamples, and the Ndata property.
    """

    def test_default_construction_no_args(self):
        """FitResult() with no arguments must not raise."""
        fr = FitResult()
        assert fr.params == {}
        assert fr.priors == {}
        assert fr.params_hessian_err == {}
        assert fr.dof is None
        assert fr.fcn is None

    def test_bad_resample_type_raises(self):
        """Unknown resample_type must raise ValueError immediately."""
        with pytest.raises(ValueError, match="resample_type"):
            FitResult(resample_type="jackknife", Nresample=100)

    def test_resample_type_without_nresample_raises(self):
        """Specifying resample_type without Nresample must raise ValueError."""
        with pytest.raises(ValueError):
            FitResult(resample_type="bst")   # Nresample not given

    def test_has_resamples_false_without_resample_type(self):
        """has_resamples must be False when resample_type is None."""
        fr = FitResult()
        assert not fr.has_resamples

    def test_has_resamples_true_with_resample_type(self):
        """has_resamples must be True when resample_type and Nresample are given."""
        fr = FitResult(resample_type="bst", Nresample=100)
        assert fr.has_resamples

    def test_has_resamples_true_for_jkn(self):
        """Jackknife resample type must also set has_resamples=True."""
        fr = FitResult(resample_type="jkn", Nresample=50)
        assert fr.has_resamples

    def test_ndata_raises_without_abscissa(self):
        """Ndata must raise RuntimeError when abscissa has not been set."""
        fr = FitResult()
        with pytest.raises(RuntimeError, match="abscissa"):
            _ = fr.Ndata

    def test_ndata_correct_for_numpy_abscissa(self):
        """Ndata must equal len(abscissa) for a 1-D numpy array abscissa."""
        x  = np.arange(1, 11, dtype=float)
        fr = FitResult(abscissa=x)
        assert fr.Ndata == 10

    def test_ndata_correct_for_2d_abscissa(self):
        """Ndata = prod(shape) for a 2-D abscissa (e.g. a lattice time-momentum grid)."""
        x  = np.ones((4, 5))
        fr = FitResult(abscissa=x)
        assert fr.Ndata == 20

    def test_resample_data_containers_allocated_on_construction(self):
        """
        When resample_type is given, chi2/p_value/AIC must already be Data
        objects with rspl of the correct length — not None.
        """
        fr = FitResult(resample_type="bst", Nresample=NBST)
        for attr in ("chi2", "p_value", "AIC"):
            val = getattr(fr, attr)
            assert isinstance(val, Data), f"{attr} must be Data after construction with resamples"
            assert val.rspl.shape == (NBST,)

    def test_serialize_empty_raises(self):
        """Calling serialize on a FitResult with no params must raise ValueError."""
        fr = FitResult(abscissa=_abscissa())
        with tempfile.NamedTemporaryFile(suffix=".h5") as f:
            with h5py.File(f.name, "w") as h5:
                with pytest.raises(ValueError, match="No fit results"):
                    fr.serialize(h5)


# =============================================================================
# 2. Serialization / deserialization
# =============================================================================

class TestSerialization:
    """
    HDF5 round-trip tests.

    Each test constructs a FitResult, writes it to an in-memory HDF5 file,
    reads it back, and asserts field-by-field equality.

    Covered combinations
    --------------------
    * CV-only with numpy abscissa
    * CV-only with Data abscissa
    * CV + resamples (bst)
    * With normal prior
    * With log-normal prior
    * Writing into a named node vs the root group
    * Multiple FitResults written into the same file under different nodes
    * Hessian errors round-trip
    """

    @staticmethod
    def _roundtrip(fr: FitResult, node: str | None = None) -> FitResult:
        """Serialize then deserialize fr in an in-memory HDF5 file."""
        buf = io.BytesIO()
        with h5py.File(buf, "w") as h5:
            fr.serialize(h5, node=node)
        buf.seek(0)
        with h5py.File(buf, "r") as h5:
            return FitResult.deserialize(h5, node=node)

    def test_cv_only_params_round_trip(self):
        """Parameter mean values must survive a CV-only round-trip exactly."""
        fr  = _minimal_cv_result()
        out = self._roundtrip(fr)

        assert set(out.params.keys()) == {"m", "b"}
        assert out.params["m"].mean == 2.5
        assert out.params["b"].mean == 0.7

    def test_cv_only_hessian_err_round_trip(self):
        """Hessian errors must round-trip correctly for a CV-only fit."""
        fr  = _minimal_cv_result()
        out = self._roundtrip(fr)

        assert "m" in out.params_hessian_err
        assert "b" in out.params_hessian_err
        assert out.params_hessian_err["m"].mean == pytest.approx(0.04)
        assert out.params_hessian_err["b"].mean == pytest.approx(0.12)

    def test_cv_only_fit_quality_round_trip(self):
        """chi2, p_value, expected_p_value, and AIC must round-trip as plain floats for CV-only."""
        fr  = _minimal_cv_result()
        out = self._roundtrip(fr)

        assert float(out.chi2)             == pytest.approx(3.2)
        assert float(out.p_value)          == pytest.approx(0.78)
        assert float(out.expected_p_value) == pytest.approx(0.81)
        assert float(out.AIC)              == pytest.approx(-7.1)

    def test_cv_only_dof_round_trip(self):
        """dof must round-trip exactly as an integer."""
        fr  = _minimal_cv_result()
        out = self._roundtrip(fr)
        assert out.dof == NDATA - 2

    def test_cv_only_abscissa_round_trip(self):
        """Numpy abscissa must round-trip element-wise."""
        fr  = _minimal_cv_result()
        out = self._roundtrip(fr)
        np.testing.assert_array_equal(out.abscissa, fr.abscissa)

    def test_resample_params_round_trip(self):
        """
        With resamples: mean and every rspl entry must survive the round-trip
        for both A and E.
        """
        fr  = _minimal_resample_result()
        out = self._roundtrip(fr)
        print(fr.params["A"])
        print(out.params["A"])

        for key in ("A", "E"):
            assert out.params[key].mean == fr.params[key].mean
            assert np.all(out.params[key].rspl == fr.params[key].rspl)

    def test_resample_hessian_err_round_trip(self):
        """Hessian error rspl arrays must survive the round-trip."""
        fr  = _minimal_resample_result()
        out = self._roundtrip(fr)

        for key in ("A", "E"):
            np.testing.assert_allclose(
                out.params_hessian_err[key].rspl,
                fr.params_hessian_err[key].rspl,
                rtol=1e-12,
            )

    def test_resample_fit_quality_round_trip(self):
        """chi2 / p_value / expected_p_value / AIC rspl arrays must round-trip correctly."""
        fr  = _minimal_resample_result()
        out = self._roundtrip(fr)

        for attr in ("chi2", "p_value", "expected_p_value", "AIC"):
            np.testing.assert_allclose(
                getattr(out, attr).rspl,
                getattr(fr,  attr).rspl,
                rtol=1e-12,
                err_msg=f"{attr} rspl mismatch after round-trip",
            )

    def test_resample_type_and_nresample_preserved(self):
        """resample_type and Nresample must be reconstructed correctly."""
        fr  = _minimal_resample_result()
        out = self._roundtrip(fr)
        assert out.resample_type == "bst"
        assert out.Nresample     == NBST

    def test_fcn_source_round_trip(self):
        """
        The model function must be reconstructed from source code and must
        produce the same output on the fit abscissa as the original.

        We compare model predictions rather than byte equality because the
        exec namespace may differ; what matters is that the reconstructed
        function is callable and numerically correct.
        """
        fr  = _minimal_cv_result()
        out = self._roundtrip(fr)

        assert out.fcn is not None, "fcn must be reconstructed after round-trip"
        x  = fr.abscissa
        p  = {k: v.mean for k, v in fr.params.items()}
        np.testing.assert_allclose(
            out.fcn(x, p),
            fr.fcn(x, p),
            rtol=1e-12,
            err_msg="Reconstructed fcn produces different output from original",
        )

    def test_normal_prior_round_trip(self):
        """
        A normal prior stored in self.priors must survive the round-trip
        with mean, sdev, and dist preserved exactly.
        """
        fr = _minimal_cv_result()
        fr.priors["E"] = Prior(0.3, 2.0)
        out = self._roundtrip(fr)

        assert "E" in out.priors
        assert out.priors["E"].mean == pytest.approx(0.3)
        assert out.priors["E"].sdev == pytest.approx(2.0)
        assert out.priors["E"].dist == "normal"

    def test_lognormal_prior_round_trip(self):
        """
        A log-normal prior must round-trip with dist='log-normal' preserved.
        """
        fr = _minimal_cv_result()
        fr.priors["E"] = Prior(np.log(0.3), 1.0, dist="log-normal")
        out = self._roundtrip(fr)

        assert "E" in out.priors
        assert out.priors["E"].dist == "log-normal"
        assert out.priors["E"].mean == pytest.approx(np.log(0.3))
        assert out.priors["E"].sdev == pytest.approx(1.0)

    def test_named_node_round_trip(self):
        """
        Writing to and reading from a named node ('fit/result') must give
        the same result as writing to the root group.
        """
        fr  = _minimal_cv_result()
        out = self._roundtrip(fr, node="fit/result")
        assert out.params["m"].mean == pytest.approx(2.5)
        assert out.params["b"].mean == pytest.approx(0.7)

    def test_multiple_nodes_in_same_file(self):
        """
        Two FitResult objects written to different nodes in the same HDF5
        file must be independently readable without cross-contamination.
        """
        fr1 = _minimal_cv_result()
        fr1.params["m"].mean = 1.0

        fr2 = _minimal_cv_result()
        fr2.params["m"].mean = 3.0

        buf = io.BytesIO()
        with h5py.File(buf, "w") as h5:
            fr1.serialize(h5, node="fit1")
            fr2.serialize(h5, node="fit2")
        buf.seek(0)
        with h5py.File(buf, "r") as h5:
            out1 = FitResult.deserialize(h5, node="fit1")
            out2 = FitResult.deserialize(h5, node="fit2")

        assert out1.params["m"].mean == pytest.approx(1.0)
        assert out2.params["m"].mean == pytest.approx(3.0)

    def test_data_abscissa_round_trip(self):
        """
        When the abscissa is a Data object (rather than a plain numpy array)
        it must be fully reconstructed after the round-trip.
        """
        rng = np.random.default_rng(7)
        x_raw = rng.normal(np.arange(1, NDATA + 1), 0.01, size=(100, NDATA))
        abscissa_data = Data(resample_type="bst", data=x_raw, Nresample=NBST)

        fr  = _minimal_cv_result(abscissa=abscissa_data)
        out = self._roundtrip(fr)

        assert isinstance(out.abscissa, Data)
        np.testing.assert_allclose(
            out.abscissa.mean, fr.abscissa.mean, rtol=1e-12
        )

    def test_round_trip_to_real_file(self, tmp_path):
        """
        Serialization must work to a real file on disk, not only in-memory.
        This catches any seek-related assumptions in the HDF5 layer.
        """
        path = tmp_path / "test_result.h5"
        fr   = _minimal_resample_result()

        with h5py.File(path, "w") as h5:
            fr.serialize(h5, node="result")
        with h5py.File(path, "r") as h5:
            out = FitResult.deserialize(h5, node="result")

        assert out.params["A"].mean == pytest.approx(1.5)
        assert out.Nresample        == NBST


# =============================================================================
# 3. eval()
# =============================================================================

class TestEval:
    """
    Test FitResult.eval() — model evaluation at the fit abscissa or a new one.

    eval() requires self.fcn to be set.  It builds the model prediction for
    both the central value (using params[key].mean) and each resample
    (using params[key].rspl[nres]).
    """

    def test_eval_cv_only_on_fit_abscissa(self):
        """
        CV-only eval on the fit abscissa must return the model value
        y = m*x + b exactly at the fit parameter means.
        """
        fr   = _minimal_cv_result()
        pred = fr.eval()

        expected = fr.params["m"].mean * fr.abscissa + fr.params["b"].mean
        np.testing.assert_allclose(pred.mean, expected, rtol=1e-12)

    def test_eval_cv_only_on_new_abscissa(self):
        """
        eval() must accept a new abscissa (extrapolation / finer grid) and
        use it instead of the fit abscissa.
        """
        fr = _minimal_cv_result()
        x_new = np.linspace(0.0, 2.0, 20)
        pred  = fr.eval(abscissa=x_new)

        expected = fr.params["m"].mean * x_new + fr.params["b"].mean
        np.testing.assert_allclose(pred.mean, expected, rtol=1e-12)

    def test_eval_resample_fills_rspl(self):
        """
        For a FitResult with resamples, eval() must populate pred.rspl for
        every resample index with the correct model value.
        """
        fr   = _minimal_resample_result()
        pred = fr.eval()

        assert pred.rspl.shape == (NBST, NDATA)
        for nres in range(NBST):
            A_nres = fr.params["A"].rspl[nres]
            E_nres = fr.params["E"].rspl[nres]
            expected = A_nres * np.exp(-E_nres * fr.abscissa)
            np.testing.assert_allclose(
                pred.rspl[nres], expected, rtol=1e-12,
                err_msg=f"rspl mismatch at nres={nres}",
            )

    def test_eval_without_fcn_raises(self):
        """eval() must raise RuntimeError when fcn is None."""
        fr     = _minimal_cv_result()
        fr.fcn = None
        with pytest.raises(RuntimeError, match="No model function"):
            fr.eval()

    def test_eval_defaults_to_fit_abscissa_when_none_given(self):
        """Calling eval() with no argument must use self.abscissa."""
        fr = _minimal_cv_result()
        pred_implicit = fr.eval()
        pred_explicit = fr.eval(abscissa=fr.abscissa)
        np.testing.assert_array_equal(pred_implicit.mean, pred_explicit.mean)

    def test_eval_after_round_trip(self):
        """
        eval() must work on a FitResult reconstructed from HDF5, i.e. the
        reconstructed fcn must be callable and numerically correct.
        """
        fr  = _minimal_cv_result()
        buf = io.BytesIO()
        with h5py.File(buf, "w") as h5:
            fr.serialize(h5)
        buf.seek(0)
        with h5py.File(buf, "r") as h5:
            out = FitResult.deserialize(h5)

        pred_original = fr.eval()
        pred_loaded   = out.eval(abscissa=fr.abscissa)
        np.testing.assert_allclose(
            pred_loaded.mean, pred_original.mean, rtol=1e-12,
            err_msg="eval() after round-trip gives different prediction",
        )


# =============================================================================
# 4. AIC / AICc
# =============================================================================

class TestAIC:
    """
    Verify the AIC and AICc formula implementations.

    AIC  = chi2 + 2*(k - d_K)
    AICc = AIC  + (2k² + 2k) / (d_K - k - 1)

    where k = Nparam and d_K = Ndata.
    """

    def _make_fr(self, nparams: int, ndata: int, chi2: float, aicc: bool = True) -> FitResult:
        """Build a minimal FitResult with given dimensions for AIC testing."""
        x  = np.arange(1, ndata + 1, dtype=float)
        fr = FitResult(abscissa=x, AIC_small_sample_correction=aicc)
        fr.dof = ndata - nparams
        # Add placeholder params so _compute_AIC can count k.
        for i in range(nparams):
            fr.params[f"p{i}"] = _make_param_data(float(i))
        fr.chi2    = chi2
        fr.p_value = 0.5
        fr.AIC     = fr._compute_AIC(chi2)   # compute and store
        return fr

    def test_aic_no_correction_formula(self):
        """
        With AIC_small_sample_correction=False the formula is
        AIC = chi2 + 2*(k - d_K).

        For chi2=4, k=2, d_K=8: AIC = 4 + 2*(2-8) = 4 - 12 = -8.
        """
        fr  = self._make_fr(nparams=2, ndata=8, chi2=4.0, aicc=False)
        expected = 4.0 + 2.0 * (2 - 8)   # = -8
        assert fr.AIC == pytest.approx(expected)

    def test_aicc_correction_formula(self):
        """
        With AIC_small_sample_correction=True the formula adds
        (2k² + 2k) / (d_K - k - 1).

        For chi2=4, k=2, d_K=8:
          AIC  = -8
          AICc = -8 + (2*4 + 4) / (8 - 2 - 1) = -8 + 12/5 = -8 + 2.4 = -5.6
        """
        fr  = self._make_fr(nparams=2, ndata=8, chi2=4.0, aicc=True)
        aic  = 4.0 + 2.0 * (2 - 8)          # = -8
        aicc = aic + (2 * 4 + 4) / (8 - 2 - 1)  # = -8 + 12/5 = -5.6
        assert fr.AIC == pytest.approx(aicc)

    def test_aicc_denominator_zero_raises(self):
        """
        AICc requires d_K > k + 1.  When d_K == k + 1 (denominator = 0)
        _compute_AIC must raise RuntimeError rather than dividing by zero.
        """
        # nparams=3, ndata=4 → d_K - k - 1 = 4 - 3 - 1 = 0
        x  = np.arange(1, 5, dtype=float)
        fr = FitResult(abscissa=x, AIC_small_sample_correction=True)
        fr.dof = 1
        for i in range(3):
            fr.params[f"p{i}"] = _make_param_data(float(i))

        with pytest.raises(RuntimeError, match="AICc"):
            fr._compute_AIC(2.0)

    def test_aicc_denominator_negative_raises(self):
        """
        When d_K < k + 1 the denominator is negative, producing a
        nonsensical AICc.  The code should raise rather than silently
        return a wrong value.
        """
        # nparams=5, ndata=4 → d_K - k - 1 = 4 - 5 - 1 = -2 (negative)
        x  = np.arange(1, 5, dtype=float)
        fr = FitResult(abscissa=x, AIC_small_sample_correction=True)
        fr.dof = -1   # degenerate, but we just want to reach _compute_AIC
        for i in range(5):
            fr.params[f"p{i}"] = _make_param_data(float(i))

        # Current code: ZeroDivisionError only when denominator == 0;
        # negative denominator silently produces a wrong value.
        # We assert the result is negative (wrong sign compared to large-N limit)
        # as a marker that the correction is misbehaving.
        with pytest.raises(RuntimeError, match="AICc"):
            result = fr._compute_AIC(2.0)

    def test_aic_no_params_raises(self):
        """_compute_AIC must raise RuntimeError when no params have been added."""
        x  = np.arange(1, 9, dtype=float)
        fr = FitResult(abscissa=x)
        fr.dof = 6
        with pytest.raises(RuntimeError, match="Cannot compute AIC"):
            fr._compute_AIC(4.0)

    def test_aicc_larger_than_aic_for_small_samples(self):
        """
        For small samples the AICc correction term is positive (k ≥ 1,
        d_K > k + 1), so AICc must always be strictly greater than AIC.
        """
        chi2 = 5.0
        k, dK = 2, 8

        x  = np.arange(1, dK + 1, dtype=float)

        fr_aic  = FitResult(abscissa=x, AIC_small_sample_correction=False)
        fr_aicc = FitResult(abscissa=x, AIC_small_sample_correction=True)
        fr_aic.dof = fr_aicc.dof = dK - k

        for label, fr in (("aic", fr_aic), ("aicc", fr_aicc)):
            for i in range(k):
                fr.params[f"p{i}"] = _make_param_data(float(i))

        aic_val  = fr_aic._compute_AIC(chi2)
        aicc_val = fr_aicc._compute_AIC(chi2)
        assert aicc_val > aic_val, (
            f"AICc ({aicc_val:.4g}) must be > AIC ({aic_val:.4g}) for small samples"
        )


# =============================================================================
# 5. compute_p_value (eq. 2.18 of arXiv:2209.14188)
# =============================================================================

class TestPValue:
    """
    Verify FitResult.compute_p_value against the paper's known analytic
    special case (eq. 4.10): for a fully correlated fit (W = C^-1 exactly),
    Q(chi2_obs, nu) must reduce to the standard incomplete-gamma p-value
    gammaincc(dof/2, chi2_obs/2), since nu then has exactly dof eigenvalues
    equal to 1 and the rest 0.
    """

    @staticmethod
    def _linear_setup(nparams=2, ndata=8, seed=42):
        """A small linear model with a random SPD (correlated) covariance."""
        rng = np.random.default_rng(seed)
        x = np.linspace(1.0, float(ndata), ndata)
        J = np.vstack([x**k for k in range(nparams)])  # (nparams, ndata)
        A = rng.normal(size=(ndata, ndata))
        cov = A @ A.T + ndata * np.eye(ndata)
        return J, cov

    def test_correlated_fit_matches_analytic_gammaincc(self):
        """For W = C^-1, Q(chi2_obs, nu) must match gammaincc(dof/2, chi2_obs/2)."""
        J, cov = self._linear_setup()
        W  = np.linalg.inv(cov)
        fr = FitResult()
        fr.dof = cov.shape[0] - J.shape[0]

        for chi2_obs in (2.0, 6.0, 10.0, 20.0):
            Q_mc = fr.compute_p_value(cov, W, J, chi2_obs, Nmc=300_000)
            Q_analytic = float(gammaincc(fr.dof / 2.0, chi2_obs / 2.0))
            assert Q_mc == pytest.approx(Q_analytic, abs=5e-3)

    def test_priors_shift_p_value_like_extra_dof(self):
        """
        Npriors extra unit eigenvalues must shift Q the same way as adding
        Npriors to dof in the analytic gammaincc formula (correlated-fit case).
        """
        J, cov = self._linear_setup()
        W  = np.linalg.inv(cov)
        fr = FitResult()
        fr.dof = cov.shape[0] - J.shape[0]

        chi2_obs = 8.0
        Q_mc = fr.compute_p_value(cov, W, J, chi2_obs, Npriors=2, Nmc=300_000)
        Q_analytic = float(gammaincc((fr.dof + 2) / 2.0, chi2_obs / 2.0))
        assert Q_mc == pytest.approx(Q_analytic, abs=5e-3)

    def test_uncorrelated_weight_gives_valid_probability(self):
        """
        For an arbitrary (non-inverse-covariance) weight matrix, Q must
        still be a valid probability in [0, 1].
        """
        J, cov = self._linear_setup()
        W  = np.diag(1.0 / np.diag(cov))  # uncorrelated weight
        fr = FitResult()
        fr.dof = cov.shape[0] - J.shape[0]

        for chi2_obs in (2.0, 6.0, 10.0, 20.0):
            Q = fr.compute_p_value(cov, W, J, chi2_obs, Nmc=50_000)
            assert 0.0 <= Q <= 1.0

    def test_zero_chi2_gives_p_value_one(self):
        """chi2_obs = 0 must give Q = 1 (certain to observe chi2 >= 0)."""
        J, cov = self._linear_setup()
        W  = np.linalg.inv(cov)
        fr = FitResult()
        fr.dof = cov.shape[0] - J.shape[0]

        Q = fr.compute_p_value(cov, W, J, 0.0, Nmc=50_000)
        assert Q == pytest.approx(1.0, abs=1e-6)


# =============================================================================
# 6. __repr__
# =============================================================================

class TestRepr:
    """
    Smoke tests for FitResult.__repr__.

    We do not assert the exact string format (it may evolve) but verify:
    * the call does not raise
    * all parameter names appear in the output
    * chi²/dof, p-value, and AIC are present
    * prior tags appear when priors are set
    """

    def test_repr_cv_only_does_not_raise(self):
        """__repr__ must not raise for a CV-only FitResult."""
        fr = _minimal_cv_result()
        s  = repr(fr)
        assert isinstance(s, str)
        assert len(s) > 0

    def test_repr_contains_param_names(self):
        """Both parameter names must appear somewhere in the repr string."""
        fr = _minimal_cv_result()
        s  = repr(fr)
        assert "m" in s
        assert "b" in s

    def test_repr_contains_chi2_dof(self):
        """The chi²/dof summary line must appear in the repr."""
        fr = _minimal_cv_result()
        s  = repr(fr)
        assert "χ²" in s or "chi" in s.lower()
        assert str(fr.dof) in s

    def test_repr_contains_p_value(self):
        """A p-value line must appear in the repr."""
        fr = _minimal_cv_result()
        s  = repr(fr)
        assert "p-value" in s or "p_value" in s

    def test_repr_contains_expected_p_value(self):
        """
        An <p-value> line (eq. 2.18 of arXiv:2209.14188) must appear in the
        repr right after the plain p-value, mirroring the <χ²> convention
        used for expected_chi2.
        """
        fr = _minimal_cv_result()
        s  = repr(fr)
        assert "<p-value>" in s

        p_line     = next(l for l in s.splitlines() if l.strip().startswith("p-value"))
        exp_p_line = next(l for l in s.splitlines() if l.strip().startswith("<p-value>"))
        assert s.index(p_line) < s.index(exp_p_line)

    def test_repr_contains_aic(self):
        """AIC must appear in the repr."""
        fr = _minimal_cv_result()
        s  = repr(fr)
        assert "AIC" in s

    def test_repr_shows_prior_tag_when_prior_set(self):
        """
        When a prior is stored, its string representation must appear in the
        repr next to the parameter name.

        We check that the prior's dist type is visible rather than asserting
        an exact format.
        """
        fr = _minimal_cv_result()
        fr.priors["m"] = Prior(2.5, 1.0)
        s  = repr(fr)
        # The Prior repr should include something like "N[2.5, 1.0]" or "normal".
        assert "m" in s
        # At minimum the prior object's repr should be non-empty and present.
        assert repr(fr.priors["m"]) in s

    def test_repr_resample_result_does_not_raise(self):
        """__repr__ must not raise for a FitResult with resample data."""
        fr = _minimal_resample_result()
        s  = repr(fr)
        assert isinstance(s, str)
        assert "A" in s
        assert "E" in s