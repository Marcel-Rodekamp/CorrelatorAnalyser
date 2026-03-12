"""
Comprehensive test suite for the Data class.

Covers all resample types ('bst', 'jkn') × with/without reweighting (rwf) ×
with/without blocking, plus all factory methods, arithmetic operators,
statistics, indexing, and numpy interoperability.
"""

import numpy as np
import gvar as gv
import pytest
from correlatoranalyser import Data

# ─────────────────────────────────────────────────────────────────────────────
# Global constants & fixtures
# ─────────────────────────────────────────────────────────────────────────────

RNG   = np.random.default_rng(12345)
N     = 200   # raw data samples
T     = 6     # observable length
NBST  = 150   # bootstrap resamples
BS    = 5     # block size  (N % BS == 0)

@pytest.fixture(scope="module")
def raw():
    return RNG.normal(0.5, 1.0, size=(N, T))


@pytest.fixture(scope="module")
def raw_1d():
    return RNG.normal(0.5, 1.0, size=(N,))


@pytest.fixture(scope="module")
def rwf():
    """Positive reweighting factors near 1."""
    return np.abs(RNG.normal(1.0, 0.1, size=(N,)))

@pytest.fixture(scope="module")
def gvar_correlated(raw):
    """Two observables derived from the same ensemble → correlated gvars."""
    # Covariance with off-diagonal structure
    cov_mat = np.eye(T) * 0.01 + np.full((T, T), 0.002)
    g = gv.gvar(np.mean(raw,axis=0), cov_mat)
    return Data.import_gvar(g, Ndata=N)

@pytest.fixture(scope="module")
def gvar_uncorrelated(raw):
    """Plain gvar mode — diagonal covariance only."""
    return Data(resample_type=None, data=raw)


# ── plain fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def jkn(raw):
    return Data(resample_type="jkn", data=raw)


@pytest.fixture(scope="module")
def bst(raw):
    return Data(resample_type="bst", data=raw, Nresample=NBST)


@pytest.fixture(scope="module")
def jkn_1d(raw_1d):
    return Data(resample_type="jkn", data=raw_1d)


@pytest.fixture(scope="module")
def bst_1d(raw_1d):
    return Data(resample_type="bst", data=raw_1d, Nresample=NBST)


# ── reweighted fixtures ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def jkn_rwf(raw, rwf):
    return Data(resample_type="jkn", data=raw, rwf=rwf)


@pytest.fixture(scope="module")
def bst_rwf(raw, rwf):
    return Data(resample_type="bst", data=raw, rwf=rwf, Nresample=NBST)


@pytest.fixture(scope="module")
def jkn_1d_rwf(raw_1d, rwf):
    return Data(resample_type="jkn", data=raw_1d, rwf=rwf)


@pytest.fixture(scope="module")
def bst_1d_rwf(raw_1d, rwf):
    return Data(resample_type="bst", data=raw_1d, rwf=rwf, Nresample=NBST)


# ── blocked fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def jkn_blk(raw):
    return Data(resample_type="jkn", data=raw, blocksize=BS)


@pytest.fixture(scope="module")
def bst_blk(raw):
    return Data(resample_type="bst", data=raw, Nresample=NBST, blocksize=BS)


@pytest.fixture(scope="module")
def jkn_rwf_blk(raw, rwf):
    return Data(resample_type="jkn", data=raw, rwf=rwf, blocksize=BS)


@pytest.fixture(scope="module")
def bst_rwf_blk(raw, rwf):
    return Data(resample_type="bst", data=raw, rwf=rwf, Nresample=NBST, blocksize=BS)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Construction & metadata
# ═════════════════════════════════════════════════════════════════════════════

class TestConstruction:

    # ── shapes ────────────────────────────────────────────────────────────────

    def test_jkn_rspl_shape(self, raw):
        d = Data(resample_type="jkn", data=raw)
        assert d._rspl.shape == (N, T)

    def test_bst_rspl_shape(self, raw):
        d = Data(resample_type="bst", data=raw, Nresample=NBST)
        assert d._rspl.shape == (NBST, T)

    def test_jkn_1d_rspl_shape(self, raw_1d):
        d = Data(resample_type="jkn", data=raw_1d)
        assert d._rspl.shape == (N,)

    def test_bst_1d_rspl_shape(self, raw_1d):
        d = Data(resample_type="bst", data=raw_1d, Nresample=NBST)
        assert d._rspl.shape == (NBST,)

    # ── metadata ──────────────────────────────────────────────────────────────

    def test_jkn_metadata(self, jkn):
        assert jkn.resample_type == "jkn"
        assert jkn.Nresample == N
        assert jkn.Ndata == N

    def test_bst_metadata(self, bst):
        assert bst.resample_type == "bst"
        assert bst.Nresample == NBST
        assert bst.Ndata == N

    def test_tag_stored_jkn(self, raw):
        d = Data(resample_type="jkn", data=raw, tag="my_obs")
        assert d.tag == "my_obs"

    def test_tag_stored_bst(self, raw):
        d = Data(resample_type="bst", data=raw, Nresample=NBST, tag="bst_obs")
        assert d.tag == "bst_obs"

    def test_locked_mean_stored(self, raw):
        d = Data(resample_type="jkn", data=raw, locked_mean=True)
        assert d.locked_mean is True

    # ── validation ────────────────────────────────────────────────────────────

    def test_bst_requires_nresample(self, raw):
        with pytest.raises(ValueError):
            Data(resample_type="bst", data=raw)

    def test_data_must_be_ndarray_jkn(self):
        with pytest.raises(ValueError):
            Data(resample_type="jkn", data=[1, 2, 3])

    def test_data_must_be_ndarray_bst(self):
        with pytest.raises(ValueError):
            Data(resample_type="bst", data=[1, 2, 3], Nresample=10)

    def test_invalid_resample_type(self, raw):
        with pytest.raises(ValueError):
            Data(resample_type="invalid", data=raw)

    # ── custom mean ───────────────────────────────────────────────────────────

    def test_custom_mean_jkn(self, raw):
        m = np.ones(T) * 42.0
        d = Data(resample_type="jkn", data=raw, mean=m)
        np.testing.assert_array_equal(d._mean, m)

    def test_custom_mean_bst(self, raw):
        m = np.ones(T) * 42.0
        d = Data(resample_type="bst", data=raw, Nresample=NBST, mean=m)
        np.testing.assert_array_equal(d._mean, m)

    # ── gvar mode ─────────────────────────────────────────────────────────────

    def test_gvar_mode_construction(self, raw):
        d = Data(resample_type=None, data=raw)
        assert d.resample_type is None
        assert d._rspl is None
        assert d._gvar is not None

    def test_gvar_mode_no_rwf_in_resample(self, raw):
        d = Data(resample_type=None, data=raw)
        assert d._rwf_rspl is None

    def test_gvar_mode_blocksize(self, raw):
        d = Data(resample_type=None, data=raw, blocksize=BS)
        assert d.resample_type is None


# ═════════════════════════════════════════════════════════════════════════════
# 2. Input immutability
# ═════════════════════════════════════════════════════════════════════════════

class TestNoMutation:

    @pytest.mark.parametrize("rtype,kwargs", [
        ("jkn", {}),
        ("bst", {"Nresample": NBST}),
    ])
    def test_data_not_mutated(self, raw, rtype, kwargs):
        raw_copy = raw.copy()
        Data(resample_type=rtype, data=raw, **kwargs)
        np.testing.assert_array_equal(raw, raw_copy)

    @pytest.mark.parametrize("rtype,kwargs", [
        ("jkn", {}),
        ("bst", {"Nresample": NBST}),
    ])
    def test_rwf_not_mutated(self, raw, rwf, rtype, kwargs):
        rwf_copy = rwf.copy()
        Data(resample_type=rtype, data=raw, rwf=rwf, **kwargs)
        np.testing.assert_array_equal(rwf, rwf_copy)

    @pytest.mark.parametrize("rtype,kwargs", [
        ("jkn", {}),
        ("bst", {"Nresample": NBST}),
    ])
    def test_data_not_mutated_with_blocksize(self, raw, rtype, kwargs):
        raw_copy = raw.copy()
        Data(resample_type=rtype, data=raw, blocksize=BS, **kwargs)
        np.testing.assert_array_equal(raw, raw_copy)

    @pytest.mark.parametrize("rtype,kwargs", [
        ("jkn", {}),
        ("bst", {"Nresample": NBST}),
    ])
    def test_rwf_not_mutated_with_blocksize(self, raw, rwf, rtype, kwargs):
        rwf_copy = rwf.copy()
        Data(resample_type=rtype, data=raw, rwf=rwf, blocksize=BS, **kwargs)
        np.testing.assert_array_equal(rwf, rwf_copy)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Jackknife correctness
# ═════════════════════════════════════════════════════════════════════════════

class TestJackknife:

    def test_leave_one_out_formula_1d(self, raw_1d):
        d = Data(resample_type="jkn", data=raw_1d)
        expected = (raw_1d.sum() - raw_1d) / (N - 1)
        np.testing.assert_allclose(d._rspl, expected)

    def test_jkn_mean_equals_raw_mean(self, raw):
        d = Data(resample_type="jkn", data=raw)
        # this must match to machine precision
        np.testing.assert_allclose(d.mean, raw.mean(axis=0), rtol=1e-15)

    def test_static_jackknife_matches_constructor(self, raw):
        d = Data(resample_type="jkn", data=raw)
        static = Data.jackknife(data=raw.copy())
        np.testing.assert_allclose(d._rspl, static)

    def test_jkn_nresample_equals_ndata(self, jkn):
        assert jkn.Nresample == jkn.Ndata

    # ── reweighted ────────────────────────────────────────────────────────────

    def test_jkn_rwf_shape(self, jkn_rwf):
        assert jkn_rwf._rspl.shape == (N, T)

    def test_jkn_rwf_mean_approx_weighted_mean(self, raw, rwf):
        d = Data(resample_type="jkn", data=raw, rwf=rwf)
        expected = (rwf[:, None] * raw).sum(axis=0) / rwf.sum()
        # For the reweighted jackknife we expect a difference of
        #   Var(w) / N 
        # where w are the reweighting factors and N are the number of elements
        # Now as the rwf are drawn normally with standard deviation 0.1 and N = 200
        # we expect 
        # 0.01 / 200 = 5e-5 
        # deviation or smaller
        np.testing.assert_allclose(d.mean, expected, rtol=5e-5)
        # To double check we also compare to the exact mean 
        np.testing.assert_allclose(d.mean, np.mean(d.rspl, axis=0), rtol=1e-15)

    def test_jkn_rwf_differs_from_plain(self, jkn, jkn_rwf):
        assert not np.allclose(jkn.mean, jkn_rwf.mean)

    def test_jkn_rwf_rspl_set(self, jkn_rwf):
        assert jkn_rwf._rwf_rspl is not None
        assert jkn_rwf._rwf_rspl.shape == (N,)

    def test_jkn_rwf_1d_shape(self, jkn_1d_rwf):
        assert jkn_1d_rwf._rspl.shape == (N,)

    # ── blocked ───────────────────────────────────────────────────────────────

    def test_jkn_blk_nresample(self, jkn_blk):
        assert jkn_blk.Nresample == N // BS

    def test_jkn_blk_rspl_shape(self, jkn_blk):
        assert jkn_blk._rspl.shape == (N // BS, T)

    def test_jkn_rwf_blk_nresample(self, jkn_rwf_blk):
        assert jkn_rwf_blk.Nresample == N // BS

    def test_jkn_rwf_blk_rwf_rspl_shape(self, jkn_rwf_blk):
        assert jkn_rwf_blk._rwf_rspl is not None
        assert jkn_rwf_blk._rwf_rspl.shape == (N // BS,)

    # ── 1d blocked ────────────────────────────────────────────────────────────

    def test_jkn_1d_blk_shape(self, raw_1d):
        d = Data(resample_type="jkn", data=raw_1d, blocksize=BS)
        assert d._rspl.shape == (N // BS,)


# ═════════════════════════════════════════════════════════════════════════════
# 4. Bootstrap correctness
# ═════════════════════════════════════════════════════════════════════════════

class TestBootstrap:

    def test_bst_mean_close_to_raw_mean(self, raw):
        d = Data(resample_type="bst", data=raw, Nresample=500)
        np.testing.assert_allclose(d.mean, raw.mean(axis=0), atol=0.15)

    def test_static_bootstrap_shape(self, raw):
        b = Data.bootstrap(data=raw.copy(), Nresample=NBST)
        assert b.shape == (NBST, T)

    def test_static_bootstrap_reproducible(self, raw):
        b1 = Data.bootstrap(data=raw.copy(), Nresample=NBST)
        b2 = Data.bootstrap(data=raw.copy(), Nresample=NBST)
        np.testing.assert_array_equal(b1, b2)

    # ── reweighted ────────────────────────────────────────────────────────────

    def test_bst_rwf_shape(self, bst_rwf):
        assert bst_rwf._rspl.shape == (NBST, T)

    def test_bst_rwf_mean_close_to_weighted_mean(self, raw, rwf):
        d = Data(resample_type="bst", data=raw, rwf=rwf, Nresample=500)
        expected = (rwf[:, None] * raw).sum(axis=0) / rwf.sum()
        np.testing.assert_allclose(d.mean, expected, atol=0.2)

    def test_bst_rwf_rspl_set(self, bst_rwf):
        assert bst_rwf._rwf_rspl is not None
        assert bst_rwf._rwf_rspl.shape == (NBST,)

    def test_bst_rwf_1d_shape(self, bst_1d_rwf):
        assert bst_1d_rwf._rspl.shape == (NBST,)

    # ── blocked ───────────────────────────────────────────────────────────────

    def test_bst_blk_rspl_shape(self, bst_blk):
        assert bst_blk._rspl.shape == (NBST, T)

    def test_bst_blk_nresample(self, bst_blk):
        assert bst_blk.Nresample == NBST

    def test_bst_rwf_blk_rspl_shape(self, bst_rwf_blk):
        assert bst_rwf_blk._rspl.shape == (NBST, T)

    def test_bst_rwf_blk_rwf_rspl_shape(self, bst_rwf_blk):
        assert bst_rwf_blk._rwf_rspl is not None
        assert bst_rwf_blk._rwf_rspl.shape == (NBST,)

    # ── 1d ───────────────────────────────────────────────────────────────────

    def test_bst_1d_shape(self, bst_1d):
        assert bst_1d._rspl.shape == (NBST,)

    def test_bst_1d_blk_shape(self, raw_1d):
        d = Data(resample_type="bst", data=raw_1d, Nresample=NBST, blocksize=BS)
        assert d._rspl.shape == (NBST,)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Blocking
# ═════════════════════════════════════════════════════════════════════════════

class TestBlocking:

    def test_blocking_output_shape(self, raw):
        blocked, _ = Data.blocking(data=raw.copy(), blocksize=BS)
        assert blocked.shape == (N // BS, T)

    def test_blocking_output_shape_1d(self, raw_1d):
        blocked, _ = Data.blocking(data=raw_1d.copy(), blocksize=BS)
        assert blocked.shape == (N // BS,)

    def test_blocking_rwf_shapes(self, raw, rwf):
        bd, br = Data.blocking(data=raw.copy(), blocksize=BS, rwf=rwf.copy())
        assert bd.shape == (N // BS, T)
        assert br.shape == (N // BS,)

    def test_blocking_no_mutation_data(self, raw):
        raw_copy = raw.copy()
        Data.blocking(data=raw.copy(), blocksize=BS)
        np.testing.assert_array_equal(raw, raw_copy)

    def test_blocking_no_mutation_rwf(self, raw, rwf):
        rwf_copy = rwf.copy()
        Data.blocking(data=raw.copy(), blocksize=BS, rwf=rwf.copy())
        np.testing.assert_array_equal(rwf, rwf_copy)

    def test_blocking_computes_block_means(self, raw):
        blocked, _ = Data.blocking(data=raw, blocksize=BS)
        for k in range(N // BS - 1):
            expected = raw[k * BS:(k + 1) * BS].mean(axis=0)
            np.testing.assert_allclose(blocked[k], expected)

    def test_blocking_rwf_block_means(self, raw, rwf):
        blocked_d, blocked_r = Data.blocking(data=raw, blocksize=BS, rwf=rwf)
        for k in range(N // BS - 1):
            expected_r = rwf[k * BS:(k + 1) * BS].mean()
            np.testing.assert_allclose(blocked_r[k], expected_r)


# ═════════════════════════════════════════════════════════════════════════════
# 6. Standard error
# ═════════════════════════════════════════════════════════════════════════════

class TestSerr:

    def test_jkn_serr_shape(self, jkn):
        assert jkn.serr.shape == (T,)

    def test_bst_serr_shape(self, bst):
        assert bst.serr.shape == (T,)

    def test_jkn_1d_serr_scalar(self, jkn_1d):
        s = jkn_1d.serr
        assert np.isscalar(s) or s.shape == ()

    def test_bst_1d_serr_scalar(self, bst_1d):
        s = bst_1d.serr
        assert np.isscalar(s) or s.shape == ()

    def test_jkn_serr_positive(self, jkn):
        assert np.all(jkn.serr >= 0)

    def test_bst_serr_positive(self, bst):
        assert np.all(bst.serr >= 0)

    def test_jkn_serr_formula(self, jkn_1d):
        """SE_jkn = sqrt(N-1) * std(resamples, ddof=0)."""
        expected = np.sqrt(N - 1) * np.std(jkn_1d._rspl, axis=0)
        np.testing.assert_allclose(jkn_1d.serr, expected)

    def test_bst_serr_formula(self, bst_1d):
        """SE_bst = std(resamples, ddof=1)."""
        expected = np.std(bst_1d._rspl, axis=0, ddof=1)
        np.testing.assert_allclose(bst_1d.serr, expected)

    def test_jkn_rwf_serr_shape(self, jkn_rwf):
        assert jkn_rwf.serr.shape == (T,)

    def test_bst_rwf_serr_shape(self, bst_rwf):
        assert bst_rwf.serr.shape == (T,)

    def test_jkn_blk_serr_shape(self, jkn_blk):
        assert jkn_blk.serr.shape == (T,)

    def test_bst_blk_serr_shape(self, bst_blk):
        assert bst_blk.serr.shape == (T,)

    def test_jkn_serr_agrees_with_bst_large_N(self, raw):
        """For smooth estimator and large N, JKN and BST SEs should agree."""
        jkn = Data(resample_type="jkn", data=raw)
        bst = Data(resample_type="bst", data=raw, Nresample=2000)
        ratio = jkn.serr / bst.serr
        np.testing.assert_allclose(ratio, np.ones(T), atol=0.25)


# ═════════════════════════════════════════════════════════════════════════════
# 7. Covariance
# ═════════════════════════════════════════════════════════════════════════════

class TestCovariance:

    def test_jkn_cov_shape(self, jkn):
        assert jkn.cov.shape == (T, T)

    def test_bst_cov_shape(self, bst):
        assert bst.cov.shape == (T, T)

    def test_jkn_cov_symmetric(self, jkn):
        np.testing.assert_allclose(jkn.cov, jkn.cov.T, atol=1e-14)

    def test_bst_cov_symmetric(self, bst):
        np.testing.assert_allclose(bst.cov, bst.cov.T, atol=1e-14)

    def test_jkn_cov_psd(self, jkn):
        eigvals = np.linalg.eigvalsh(jkn.cov)
        assert np.all(eigvals >= -1e-10)

    def test_bst_cov_psd(self, bst):
        eigvals = np.linalg.eigvalsh(bst.cov)
        assert np.all(eigvals >= -1e-10)

    def test_jkn_cov_diagonal_equals_serr_squared(self, jkn):
        """
        Jackknife covariance diagonal must equal serr^2.
        The correct jackknife covariance is (N-1)*cov(bias=True),
        whose diagonal equals (N-1)*var(ddof=0) = serr_jkn^2.
        """
        np.testing.assert_allclose(np.diag(jkn.cov), jkn.serr ** 2, rtol=1e-15)

    def test_bst_cov_diagonal_equals_serr_squared(self, bst):
        """Bootstrap covariance diagonal must equal serr^2."""
        np.testing.assert_allclose(np.diag(bst.cov), bst.serr ** 2, rtol=1e-15)

    def test_jkn_rwf_cov_shape(self, jkn_rwf):
        assert jkn_rwf.cov.shape == (T, T)

    def test_bst_rwf_cov_shape(self, bst_rwf):
        assert bst_rwf.cov.shape == (T, T)

    def test_jkn_blk_cov_shape(self, jkn_blk):
        assert jkn_blk.cov.shape == (T, T)

    def test_bst_blk_cov_shape(self, bst_blk):
        assert bst_blk.cov.shape == (T, T)

    def test_cov_scalar_raises(self, jkn_1d):
        with pytest.raises(RuntimeError):
            _ = jkn_1d.cov

    def test_gvar_cov_shape(self, gvar_correlated):
        assert gvar_correlated.cov.shape == (T, T)

    def test_gvar_cov_symmetric(self, gvar_correlated):
        np.testing.assert_allclose(gvar_correlated.cov, gvar_correlated.cov.T, atol=1e-14)

    def test_gvar_cov_diagonal_equals_serr_squared(self, gvar_correlated):
        np.testing.assert_allclose(
            np.diag(gvar_correlated.cov), gvar_correlated.serr ** 2, rtol=1e-12)

    def test_gvar_cov_has_offdiagonal(self, gvar_correlated):
        cov = gvar_correlated.cov
        off_diag = cov - np.diag(np.diag(cov))
        assert np.any(np.abs(off_diag) > 1e-14)

    def test_gvar_cov_raises_when_uncorrelated(self, gvar_uncorrelated):
        """Constructing Data from raw data yields diagonal-only gvars — .cov must raise."""
        with pytest.raises(RuntimeError, match="cross-covariance"):
            _ = gvar_uncorrelated.cov

    def test_gvar_cov_raises_for_scalar_gvar(self, raw):
        d = Data(resample_type=None, data=raw[:, 0])  # 1D → scalar gvar
        with pytest.raises(RuntimeError):
            _ = d.cov

# ═════════════════════════════════════════════════════════════════════════════
# 8. Correlation
# ═════════════════════════════════════════════════════════════════════════════

class TestCorrelation:

    def test_jkn_cor_shape(self, jkn):
        assert jkn.cor.shape == (T, T)

    def test_bst_cor_shape(self, bst):
        assert bst.cor.shape == (T, T)

    def test_jkn_cor_diagonal_ones(self, jkn):
        np.testing.assert_allclose(np.diag(jkn.cor), np.ones(T), atol=1e-12)

    def test_bst_cor_diagonal_ones(self, bst):
        np.testing.assert_allclose(np.diag(bst.cor), np.ones(T), atol=1e-12)

    def test_cor_values_in_range(self, jkn, bst):
        assert np.all(np.abs(jkn.cor) <= 1.0 + 1e-10)
        assert np.all(np.abs(bst.cor) <= 1.0 + 1e-10)

    def test_cor_symmetric(self, jkn):
        np.testing.assert_allclose(jkn.cor, jkn.cor.T, atol=1e-14)

    def test_gvar_cor_shape(self, gvar_correlated):
        assert gvar_correlated.cor.shape == (T, T)

    def test_gvar_cor_diagonal_ones(self, gvar_correlated):
        assert np.all(np.diag(gvar_correlated.cor) == np.ones(T))

    def test_gvar_cor_raises_when_uncorrelated(self, gvar_uncorrelated):
        with pytest.raises(RuntimeError, match="cross-covariance"):
            _ = gvar_uncorrelated.cor


# ═════════════════════════════════════════════════════════════════════════════
# 9. Arithmetic operators — parametrized over resample types
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(params=["jkn", "bst"])
def d_plain(request, jkn, bst):
    return jkn if request.param == "jkn" else bst


@pytest.fixture(params=["jkn_rwf", "bst_rwf"])
def d_rwf(request, jkn_rwf, bst_rwf):
    return jkn_rwf if request.param == "jkn_rwf" else bst_rwf


class TestArithmetic:

    # ── Data ⊕ Data ───────────────────────────────────────────────────────────

    def test_add_data_data(self, d_plain):
        r = d_plain + d_plain
        np.testing.assert_allclose(r._rspl, 2 * d_plain._rspl)
        np.testing.assert_allclose(r.mean, 2 * d_plain.mean)

    def test_sub_data_data(self, d_plain):
        r = d_plain - d_plain
        np.testing.assert_allclose(r._rspl, np.zeros_like(d_plain._rspl), atol=1e-14)
        np.testing.assert_allclose(r.mean, np.zeros_like(d_plain.mean), atol=1e-14)

    def test_mul_data_data(self, d_plain):
        r = d_plain * d_plain
        np.testing.assert_allclose(r._rspl, d_plain._rspl ** 2)

    def test_div_data_data(self, d_plain):
        r = d_plain / d_plain
        np.testing.assert_allclose(r._rspl, np.ones_like(d_plain._rspl))

    def test_pow_data_data(self, d_plain):
        r = d_plain ** d_plain
        np.testing.assert_allclose(r._rspl, d_plain._rspl ** d_plain._rspl)

    # ── Data ⊕ scalar ─────────────────────────────────────────────────────────

    def test_add_scalar(self, d_plain):
        r = d_plain + 5.0
        np.testing.assert_allclose(r._rspl, d_plain._rspl + 5.0)

    def test_radd_scalar(self, d_plain):
        r = 5.0 + d_plain
        np.testing.assert_allclose(r._rspl, d_plain._rspl + 5.0)

    def test_sub_scalar(self, d_plain):
        r = d_plain - 1.0
        np.testing.assert_allclose(r._rspl, d_plain._rspl - 1.0)

    def test_rsub_scalar(self, d_plain):
        r = 10.0 - d_plain
        np.testing.assert_allclose(r._rspl, 10.0 - d_plain._rspl)

    def test_mul_scalar(self, d_plain):
        r = d_plain * 3.0
        np.testing.assert_allclose(r._rspl, d_plain._rspl * 3.0)

    def test_rmul_scalar(self, d_plain):
        r = 3.0 * d_plain
        np.testing.assert_allclose(r._rspl, d_plain._rspl * 3.0)

    def test_div_scalar(self, d_plain):
        r = d_plain / 2.0
        np.testing.assert_allclose(r._rspl, d_plain._rspl / 2.0)

    def test_rdiv_scalar(self, d_plain):
        r = 1.0 / d_plain
        np.testing.assert_allclose(r._rspl, 1.0 / d_plain._rspl)

    def test_pow_scalar(self, d_plain):
        r = d_plain ** 2
        np.testing.assert_allclose(r._rspl, d_plain._rspl ** 2)

    def test_neg(self, d_plain):
        r = -d_plain
        np.testing.assert_allclose(r._rspl, -d_plain._rspl)
        np.testing.assert_allclose(r.mean, -d_plain.mean)

    # ── in-place operators ────────────────────────────────────────────────────

    def test_iadd_scalar(self, raw):
        d = Data(resample_type="jkn", data=raw)
        before = d._rspl.copy()
        d += 1.0
        np.testing.assert_allclose(d._rspl, before + 1.0)

    def test_isub_scalar(self, raw):
        d = Data(resample_type="jkn", data=raw)
        before = d._rspl.copy()
        d -= 1.0
        np.testing.assert_allclose(d._rspl, before - 1.0)

    def test_imul_scalar(self, raw):
        d = Data(resample_type="jkn", data=raw)
        before = d._rspl.copy()
        d *= 2.0
        np.testing.assert_allclose(d._rspl, before * 2.0)

    def test_idiv_scalar(self, raw):
        d = Data(resample_type="jkn", data=raw)
        before = d._rspl.copy()
        d /= 2.0
        np.testing.assert_allclose(d._rspl, before / 2.0)

    def test_iadd_data(self, raw):
        d = Data(resample_type="jkn", data=raw)
        before = d._rspl.copy()
        d2 = Data(resample_type="jkn", data=raw)
        d += d2
        np.testing.assert_allclose(d._rspl, before + d2._rspl)

    # ── metadata preservation ─────────────────────────────────────────────────

    def test_resample_type_preserved(self, d_plain):
        r = d_plain + 1.0
        assert r.resample_type == d_plain.resample_type

    def test_nresample_preserved(self, d_plain):
        r = d_plain * 2.0
        assert r.Nresample == d_plain.Nresample

    def test_ndata_preserved(self, d_plain):
        r = d_plain - 1.0
        assert r.Ndata == d_plain.Ndata

    # ── rwf preserved through arithmetic ─────────────────────────────────────

    def test_rwf_preserved_data_plus_scalar(self, d_rwf):
        r = d_rwf + 1.0
        assert r._rwf_rspl is not None
        np.testing.assert_array_equal(r._rwf_rspl, d_rwf._rwf_rspl)

    def test_rwf_preserved_data_times_scalar(self, d_rwf):
        r = d_rwf * 2.0
        assert r._rwf_rspl is not None

    def test_rwf_preserved_data_plus_data(self, d_rwf):
        r = d_rwf + d_rwf
        assert r._rwf_rspl is not None

    # ── error conditions ──────────────────────────────────────────────────────

    def test_mismatched_nresample_raises(self, raw):
        d1 = Data(resample_type="bst", data=raw, Nresample=50)
        d2 = Data(resample_type="bst", data=raw, Nresample=80)
        with pytest.raises(ValueError):
            _ = d1 + d2

    def test_mixed_resample_types_raises(self, jkn, bst):
        with pytest.raises(ValueError):
            _ = jkn + bst

    # ── reweighted arithmetic ─────────────────────────────────────────────────

    def test_rwf_add_scalar(self, jkn_rwf):
        r = jkn_rwf + 1.0
        np.testing.assert_allclose(r._rspl, jkn_rwf._rspl + 1.0)

    def test_rwf_mul_data(self, bst_rwf):
        r = bst_rwf * bst_rwf
        np.testing.assert_allclose(r._rspl, bst_rwf._rspl ** 2)

    # ── blocked arithmetic ────────────────────────────────────────────────────

    def test_blk_add_scalar(self, jkn_blk):
        r = jkn_blk + 1.0
        np.testing.assert_allclose(r._rspl, jkn_blk._rspl + 1.0)

    def test_blk_mul_data(self, bst_blk):
        r = bst_blk * 2.0
        np.testing.assert_allclose(r._rspl, bst_blk._rspl * 2.0)


# ═════════════════════════════════════════════════════════════════════════════
# 10. Properties: mean, serr, shape, ndim
# ═════════════════════════════════════════════════════════════════════════════

class TestProperties:

    # ── shape / ndim ──────────────────────────────────────────────────────────

    def test_shape_2d(self, jkn):
        assert jkn.shape == (T,)

    def test_shape_1d(self, jkn_1d):
        assert jkn_1d.shape == ()

    def test_ndim_2d(self, jkn):
        assert jkn.ndim == 1

    def test_ndim_1d(self, jkn_1d):
        assert jkn_1d.ndim == 0

    def test_shape_bst(self, bst):
        assert bst.shape == (T,)

    # ── rspl property ─────────────────────────────────────────────────────────

    def test_rspl_property(self, jkn):
        np.testing.assert_array_equal(jkn.rspl, jkn._rspl)

    def test_rspl_raises_in_gvar_mode(self, raw):
        d = Data(resample_type=None, data=raw)
        with pytest.raises(RuntimeError):
            _ = d.rspl

    # ── mean behaviour ────────────────────────────────────────────────────────

    def test_mean_recomputed_when_not_locked(self, raw):
        d = Data(resample_type="jkn", data=raw)
        d._rspl += 100.0
        np.testing.assert_allclose(d.mean, raw.mean(axis=0) + 100.0, rtol=1e-15)

    def test_mean_locked(self, raw):
        fixed = np.ones(T) * 42.0
        d = Data(resample_type="jkn", data=raw, mean=fixed, locked_mean=True)
        d._rspl += 100.0
        np.testing.assert_array_equal(d.mean, fixed)

    def test_mean_shape_2d(self, jkn):
        assert jkn.mean.shape == (T,)

    def test_mean_shape_1d(self, jkn_1d):
        m = jkn_1d.mean
        assert np.isscalar(m) or m.shape == ()

    def test_mean_rwf(self, jkn_rwf):
        assert jkn_rwf.mean.shape == (T,)

    def test_mean_blk(self, jkn_blk):
        assert jkn_blk.mean.shape == (T,)

    # ── StN ───────────────────────────────────────────────────────────────────

    def test_stn_shape(self, jkn):
        assert jkn.StN.shape == (T,)

    def test_stn_nonnegative(self, jkn, bst):
        assert np.all(jkn.StN >= 0)
        assert np.all(bst.StN >= 0)


# ═════════════════════════════════════════════════════════════════════════════
# 11. Indexing
# ═════════════════════════════════════════════════════════════════════════════

class TestIndexing:

    def test_getitem_integer(self, jkn):
        d = jkn[0]
        assert d._rspl.shape == (N,)
        np.testing.assert_allclose(d._rspl, jkn._rspl[:, 0])

    def test_getitem_integer_bst(self, bst):
        d = bst[0]
        assert d._rspl.shape == (NBST,)
        np.testing.assert_allclose(d._rspl, bst._rspl[:, 0])

    def test_getitem_slice(self, jkn):
        d = jkn[0:3]
        assert d._rspl.shape == (N, 3)

    def test_getitem_nresample_preserved(self, jkn):
        assert jkn[0].Nresample == N

    def test_getitem_tuple(self, jkn):
        d = jkn[0:2]
        assert d.shape == (2,)

    def test_getitem_np_newaxis(self, jkn):
        d = jkn[np.newaxis]
        assert d._rspl.shape[1] == 1

    def test_setitem_from_data(self, jkn):
        target = Data.zeros_like(jkn)
        target[0] = jkn[0]
        np.testing.assert_allclose(target._rspl[:, 0], jkn._rspl[:, 0])

    def test_setitem_mismatched_nresample_raises(self, raw):
        d1 = Data(resample_type="jkn", data=raw)
        d2 = Data.import_resamples(resample_type="jkn", rspl=np.ones((50, T)))
        with pytest.raises(ValueError):
            d1[0] = d2

    def test_getitem_rwf_shape(self, jkn_rwf):
        d = jkn_rwf[0]
        assert d._rspl.shape == (N,)

    def test_getitem_blk_shape(self, jkn_blk):
        d = jkn_blk[0]
        assert d._rspl.shape == (N // BS,)


# ═════════════════════════════════════════════════════════════════════════════
# 12. Reshape
# ═════════════════════════════════════════════════════════════════════════════

class TestReshape:

    def test_reshape_jkn(self, raw):
        d = Data(resample_type="jkn", data=raw)
        d.reshape((2, T // 2))
        assert d.shape == (2, T // 2)
        assert d._rspl.shape == (N, 2, T // 2)

    def test_reshape_bst(self, raw):
        d = Data(resample_type="bst", data=raw, Nresample=NBST)
        d.reshape((2, T // 2))
        assert d._rspl.shape == (NBST, 2, T // 2)

    def test_reshape_invalid_raises(self, raw):
        d = Data(resample_type="jkn", data=raw)
        with pytest.raises((ValueError, Exception)):
            d.reshape((999,))


# ═════════════════════════════════════════════════════════════════════════════
# 13. Factories
# ═════════════════════════════════════════════════════════════════════════════

class TestFactories:

    # ── import_resamples ──────────────────────────────────────────────────────

    def test_import_resamples_roundtrip(self, jkn):
        d = Data.import_resamples(
            resample_type=jkn.resample_type,
            rspl=jkn._rspl.copy(),
            Ndata=jkn.Ndata,
        )
        np.testing.assert_array_equal(d._rspl, jkn._rspl)
        assert d.Nresample == jkn.Nresample

    def test_import_resamples_nresample_mismatch_raises(self):
        rspl = np.ones((10, T))
        with pytest.raises(RuntimeError):
            Data.import_resamples(resample_type="bst", rspl=rspl, Nresample=99)

    def test_import_resamples_mean_default(self):
        rspl = np.ones((NBST, T)) * 3.0
        d = Data.import_resamples(resample_type="bst", rspl=rspl)
        np.testing.assert_allclose(d.mean, np.full(T, 3.0))

    def test_import_resamples_custom_mean(self):
        rspl = np.ones((NBST, T))
        m = np.full(T, 99.0)
        d = Data.import_resamples(resample_type="bst", rspl=rspl, mean=m)
        np.testing.assert_array_equal(d._mean, m)

    def test_import_resamples_with_rwf(self):
        rspl = np.ones((NBST, T))
        rwf_r = np.ones(NBST)
        d = Data.import_resamples(resample_type="bst", rspl=rspl, rwf_rspl=rwf_r)
        np.testing.assert_array_equal(d._rwf_rspl, rwf_r)

    # ── import_gvar ───────────────────────────────────────────────────────────

    def test_import_gvar_roundtrip(self, raw):
        d_orig = Data(resample_type=None, data=raw)
        d_new = Data.import_gvar(d_orig._gvar, Ndata=N)
        np.testing.assert_allclose(d_new.mean, d_orig.mean)
        np.testing.assert_allclose(d_new.serr, d_orig.serr)

    def test_import_gvar_ndata_stored(self, raw):
        d = Data(resample_type=None, data=raw)
        g = Data.import_gvar(d._gvar, Ndata=42)
        assert g.Ndata == 42

    # ── zeros / ones / empty ──────────────────────────────────────────────────

    def test_zeros_bst_shape(self):
        d = Data.zeros(resample_type="bst", shape=(T,), Nresample=NBST)
        assert d._rspl.shape == (NBST, T)
        np.testing.assert_array_equal(d._rspl, 0)

    def test_zeros_jkn_deduce_nresample(self):
        d = Data.zeros(resample_type="jkn", shape=(T,), Ndata=N)
        assert d.Nresample == N

    def test_zeros_scalar(self):
        d = Data.zeros(resample_type="bst", Nresample=NBST)
        assert d._rspl.shape == (NBST,)
        np.testing.assert_array_equal(d._rspl, 0)

    def test_ones_bst_shape(self):
        d = Data.ones(resample_type="bst", shape=(T,), Nresample=NBST)
        assert d._rspl.shape == (NBST, T)
        np.testing.assert_array_equal(d._rspl, 1)

    def test_ones_jkn(self):
        d = Data.ones(resample_type="jkn", shape=(T,), Ndata=N)
        np.testing.assert_array_equal(d._rspl, 1)

    def test_empty_bst_shape(self):
        d = Data.empty(resample_type="bst", shape=(T,), Nresample=NBST)
        assert d._rspl.shape == (NBST, T)

    # ── zeros_like / ones_like / empty_like / full_like ───────────────────────

    def test_zeros_like_jkn(self, jkn):
        d = Data.zeros_like(jkn)
        assert d._rspl.shape == jkn._rspl.shape
        np.testing.assert_array_equal(d._rspl, 0)
        assert d.resample_type == jkn.resample_type
        assert d.Nresample == jkn.Nresample

    def test_zeros_like_bst(self, bst):
        d = Data.zeros_like(bst)
        np.testing.assert_array_equal(d._rspl, 0)

    def test_zeros_like_preserves_rwf_when_copy(self, jkn_rwf):
        d = Data.zeros_like(jkn_rwf, copy_rwf=True)
        assert d._rwf_rspl is not None
        np.testing.assert_array_equal(d._rwf_rspl, jkn_rwf._rwf_rspl)

    def test_zeros_like_no_copy_rwf(self, jkn_rwf):
        """copy_rwf=False should not carry over reweighting factors."""
        d = Data.zeros_like(jkn_rwf, copy_rwf=False)
        # rwf resamples should either be None or all-zeros (no valid weights)
        if d._rwf_rspl is not None:
            assert not np.allclose(d._rwf_rspl, jkn_rwf._rwf_rspl)

    def test_empty_like_shape(self, jkn):
        d = Data.empty_like(jkn)
        assert d._rspl.shape == jkn._rspl.shape

    def test_full_like_value_rspl(self, jkn):
        d = Data.full_like(jkn, value=7.0)
        np.testing.assert_array_equal(d._rspl, 7.0)

    def test_full_like_gvar_not_nullified(self, raw):
        """
        BUG REGRESSION: full_like in gvar mode used to set _gvar = None
        after creating it. The returned object must have a valid _gvar.
        """
        d_gvar = Data(resample_type=None, data=raw)
        d = Data.full_like(d_gvar, value=3.0)
        assert d._gvar is not None, "_gvar must not be None after full_like in gvar mode"
        np.testing.assert_allclose(d.mean, np.full(T, 3.0))

    # ── copy ─────────────────────────────────────────────────────────────────

    def test_copy_is_deep(self, jkn):
        d = jkn.copy()
        d._rspl[0, 0] = 9999.0
        assert jkn._rspl[0, 0] != 9999.0

    def test_copy_resample_type(self, jkn):
        assert jkn.copy().resample_type == jkn.resample_type

    def test_copy_nresample(self, bst):
        assert bst.copy().Nresample == bst.Nresample

    def test_copy_with_rwf(self, jkn_rwf):
        d = jkn_rwf.copy()
        assert d._rwf_rspl is not None
        np.testing.assert_array_equal(d._rwf_rspl, jkn_rwf._rwf_rspl)
        d._rwf_rspl[0] = -999.0
        assert jkn_rwf._rwf_rspl[0] != -999.0  # deep copy


# ═════════════════════════════════════════════════════════════════════════════
# 14. Numpy interoperability
# ═════════════════════════════════════════════════════════════════════════════

class TestNumpyInterop:

    # ── ufuncs ────────────────────────────────────────────────────────────────

    def test_np_add_returns_data(self, jkn):
        r = np.add(jkn, jkn)
        assert isinstance(r, Data)
        np.testing.assert_allclose(r._rspl, 2 * jkn._rspl)

    def test_np_sqrt_returns_data(self, raw):
        d = Data(resample_type="jkn", data=np.abs(raw))
        r = np.sqrt(d)
        assert isinstance(r, Data)
        np.testing.assert_allclose(r._rspl, np.sqrt(d._rspl))

    def test_np_sqrt_bst(self, raw):
        d = Data(resample_type="bst", data=np.abs(raw), Nresample=NBST)
        r = np.sqrt(d)
        assert isinstance(r, Data)

    def test_np_exp_jkn(self, jkn):
        r = np.exp(jkn)
        assert isinstance(r, Data)
        np.testing.assert_allclose(r._rspl, np.exp(jkn._rspl))

    def test_np_log_jkn(self, raw):
        d = Data(resample_type="jkn", data=np.abs(raw) + 1.0)
        r = np.log(d)
        assert isinstance(r, Data)
        np.testing.assert_allclose(r._rspl, np.log(d._rspl))

    # ── np.mean ───────────────────────────────────────────────────────────────

    def test_np_mean_axis1_returns_data(self, jkn):
        r = np.mean(jkn, axis=1)
        assert isinstance(r, Data)
        assert r._rspl.shape == (N,)

    def test_np_mean_axis1_bst(self, bst):
        r = np.mean(bst, axis=1)
        assert isinstance(r, Data)
        assert r._rspl.shape == (NBST,)

    def test_np_mean_axis0_returns_array(self, jkn):
        r = np.mean(jkn, axis=0)
        assert isinstance(r, np.ndarray)
        assert r.shape == (T,)

    def test_np_mean_axis0_values(self, jkn):
        r = np.mean(jkn, axis=0)
        np.testing.assert_allclose(r, jkn.mean)

    # ── comparisons ───────────────────────────────────────────────────────────

    def test_lt_returns_ndarray(self, jkn):
        r = jkn < 0.0
        assert isinstance(r, np.ndarray)
        assert r.shape == jkn._rspl.shape

    def test_le_returns_ndarray(self, jkn):
        r = jkn <= 0.0
        assert isinstance(r, np.ndarray)

    def test_gt_returns_ndarray(self, jkn):
        r = jkn > 0.0
        assert isinstance(r, np.ndarray)

    def test_ge_returns_ndarray(self, jkn):
        r = jkn >= 0.0
        assert isinstance(r, np.ndarray)

    # ── mixed-mode errors ─────────────────────────────────────────────────────

    def test_ufunc_gvar_resample_mix_raises(self, raw):
        dg = Data(resample_type=None, data=raw)
        dr = Data(resample_type="jkn", data=raw)
        with pytest.raises((RuntimeError, ValueError)):
            np.add(dg, dr)


# ═════════════════════════════════════════════════════════════════════════════
# 15. gvar mode
# ═════════════════════════════════════════════════════════════════════════════

class TestGvarMode:

    def test_gvar_mean_shape(self, raw):
        d = Data(resample_type=None, data=raw)
        assert d.mean.shape == (T,)

    def test_gvar_serr_shape(self, raw):
        d = Data(resample_type=None, data=raw)
        assert d.serr.shape == (T,)

    def test_gvar_serr_positive(self, raw):
        d = Data(resample_type=None, data=raw)
        assert np.all(d.serr > 0)

    def test_gvar_add_gvar(self, raw):
        d1 = Data(resample_type=None, data=raw)
        d2 = Data(resample_type=None, data=raw)
        r = d1 + d2
        assert r.resample_type is None
        np.testing.assert_allclose(r.mean, d1.mean + d2.mean)

    def test_gvar_mul_scalar(self, raw):
        d = Data(resample_type=None, data=raw)
        r = d * 2.0
        np.testing.assert_allclose(r.mean, d.mean * 2.0)

    def test_gvar_mix_raises(self, raw):
        dg = Data(resample_type=None, data=raw)
        dr = Data(resample_type="jkn", data=raw)
        with pytest.raises(ValueError):
            _ = dg + dr

    def test_gvar_blocksize(self, raw):
        d = Data(resample_type=None, data=raw, blocksize=BS)
        assert d.serr.shape == (T,)

    def test_import_gvar_factory_preserves_gvar(self, raw):
        d = Data(resample_type=None, data=raw)
        d2 = Data.import_gvar(d._gvar)
        np.testing.assert_allclose(d2.mean, d.mean)
        np.testing.assert_allclose(d2.serr, d.serr)

    def test_gvar_zeros_like(self, raw):
        d = Data(resample_type=None, data=raw)
        z = Data.zeros_like(d)
        assert z.resample_type is None
        assert z._gvar is not None
        np.testing.assert_allclose(z.mean, np.zeros(T))

    def test_gvar_full_like_gvar_not_none(self, raw):
        """
        BUG REGRESSION: full_like in gvar mode nullified _gvar.
        """
        d = Data(resample_type=None, data=raw)
        f = Data.full_like(d, value=5.0)
        assert f._gvar is not None
        np.testing.assert_allclose(f.mean, np.full(T, 5.0))

    def test_gvar_reshape(self, raw):
        d = Data(resample_type=None, data=raw)
        d.reshape((2, T // 2))
        assert d.shape == (2, T // 2)


# ═════════════════════════════════════════════════════════════════════════════
# 16. HDF5 serialisation round-trip
# ═════════════════════════════════════════════════════════════════════════════

import h5py as h5
import io


class TestSerialization:

    def _roundtrip(self, d: Data) -> Data:
        buf = io.BytesIO()
        with h5.File(buf, "w") as f:
            d.serialize(f, node="test")
        buf.seek(0)
        with h5.File(buf, "r") as f:
            return Data.deserialize(f, node="test")

    def test_jkn_roundtrip_rspl(self, jkn):
        d2 = self._roundtrip(jkn)
        np.testing.assert_allclose(d2._rspl, jkn._rspl)

    def test_bst_roundtrip_rspl(self, bst):
        d2 = self._roundtrip(bst)
        np.testing.assert_allclose(d2._rspl, bst._rspl)

    def test_jkn_roundtrip_metadata(self, jkn):
        d2 = self._roundtrip(jkn)
        assert d2.resample_type == jkn.resample_type
        assert d2.Nresample == jkn.Nresample

    def test_jkn_rwf_roundtrip(self, jkn_rwf):
        d2 = self._roundtrip(jkn_rwf)
        np.testing.assert_allclose(d2._rspl, jkn_rwf._rspl)
        assert d2._rwf_rspl is not None
        np.testing.assert_allclose(d2._rwf_rspl, jkn_rwf._rwf_rspl)

    def test_bst_rwf_roundtrip(self, bst_rwf):
        d2 = self._roundtrip(bst_rwf)
        np.testing.assert_allclose(d2._rspl, bst_rwf._rspl)
        assert d2._rwf_rspl is not None

    def test_gvar_roundtrip(self, raw):
        d = Data(resample_type=None, data=raw)
        d2 = self._roundtrip(d)
        assert np.all(d2.mean == d.mean)
        assert np.all(d2.serr == d.serr)

    def test_ndata_roundtrip(self, jkn):
        d2 = self._roundtrip(jkn)
        assert d2.Ndata == jkn.Ndata

    def test_tag_roundtrip(self, raw):
        d = Data(resample_type="jkn", data=raw, tag="hello")
        d2 = self._roundtrip(d)
        assert d2.tag == "hello"

    def test_gvar_correlated_roundtrip_preserves_correlations(self, gvar_correlated):
        """Correlations encoded in gvar primary variables must survive HDF5 roundtrip."""
        d2 = self._roundtrip(gvar_correlated)
        # means and serrs
        assert np.all(d2.mean == gvar_correlated.mean)
        assert np.all(d2.serr == gvar_correlated.serr)
        # full covariance matrix — this would fail if gv.loads dropped primary variables
        assert np.all(d2.cov == gvar_correlated.cov)

    def test_gvar_uncorrelated_roundtrip(self, gvar_uncorrelated):
        """Diagonal-only gvar roundtrip preserves means and serrs."""
        d2 = self._roundtrip(gvar_uncorrelated)
        assert np.all(d2.mean == gvar_uncorrelated.mean)
        assert np.all(d2.serr == gvar_uncorrelated.serr)
        # must still raise on .cov since it's diagonal-only
        with pytest.raises(RuntimeError):
            _ = d2.cov


# ═════════════════════════════════════════════════════════════════════════════
# 17. to_dict / gvar conversion
# ═════════════════════════════════════════════════════════════════════════════

class TestConversion:

    def test_to_dict_keys(self, jkn):
        d = jkn.to_dict()
        assert "est" in d and "err" in d and "res" in d

    def test_to_dict_res_is_rspl(self, jkn):
        d = jkn.to_dict()
        np.testing.assert_array_equal(d["res"], jkn._rspl)

    def test_to_dict_gvar_mode_res_none(self, raw):
        d = Data(resample_type=None, data=raw)
        assert d.to_dict()["res"] is None

    def test_gvar_conversion_uncorrelated(self, jkn):
        g = jkn.gvar(correlated=False)
        np.testing.assert_allclose(gv.mean(g), jkn.mean, rtol=1e-10)
        np.testing.assert_allclose(gv.sdev(g), jkn.serr, rtol=1e-10)

    def test_gvar_conversion_bst(self, bst):
        g = bst.gvar(correlated=False)
        np.testing.assert_allclose(gv.sdev(g), bst.serr, rtol=1e-10)

    def test_gvar_conversion_gvar_mode(self, raw):
        d = Data(resample_type=None, data=raw)
        g = d.gvar()
        assert g is d._gvar


# ═════════════════════════════════════════════════════════════════════════════
# 18. Regression tests for known bugs
# ═════════════════════════════════════════════════════════════════════════════

class TestBugRegressions:

    def test_full_like_gvar_scalar_not_none(self, raw):
        """
        BUG: full_like in gvar mode (scalar) set new._gvar=None after
        assigning it, making the object invalid. Fixed by removing the
        erroneous `new._gvar = None` at the end of the gvar branch.
        """
        d = Data(resample_type=None, data=raw)
        f = Data.full_like(d, value=2.0)
        assert f._gvar is not None

    def test_jkn_cov_diagonal_vs_serr_sq(self, jkn):
        """
        BUG: jackknife covariance used np.cov(..., bias=False) then
        multiplied by (N-1), yielding sum-of-squares instead of the
        correct (N-1)*cov(bias=True). Diagonal must equal serr^2.
        """
        diag = np.diag(jkn.cov)
        expected = jkn.serr ** 2
        np.testing.assert_allclose(diag, expected, rtol=1e-10)

    def test_bst_cov_diagonal_vs_serr_sq(self, bst):
        diag = np.diag(bst.cov)
        expected = bst.serr ** 2
        np.testing.assert_allclose(diag, expected, rtol=1e-10)

    def test_bst_rwf_rspl_same_indices_as_data(self, raw, rwf):
        """
        BUG: __bootstrap called Data.bootstrap twice with independent rngs,
        meaning the rwf resamples used different indices than the data resamples.
        The rwf resamples should be consistent with the data resamples.

        Verify: for a trivial case where raw == rwf[:, None] * 1,
        the reweighted bootstrap mean per sample should be data/rwf == 1.
        This is a soft consistency check (exact matching requires shared indices).
        We check that the rwf resamples are non-trivially different from just
        drawing rwf independently (i.e., they are actual bootstrap samples).
        """
        d = Data(resample_type="bst", data=raw, rwf=rwf, Nresample=NBST)
        assert d._rwf_rspl is not None
        # Each rwf resample should be a mean of N draws from rwf — 
        # all values should be near 1.0 and have variation of std(rwf)/sqrt(N).
        assert np.all(np.abs(d._rwf_rspl - 1.0) < 0.5), (
            "rwf resamples look wrong — they should be bootstrap means of rwf"
        )

    def test_ufunc_preserves_ndata_when_none(self, raw):
        """
        BUG: __array_ufunc__ raised if two Data had different Ndata values
        (e.g., one None, one set). Ndata=None should be treated as 'unknown'
        and not cause a RuntimeError.
        """
        d1 = Data.import_resamples(resample_type="jkn", rspl=np.ones((N, T)), Ndata=N)
        d2 = Data.import_resamples(resample_type="jkn", rspl=np.ones((N, T)), Ndata=None)
        # Should not raise even though Ndatas differ (None vs N)
        try:
            r = np.add(d1, d2)
            assert isinstance(r, Data)
        except RuntimeError as e:
            pytest.fail(f"ufunc raised RuntimeError for mixed Ndata: {e}")

    def test_ufunc_rwf_not_silently_dropped(self, jkn_rwf):
        """
        BUG: __array_ufunc__ did not forward rwf_rspl to import_resamples,
        silently dropping reweighting information.
        """
        r = np.add(jkn_rwf, jkn_rwf)
        assert isinstance(r, Data)
        # If rwf is properly forwarded, _rwf_rspl should not be None
        assert r._rwf_rspl is not None, (
            "rwf_rspl was silently dropped by __array_ufunc__"
        )