import numpy as np
import pytest
from correlatoranalyser import Data

# =============================================================================
# Fixtures
# =============================================================================

RNG = np.random.default_rng(42)
N, T, NBST = 1000, 8, 200


@pytest.fixture
def raw():
    return RNG.normal(0, 1, size=(N, T))


@pytest.fixture
def raw_1d():
    return RNG.normal(0, 1, size=(N,))


@pytest.fixture
def rwf():
    return RNG.normal(1, 0.1, size=(N,))


@pytest.fixture
def jkn(raw):
    return Data(resample_type="jkn", data=raw)


@pytest.fixture
def bst(raw):
    return Data(resample_type="bst", data=raw, Nresample=NBST)


@pytest.fixture
def jkn_rwf(raw, rwf):
    return Data(resample_type="jkn", data=raw, rwf=rwf)


@pytest.fixture
def bst_rwf(raw, rwf):
    return Data(resample_type="bst", data=raw, rwf=rwf, Nresample=NBST)


# =============================================================================
# 1. Construction & shapes
# =============================================================================

class TestConstruction:

    def test_jkn_resample_shape(self, raw):
        d = Data(resample_type="jkn", data=raw)
        assert d._rspl.shape == (N, T)

    def test_bst_resample_shape(self, raw):
        d = Data(resample_type="bst", data=raw, Nresample=NBST)
        assert d._rspl.shape == (NBST, T)

    def test_jkn_nresample(self, jkn):
        assert jkn.Nresample == N

    def test_bst_nresample(self, bst):
        assert bst.Nresample == NBST

    def test_ndata_set(self, jkn, bst, raw):
        assert jkn.Ndata == N
        assert bst.Ndata == N

    def test_resample_type_stored(self, jkn, bst):
        assert jkn.resample_type == "jkn"
        assert bst.resample_type == "bst"

    def test_tag_stored(self, raw):
        d = Data(resample_type="jkn", data=raw, tag="test_tag")
        assert d.tag == "test_tag"

    def test_bst_requires_nresample(self, raw):
        with pytest.raises(ValueError):
            Data(resample_type="bst", data=raw)

    def test_data_must_be_ndarray(self):
        with pytest.raises(ValueError):
            Data(resample_type="jkn", data=[1, 2, 3])

    def test_custom_mean_respected(self, raw):
        custom_mean = np.ones(T) * 99.0
        d = Data(resample_type="jkn", data=raw, mean=custom_mean)
        np.testing.assert_array_equal(d._mean, custom_mean)

    def test_1d_data_jkn(self, raw_1d):
        d = Data(resample_type="jkn", data=raw_1d)
        assert d._rspl.shape == (N,)

    def test_1d_data_bst(self, raw_1d):
        d = Data(resample_type="bst", data=raw_1d, Nresample=NBST)
        assert d._rspl.shape == (NBST,)


# =============================================================================
# 2. No mutation of input arrays
# =============================================================================

class TestNoMutation:

    def test_jkn_does_not_mutate_data(self, raw):
        raw_copy = raw.copy()
        Data(resample_type="jkn", data=raw)
        np.testing.assert_array_equal(raw, raw_copy)

    def test_bst_does_not_mutate_data(self, raw):
        raw_copy = raw.copy()
        Data(resample_type="bst", data=raw, Nresample=NBST)
        np.testing.assert_array_equal(raw, raw_copy)

    def test_jkn_rwf_does_not_mutate_data(self, raw, rwf):
        raw_copy = raw.copy()
        Data(resample_type="jkn", data=raw, rwf=rwf)
        np.testing.assert_array_equal(raw, raw_copy)

    def test_bst_rwf_does_not_mutate_data(self, raw, rwf):
        raw_copy = raw.copy()
        Data(resample_type="bst", data=raw, rwf=rwf, Nresample=NBST)
        np.testing.assert_array_equal(raw, raw_copy)

    def test_jkn_does_not_mutate_rwf(self, raw, rwf):
        rwf_copy = rwf.copy()
        Data(resample_type="jkn", data=raw, rwf=rwf)
        np.testing.assert_array_equal(rwf, rwf_copy)

    def test_bst_does_not_mutate_rwf(self, raw, rwf):
        rwf_copy = rwf.copy()
        Data(resample_type="bst", data=raw, rwf=rwf, Nresample=NBST)
        np.testing.assert_array_equal(rwf, rwf_copy)


# =============================================================================
# 3. Jackknife correctness
# =============================================================================

class TestJackknife:

    def test_leave_one_out_mean(self, raw_1d):
        """Each jackknife sample should equal (sum - x_k) / (N-1)."""
        d = Data(resample_type="jkn", data=raw_1d)
        expected = (raw_1d.sum() - raw_1d) / (N - 1)
        np.testing.assert_allclose(d._rspl, expected)

    def test_jackknife_mean_close_to_true_mean(self, raw):
        """Mean of jackknife resamples should equal mean of raw data."""
        d = Data(resample_type="jkn", data=raw)
        np.testing.assert_allclose(d.mean, raw.mean(axis=0), rtol=1e-10)

    def test_static_jackknife_matches_init(self, raw):
        d = Data(resample_type="jkn", data=raw)
        static = Data.jackknife(data=raw.copy())
        np.testing.assert_allclose(d._rspl, static)

    def test_jackknife_rwf_shape(self, jkn_rwf):
        assert jkn_rwf._rspl.shape == (N, T)

    def test_jackknife_rwf_central_value(self, raw, rwf):
        """Reweighted JKN mean should approximate <w*x>/<w>."""
        d = Data(resample_type="jkn", data=raw, rwf=rwf)
        expected_mean = (rwf[:, None] * raw).sum(axis=0) / rwf.sum()
        np.testing.assert_allclose(d.mean, expected_mean, rtol=1e-5)

    def test_jackknife_rwf_does_not_equal_unweighted(self, raw, rwf):
        """Reweighted and unweighted results should differ when rwf != 1."""
        d_plain = Data(resample_type="jkn", data=raw)
        d_rwf   = Data(resample_type="jkn", data=raw, rwf=rwf)
        assert not np.allclose(d_plain.mean, d_rwf.mean)


# =============================================================================
# 4. Bootstrap correctness
# =============================================================================

class TestBootstrap:

    def test_bootstrap_mean_close_to_true(self, raw):
        """Bootstrap mean should be close to the sample mean."""
        d = Data(resample_type="bst", data=raw, Nresample=500)
        np.testing.assert_allclose(d.mean, raw.mean(axis=0), atol=0.1)

    def test_static_bootstrap_shape(self, raw):
        bst = Data.bootstrap(data=raw.copy(), Nresample=NBST)
        assert bst.shape == (NBST, T)

    def test_bootstrap_rwf_shape(self, bst_rwf):
        assert bst_rwf._rspl.shape == (NBST, T)

    def test_bootstrap_reproducible(self, raw):
        """Two bootstraps with the same fixed rng seed should be identical."""
        b1 = Data.bootstrap(data=raw.copy(), Nresample=NBST)
        b2 = Data.bootstrap(data=raw.copy(), Nresample=NBST)
        np.testing.assert_array_equal(b1, b2)

    def test_bootstrap_rwf_central_value(self, raw, rwf):
        """Bootstrap reweighted mean should approximate <w*x>/<w>."""
        d = Data(resample_type="bst", data=raw, rwf=rwf, Nresample=500)
        expected = (rwf[:, None] * raw).sum(axis=0) / rwf.sum()
        np.testing.assert_allclose(d.mean, expected, atol=0.15)


# =============================================================================
# 5. Standard error
# =============================================================================

class TestSerr:

    def test_jkn_serr_shape(self, jkn):
        assert jkn.serr.shape == (T,)

    def test_bst_serr_shape(self, bst):
        assert bst.serr.shape == (T,)

    def test_jkn_serr_positive(self, jkn):
        assert np.all(jkn.serr >= 0)

    def test_bst_serr_positive(self, bst):
        assert np.all(bst.serr >= 0)

    def test_jkn_serr_formula(self, raw_1d):
        """SE_jkn = sqrt(N-1) * std(resamples, ddof=0)."""
        d = Data(resample_type="jkn", data=raw_1d)
        expected = np.sqrt(N - 1) * np.std(d._rspl, axis=0)
        np.testing.assert_allclose(d.serr, expected)

    def test_bst_serr_formula(self, raw_1d):
        """SE_bst = std(resamples, ddof=1)."""
        d = Data(resample_type="bst", data=raw_1d, Nresample=NBST)
        expected = np.std(d._rspl, axis=0, ddof=1)
        np.testing.assert_allclose(d.serr, expected)

    def test_jkn_bst_serr_agree_large_N(self, raw):
        """For smooth estimator and large N, JKN and BST SE should agree within ~20%."""
        jkn = Data(resample_type="jkn", data=raw)
        bst = Data(resample_type="bst", data=raw, Nresample=1000)
        ratio = jkn.serr / bst.serr
        np.testing.assert_allclose(ratio, np.ones(T), atol=0.2)


# =============================================================================
# 6. Covariance
# =============================================================================

class TestCovariance:

    def test_jkn_cov_shape(self, jkn):
        assert jkn.cov.shape == (T, T)

    def test_bst_cov_shape(self, bst):
        assert bst.cov.shape == (T, T)

    def test_cov_diagonal_equals_variance_squared(self, jkn, bst):
        """Diagonal of covariance matrix should equal serr**2."""
        np.testing.assert_allclose(np.diag(jkn.cov), (jkn.Nresample-1)*np.var(jkn.rspl, axis=0, ddof=1), rtol=1e-10)
        np.testing.assert_allclose(np.diag(bst.cov), np.var(bst.rspl, axis=0), rtol=1e-10)

    def test_cov_symmetric(self, jkn, bst):
        np.testing.assert_allclose(jkn.cov, jkn.cov.T)
        np.testing.assert_allclose(bst.cov, bst.cov.T)

    def test_cov_positive_semidefinite(self, jkn):
        eigvals = np.linalg.eigvalsh(jkn.cov)
        assert np.all(eigvals >= -1e-10)


# =============================================================================
# 7. Arithmetic operators
# =============================================================================

class TestArithmetic:

    def test_add_data_data(self, jkn):
        result = jkn + jkn
        np.testing.assert_allclose(result._rspl, 2 * jkn._rspl)
        np.testing.assert_allclose(result.mean,  2 * jkn.mean)

    def test_add_data_scalar(self, jkn):
        result = jkn + 5.0
        np.testing.assert_allclose(result._rspl, jkn._rspl + 5.0)

    def test_radd_scalar(self, jkn):
        result = 5.0 + jkn
        np.testing.assert_allclose(result._rspl, jkn._rspl + 5.0)

    def test_sub_data_data(self, jkn):
        result = jkn - jkn
        np.testing.assert_allclose(result._rspl, np.zeros_like(jkn._rspl), atol=1e-14)

    def test_sub_scalar(self, jkn):
        result = jkn - 1.0
        np.testing.assert_allclose(result._rspl, jkn._rspl - 1.0)

    def test_mul_data_data(self, jkn):
        result = jkn * jkn
        np.testing.assert_allclose(result._rspl, jkn._rspl ** 2)

    def test_mul_scalar(self, jkn):
        result = jkn * 3.0
        np.testing.assert_allclose(result._rspl, jkn._rspl * 3.0)

    def test_rmul_scalar(self, jkn):
        result = 3.0 * jkn
        np.testing.assert_allclose(result._rspl, jkn._rspl * 3.0)

    def test_truediv_scalar(self, jkn):
        result = jkn / 2.0
        np.testing.assert_allclose(result._rspl, jkn._rspl / 2.0)

    def test_rtruediv_scalar(self, jkn):
        result = 1.0 / jkn
        np.testing.assert_allclose(result._rspl, 1.0 / jkn._rspl)

    def test_pow_scalar(self, jkn):
        result = jkn ** 2
        np.testing.assert_allclose(result._rspl, jkn._rspl ** 2)

    def test_neg(self, jkn):
        result = -jkn
        np.testing.assert_allclose(result._rspl, -jkn._rspl)
        np.testing.assert_allclose(result.mean, -jkn.mean)

    def test_iadd(self, raw):
        d = Data(resample_type="jkn", data=raw)
        rspl_before = d._rspl.copy()
        d += 1.0
        np.testing.assert_allclose(d._rspl, rspl_before + 1.0)

    def test_isub(self, raw):
        d = Data(resample_type="jkn", data=raw)
        rspl_before = d._rspl.copy()
        d -= 1.0
        np.testing.assert_allclose(d._rspl, rspl_before - 1.0)

    def test_imul(self, raw):
        d = Data(resample_type="jkn", data=raw)
        rspl_before = d._rspl.copy()
        d *= 2.0
        np.testing.assert_allclose(d._rspl, rspl_before * 2.0)

    def test_arithmetic_preserves_resample_type(self, jkn):
        result = jkn + jkn
        assert result.resample_type == "jkn"

    def test_arithmetic_preserves_nresample(self, jkn):
        result = jkn * 2.0
        assert result.Nresample == jkn.Nresample

    def test_mismatched_nresample_raises(self, raw):
        d1 = Data(resample_type="bst", data=raw, Nresample=50)
        d2 = Data(resample_type="bst", data=raw, Nresample=100)
        with pytest.raises(ValueError):
            _ = d1 + d2


# =============================================================================
# 8. Factories
# =============================================================================

class TestFactories:

    def test_import_resamples_roundtrip(self, jkn):
        d = Data.import_resamples(
            resample_type=jkn.resample_type,
            rspl=jkn._rspl.copy(),
            Ndata=jkn.Ndata,
        )
        np.testing.assert_array_equal(d._rspl, jkn._rspl)
        assert d.Nresample == jkn.Nresample

    def test_import_resamples_nresample_mismatch_raises(self):
        rspl = np.ones((10, 5))
        with pytest.raises(RuntimeError):
            Data.import_resamples(resample_type="bst", rspl=rspl, Nresample=99)

    def test_zeros_shape(self):
        d = Data.zeros(resample_type="bst", shape=(T,), Nresample=NBST)
        assert d._rspl.shape == (NBST, T)
        np.testing.assert_array_equal(d._rspl, 0)

    def test_zeros_jkn_deduces_nresample(self):
        d = Data.zeros(resample_type="jkn", shape=(T,), Ndata=N)
        assert d.Nresample == N

    def test_ones_shape(self):
        d = Data.ones(resample_type="bst", shape=(T,), Nresample=NBST)
        np.testing.assert_array_equal(d._rspl, 1)

    def test_zeros_like(self, jkn):
        d = Data.zeros_like(jkn)
        assert d._rspl.shape == jkn._rspl.shape
        np.testing.assert_array_equal(d._rspl, 0)
        assert d.resample_type == jkn.resample_type
        assert d.Nresample == jkn.Nresample

    def test_copy_is_independent(self, jkn):
        d = jkn.copy()
        d._rspl[0, 0] = 9999.0
        assert jkn._rspl[0, 0] != 9999.0


# =============================================================================
# 9. Blocking
# =============================================================================

class TestBlocking:

    def test_blocking_output_shape(self, raw):
        blocksize = 10
        blocked, _ = Data.blocking(data=raw.copy(), blocksize=blocksize)
        assert blocked.shape == (N // blocksize, T)

    def test_blocking_does_not_mutate_data(self, raw):
        raw_copy = raw.copy()
        Data.blocking(data=raw.copy(), blocksize=10)
        np.testing.assert_array_equal(raw, raw_copy)

    def test_blocking_rwf_does_not_mutate(self, raw, rwf):
        raw_copy = raw.copy()
        rwf_copy = rwf.copy()
        Data.blocking(data=raw.copy(), blocksize=10, rwf=rwf.copy())
        np.testing.assert_array_equal(raw, raw_copy)
        np.testing.assert_array_equal(rwf, rwf_copy)

    def test_blocking_rwf_output_shapes(self, raw, rwf):
        blocksize = 10
        blocked_data, blocked_rwf = Data.blocking(data=raw.copy(), blocksize=blocksize, rwf=rwf.copy())
        assert blocked_data.shape == (N // blocksize, T)
        assert blocked_rwf.shape == (N // blocksize,)

    def test_jkn_with_blocksize(self, raw):
        blocksize = 5
        d = Data(resample_type="jkn", data=raw, blocksize=blocksize)
        assert d.Nresample == N // blocksize

    def test_bst_with_blocksize(self, raw):
        blocksize = 5
        d = Data(resample_type="bst", data=raw, Nresample=NBST, blocksize=blocksize)
        assert d._rspl.shape == (NBST, T)


# =============================================================================
# 10. Indexing
# =============================================================================

class TestIndexing:

    def test_getitem_scalar_index(self, jkn):
        d = jkn[0]
        assert d._rspl.shape == (N,)
        np.testing.assert_allclose(d._rspl, jkn._rspl[:, 0])

    def test_getitem_slice(self, jkn):
        d = jkn[0:3]
        assert d._rspl.shape == (N, 3)

    def test_getitem_preserves_nresample(self, jkn):
        d = jkn[0]
        assert d.Nresample == jkn.Nresample

    def test_setitem_from_data(self, jkn):
        target = Data.zeros_like(jkn)
        target[0] = jkn[0]
        np.testing.assert_allclose(target._rspl[:, 0], jkn._rspl[:, 0])

    def test_setitem_mismatched_nresample_raises(self, raw):
        d1 = Data(resample_type="jkn", data=raw)
        d2_rspl = np.ones((50, T))
        d2 = Data.import_resamples(resample_type="jkn", rspl=d2_rspl)
        with pytest.raises(ValueError):
            d1[0] = d2


# =============================================================================
# 11. Shape & properties
# =============================================================================

class TestProperties:

    def test_shape_property(self, jkn):
        assert jkn.shape == (T,)

    def test_ndim_property(self, jkn):
        assert jkn.ndim == 1

    def test_rspl_property(self, jkn):
        np.testing.assert_array_equal(jkn.rspl, jkn._rspl)

    def test_mean_recomputed_when_not_locked(self, raw):
        d = Data(resample_type="jkn", data=raw)
        d._rspl += 100.0
        new_mean = d.mean
        np.testing.assert_allclose(new_mean, raw.mean(axis=0) + 100.0, rtol=1e-10)

    def test_mean_locked(self, raw):
        fixed_mean = np.ones(T) * 42.0
        d = Data(resample_type="jkn", data=raw, mean=fixed_mean, locked_mean=True)
        d._rspl += 100.0          # would change the computed mean
        np.testing.assert_array_equal(d.mean, fixed_mean)

    def test_reshape(self, raw):
        d = Data(resample_type="jkn", data=raw)
        d.reshape((2, T // 2))
        assert d.shape == (2, T // 2)
        assert d._rspl.shape == (N, 2, T // 2)


# =============================================================================
# 12. Numpy interoperability
# =============================================================================

class TestNumpyInterop:

    def test_np_mean_axis1(self, jkn):
        result = np.mean(jkn, axis=1)
        assert isinstance(result, Data)
        assert result._rspl.shape == (N,)

    def test_np_mean_axis0_returns_array(self, jkn):
        result = np.mean(jkn, axis=0)
        assert isinstance(result, np.ndarray)
        assert result.shape == (T,)

    def test_array_ufunc_add(self, jkn):
        result = np.add(jkn, jkn)
        assert isinstance(result, Data)
        np.testing.assert_allclose(result._rspl, 2 * jkn._rspl)

    def test_np_sqrt(self, raw):
        d = Data(resample_type="jkn", data=np.abs(raw))
        result = np.sqrt(d)
        assert isinstance(result, Data)
        np.testing.assert_allclose(result._rspl, np.sqrt(d._rspl))

    def test_comparison_lt(self, jkn):
        result = jkn < 0.0
        assert isinstance(result, np.ndarray)
        assert result.shape == jkn._rspl.shape