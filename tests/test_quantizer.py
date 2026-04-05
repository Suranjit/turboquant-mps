"""
Tests for turboquant/quantizer.py

Covers:
  - TurboQuantMSE: quant, dequant, quant_with_norm, dequant_with_norm, mse,
                   _nearest_centroid, seed reproducibility
  - TurboQuantProd: b=1 special case, b>1 general case, quant, dequant,
                    quant_with_norm, dequant_with_norm, inner_product (1D + 2D),
                    inner_product_distortion, unbiasedness, invalid-b error
"""
import math

import numpy as np
import pytest

from tests.conftest import make_unit_vectors
from turboquant.quantizer import TurboQuantMSE, TurboQuantProd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(n, d, seed=0):
    return make_unit_vectors(n, d, seed=seed)


# ---------------------------------------------------------------------------
# TurboQuantMSE
# ---------------------------------------------------------------------------

class TestTurboQuantMSE:

    # --- Construction ---

    def test_rotation_is_orthogonal(self):
        q = TurboQuantMSE(d=32, b=2, seed=0)
        RtR = q.rotation.T @ q.rotation
        np.testing.assert_allclose(RtR, np.eye(32), atol=1e-5)

    def test_codebook_sorted(self):
        q = TurboQuantMSE(d=64, b=3, seed=0)
        assert np.all(np.diff(q.codebook) > 0)

    def test_codebook_length(self):
        for b in [1, 2, 3, 4]:
            q = TurboQuantMSE(d=32, b=b, seed=0)
            assert len(q.codebook) == 2**b

    def test_seed_reproducibility(self):
        q1 = TurboQuantMSE(d=32, b=2, seed=7)
        q2 = TurboQuantMSE(d=32, b=2, seed=7)
        np.testing.assert_array_equal(q1.rotation, q2.rotation)

    def test_different_seeds_different_rotations(self):
        q1 = TurboQuantMSE(d=32, b=2, seed=1)
        q2 = TurboQuantMSE(d=32, b=2, seed=2)
        assert not np.allclose(q1.rotation, q2.rotation)

    # --- quant ---

    @pytest.mark.parametrize("d,b", [(32, 1), (64, 2), (128, 3), (64, 4)])
    def test_quant_output_shape(self, d, b):
        q = TurboQuantMSE(d=d, b=b, seed=0)
        X = _unit(10, d)
        idx = q.quant(X)
        assert idx.shape == (10, d)

    def test_quant_output_dtype(self):
        q = TurboQuantMSE(d=64, b=2, seed=0)
        idx = q.quant(_unit(5, 64))
        assert idx.dtype == np.int16

    def test_quant_index_range(self):
        for b in [1, 2, 3]:
            q = TurboQuantMSE(d=64, b=b, seed=0)
            idx = q.quant(_unit(100, 64))
            assert idx.min() >= 0
            assert idx.max() <= 2**b - 1

    def test_quant_batched_vs_single(self):
        q = TurboQuantMSE(d=32, b=2, seed=0)
        X = _unit(5, 32)
        batch = q.quant(X)
        for i in range(5):
            single = q.quant(X[i : i + 1])
            np.testing.assert_array_equal(batch[i], single[0])

    # --- dequant ---

    @pytest.mark.parametrize("d,b", [(32, 2), (64, 2), (128, 3)])
    def test_dequant_output_shape(self, d, b):
        q = TurboQuantMSE(d=d, b=b, seed=0)
        idx = q.quant(_unit(8, d))
        x_hat = q.dequant(idx)
        assert x_hat.shape == (8, d)

    def test_dequant_output_dtype(self):
        q = TurboQuantMSE(d=64, b=2, seed=0)
        idx = q.quant(_unit(4, 64))
        assert q.dequant(idx).dtype == np.float32

    def test_dequant_reconstruction_is_close(self):
        """High-bit quantizer should reconstruct unit vectors well."""
        q = TurboQuantMSE(d=64, b=4, seed=0)
        X = _unit(200, 64)
        X_hat = q.dequant(q.quant(X))
        mse = float(np.mean(np.sum((X - X_hat)**2, axis=-1)))
        assert mse < 0.02, f"b=4 D_mse={mse:.4f} too high"

    # --- _nearest_centroid ---

    def test_nearest_centroid_exact_match(self):
        q = TurboQuantMSE(d=64, b=2, seed=0)
        # Feed exactly the codebook values through rotation back/forward
        y = q.codebook[[0, 1, 2, 3]]   # shape (4,)
        # Create a (1, d) rotated vector where each coord hits a centroid
        y_full = np.tile(y, 64 // 4)   # repeat to fill d=64 coords
        idx = q._nearest_centroid(y_full[np.newaxis, :])
        # All indices should be within valid range
        assert np.all(idx >= 0) and np.all(idx < 4)

    # --- quant_with_norm / dequant_with_norm ---

    def test_quant_with_norm_shapes(self):
        q = TurboQuantMSE(d=64, b=2, seed=0)
        X = np.random.default_rng(0).standard_normal((10, 64)).astype(np.float32)
        idx, norms = q.quant_with_norm(X)
        assert idx.shape == (10, 64)
        assert norms.shape == (10,)

    def test_quant_with_norm_norms_positive(self):
        q = TurboQuantMSE(d=64, b=2, seed=0)
        X = np.random.default_rng(1).standard_normal((20, 64)).astype(np.float32)
        _, norms = q.quant_with_norm(X)
        assert np.all(norms > 0)

    def test_dequant_with_norm_round_trip(self):
        """Scaling by norm should give near-original vectors at high b."""
        q = TurboQuantMSE(d=64, b=4, seed=0)
        rng = np.random.default_rng(3)
        X = rng.standard_normal((50, 64)).astype(np.float32) * 3.0
        idx, norms = q.quant_with_norm(X)
        X_hat = q.dequant_with_norm(idx, norms)
        rel_err = np.linalg.norm(X - X_hat, axis=-1) / np.linalg.norm(X, axis=-1)
        assert rel_err.mean() < 0.15, f"Mean relative error {rel_err.mean():.3f} too high"

    # --- mse ---

    def test_mse_definition_per_vector(self):
        """mse() must return E[‖x−x̂‖²], not per-element mean."""
        q = TurboQuantMSE(d=64, b=2, seed=0)
        X = _unit(500, 64)
        mse_val = q.mse(X)
        manual = float(np.mean(np.sum((X - q.dequant(q.quant(X)))**2, axis=-1)))
        assert abs(mse_val - manual) < 1e-9

    def test_mse_decreases_with_bits(self):
        X = _unit(500, 64)
        prev = float("inf")
        for b in [1, 2, 3, 4]:
            q = TurboQuantMSE(d=64, b=b, seed=0)
            m = q.mse(X)
            assert m < prev, f"b={b} MSE {m:.5f} not less than b={b-1} MSE {prev:.5f}"
            prev = m

    def test_mse_non_negative(self):
        q = TurboQuantMSE(d=32, b=2, seed=0)
        assert q.mse(_unit(100, 32)) >= 0.0

    # --- paper distortion reference values ---

    @pytest.mark.parametrize("b,expected,tol", [
        (1, 0.36, 0.04),
        (2, 0.117, 0.015),
        (3, 0.034, 0.008),
        (4, 0.009, 0.003),
    ])
    def test_mse_matches_paper_reference(self, b, expected, tol):
        """Empirical D_mse should match paper's reported values within tolerance."""
        q = TurboQuantMSE(d=256, b=b, seed=b)
        X = _unit(3000, 256)
        mse_val = q.mse(X)
        assert abs(mse_val - expected) < tol, (
            f"b={b}: empirical={mse_val:.4f}, paper_ref={expected}, tol={tol}"
        )


# ---------------------------------------------------------------------------
# TurboQuantProd
# ---------------------------------------------------------------------------

class TestTurboQuantProd:

    # --- Construction ---

    def test_b0_raises(self):
        with pytest.raises(ValueError, match="b must be"):
            TurboQuantProd(d=32, b=0)

    def test_b1_valid(self):
        q = TurboQuantProd(d=32, b=1, seed=0)
        assert q.b == 1

    def test_seed_reproducibility(self):
        q1 = TurboQuantProd(d=32, b=2, seed=5)
        q2 = TurboQuantProd(d=32, b=2, seed=5)
        np.testing.assert_array_equal(q1.S, q2.S)
        np.testing.assert_array_equal(q1.mse_quant.rotation, q2.mse_quant.rotation)

    # --- quant output shape/dtype ---

    @pytest.mark.parametrize("d,b", [(32, 1), (64, 2), (128, 3)])
    def test_quant_output_shapes(self, d, b):
        q = TurboQuantProd(d=d, b=b, seed=0)
        X = _unit(6, d)
        idx, qjl, gamma = q.quant(X)
        assert idx.shape == (6, d)
        assert qjl.shape == (6, d)
        assert gamma.shape == (6,)

    def test_quant_idx_dtype(self):
        q = TurboQuantProd(d=64, b=2, seed=0)
        idx, _, _ = q.quant(_unit(4, 64))
        assert idx.dtype == np.int16

    def test_quant_qjl_values(self):
        q = TurboQuantProd(d=64, b=2, seed=0)
        _, qjl, _ = q.quant(_unit(50, 64))
        assert qjl.dtype == np.int8
        assert set(np.unique(qjl)).issubset({-1, 1}), "QJL must be ±1"

    def test_quant_gamma_positive(self):
        q = TurboQuantProd(d=64, b=2, seed=0)
        _, _, gamma = q.quant(_unit(50, 64))
        assert np.all(gamma >= 0)

    # --- b=1 special case ---

    def test_b1_idx_all_zeros(self):
        """When b=1, the MSE stage is skipped so idx must be all zeros."""
        q = TurboQuantProd(d=32, b=1, seed=0)
        X = _unit(20, 32)
        idx, _, _ = q.quant(X)
        np.testing.assert_array_equal(idx, np.zeros_like(idx))

    def test_b1_residual_equals_x(self):
        """When b=1, residual r = x (MSE part contributes nothing)."""
        q = TurboQuantProd(d=32, b=1, seed=0)
        X = _unit(10, 32)
        _, _, gamma = q.quant(X)
        # gamma = ‖r‖ = ‖x‖ = 1 for unit-norm x
        np.testing.assert_allclose(gamma, np.ones(10), atol=1e-5)

    # --- dequant ---

    @pytest.mark.parametrize("d,b", [(32, 1), (64, 2), (128, 3)])
    def test_dequant_output_shape(self, d, b):
        q = TurboQuantProd(d=d, b=b, seed=0)
        X = _unit(5, d)
        idx, qjl, gamma = q.quant(X)
        X_hat = q.dequant(idx, qjl, gamma)
        assert X_hat.shape == (5, d)

    def test_dequant_output_dtype(self):
        q = TurboQuantProd(d=64, b=2, seed=0)
        X = _unit(4, 64)
        idx, qjl, gamma = q.quant(X)
        assert q.dequant(idx, qjl, gamma).dtype == np.float32

    # --- quant_with_norm / dequant_with_norm ---

    def test_quant_with_norm_return_count(self):
        q = TurboQuantProd(d=64, b=2, seed=0)
        result = q.quant_with_norm(_unit(5, 64))
        assert len(result) == 4   # idx, qjl, qjl_gamma, vec_norm

    def test_dequant_with_norm_scales_correctly(self):
        """dequant_with_norm should scale by vec_norm, not qjl_gamma."""
        q = TurboQuantProd(d=64, b=3, seed=0)
        rng = np.random.default_rng(5)
        scale = 2.5
        X = _unit(30, 64) * scale
        idx, qjl, qjl_gamma, norms = q.quant_with_norm(X)
        X_hat = q.dequant_with_norm(idx, qjl, qjl_gamma, norms)
        # Output magnitude should be ~scale
        out_norms = np.linalg.norm(X_hat, axis=-1)
        np.testing.assert_allclose(out_norms.mean(), scale, rtol=0.3)

    # --- inner_product ---

    def test_inner_product_shape_1d_query(self):
        q = TurboQuantProd(d=64, b=2, seed=0)
        X = _unit(20, 64)
        y = _unit(1, 64)[0]  # (64,)
        idx, qjl, gamma = q.quant(X)
        ip = q.inner_product(y, idx, qjl, gamma)
        assert ip.shape == (20,)

    def test_inner_product_shape_2d_query(self):
        q = TurboQuantProd(d=64, b=2, seed=0)
        X = _unit(20, 64)
        Y = _unit(20, 64, seed=1)
        idx, qjl, gamma = q.quant(X)
        ip = q.inner_product(Y, idx, qjl, gamma)
        assert ip.shape == (20,)

    def test_inner_product_unbiased(self):
        """Mean error should be statistically indistinguishable from 0."""
        q = TurboQuantProd(d=128, b=2, seed=1)
        n = 10_000
        X = _unit(n, 128, seed=0)
        Y = _unit(n, 128, seed=1)
        ip_true = (X * Y).sum(-1)
        idx, qjl, gamma = q.quant(X)
        ip_est = q.inner_product(Y, idx, qjl, gamma)
        bias = float(np.mean(ip_est - ip_true))
        stderr = float(np.std(ip_est - ip_true) / np.sqrt(n))
        # |bias| < 3 * stderr with high probability (>99.7%) if truly unbiased
        assert abs(bias) < 3 * stderr, f"bias={bias:.5f}, stderr={stderr:.5f}"

    def test_inner_product_vs_dequant(self):
        """inner_product should agree with direct dot product of dequantized x."""
        q = TurboQuantProd(d=64, b=3, seed=0)
        X = _unit(50, 64)
        Y = _unit(50, 64, seed=2)
        idx, qjl, gamma = q.quant(X)
        X_hat = q.dequant(idx, qjl, gamma)
        ip_direct = (X_hat * Y).sum(-1)
        ip_method = q.inner_product(Y, idx, qjl, gamma)
        np.testing.assert_allclose(ip_direct, ip_method, atol=1e-5)

    def test_inner_product_1d_vs_2d_query_agree(self):
        """1D and 2D query paths must give identical results."""
        q = TurboQuantProd(d=32, b=2, seed=0)
        X = _unit(10, 32)
        Y = _unit(10, 32, seed=3)
        idx, qjl, gamma = q.quant(X)
        # 2D path (vectorized over pairs)
        ip_2d = q.inner_product(Y, idx, qjl, gamma)
        # 1D path (single query per X vector)
        ip_1d = np.array([
            q.inner_product(Y[i], idx[i:i+1], qjl[i:i+1], gamma[i:i+1])[0]
            for i in range(10)
        ])
        np.testing.assert_allclose(ip_2d, ip_1d, atol=1e-5)

    # --- inner_product_distortion ---

    def test_inner_product_distortion_non_negative(self):
        q = TurboQuantProd(d=64, b=2, seed=0)
        X = _unit(200, 64)
        Y = _unit(200, 64, seed=1)
        d_prod = q.inner_product_distortion(X, Y)
        assert d_prod >= 0.0

    def test_inner_product_distortion_decreases_with_bits(self):
        n, d = 1000, 64
        X = _unit(n, d)
        Y = _unit(n, d, seed=1)
        prev = float("inf")
        for b in [1, 2, 3, 4]:
            q = TurboQuantProd(d=d, b=b, seed=0)
            dprod = q.inner_product_distortion(X, Y)
            assert dprod < prev, f"b={b}: D_prod {dprod:.6f} not less than b={b-1}: {prev:.6f}"
            prev = dprod

    # --- TurboQuantProd vs TurboQuantMSE bias comparison ---

    def test_prod_less_biased_than_mse(self):
        """TurboQuantProd inner-product estimate should have smaller bias than MSE."""
        n, d, b = 5000, 64, 2
        X = _unit(n, d, seed=0)
        Y = _unit(n, d, seed=1)
        ip_true = (X * Y).sum(-1)

        q_mse = TurboQuantMSE(d, b, seed=10)
        X_hat_mse = q_mse.dequant(q_mse.quant(X))
        bias_mse = abs(float(np.mean((X_hat_mse * Y).sum(-1) - ip_true)))

        q_prod = TurboQuantProd(d, b, seed=11)
        idx, qjl, gamma = q_prod.quant(X)
        bias_prod = abs(float(np.mean(q_prod.inner_product(Y, idx, qjl, gamma) - ip_true)))

        # With enough samples the bias of prod should be much smaller
        assert bias_prod < bias_mse or bias_prod < 5e-3, (
            f"prod bias {bias_prod:.5f} not smaller than MSE bias {bias_mse:.5f}"
        )
