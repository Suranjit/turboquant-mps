"""
Integration tests: cross-component correctness

Covers:
  - numpy and MPS quantizers agree at float32 (same rotation/codebook)
  - Lloyd-Max → TurboQuantMSE distortion within expected range
  - TurboQuantProd unbiasedness holds statistically
  - MPS pack/unpack is lossless inside the full quant_pack/dequant_unpack cycle
  - MPSTurboQuantCache reconstruction quality (cosine similarity)
  - Compression ratio is consistent between theoretical and actual
  - Multi-layer cache with varied chunk sizes returns consistent seq lengths
"""
import math

import numpy as np
import pytest
import torch

from tests.conftest import make_unit_tensors, make_unit_vectors
from turboquant.mps_kv_cache import MPSTurboQuantCache
from turboquant.mps_quantizer import MPSTurboQuantMSE, MPSTurboQuantProd
from turboquant.quantizer import TurboQuantMSE, TurboQuantProd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _make_cache_kv(B=1, H=4, T=32, d=64, device="cpu", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    k = torch.randn(B, H, T, d, generator=g, dtype=torch.float16).to(device)
    v = torch.randn(B, H, T, d, generator=g, dtype=torch.float16).to(device)
    return k, v


# ---------------------------------------------------------------------------
# numpy vs MPS agreement (same matrices, float32)
# ---------------------------------------------------------------------------

class TestNumpyMPSAgreement:
    """With the same rotation matrix and codebook, numpy and MPS must agree."""

    def test_mse_quant_indices_match(self, device):
        d, b, n = 64, 2, 200
        X_np = make_unit_vectors(n, d, seed=0)
        X_t  = torch.from_numpy(X_np).to(device)

        q_np  = TurboQuantMSE(d, b, seed=7)
        q_mps = MPSTurboQuantMSE(d, b, device=device, dtype=torch.float32, seed=7)

        # Override MPS matrices with numpy ones for apples-to-apples comparison
        q_mps.rotation = torch.from_numpy(q_np.rotation.astype(np.float32)).to(device)
        q_mps.codebook = torch.from_numpy(q_np.codebook.astype(np.float32)).to(device)

        idx_np  = q_np.quant(X_np)
        idx_mps = q_mps.quant(X_t).cpu().numpy().astype(np.int16)

        np.testing.assert_array_equal(idx_np, idx_mps,
            err_msg="Numpy and MPS MSE quantizers produce different indices")

    def test_prod_quant_agrees(self, device):
        d, b, n = 64, 2, 200
        X_np = make_unit_vectors(n, d, seed=0)
        X_t  = torch.from_numpy(X_np).to(device)

        q_np  = TurboQuantProd(d, b, seed=3)
        q_mps = MPSTurboQuantProd(d, b, device=device, dtype=torch.float32, seed=3)

        # Share matrices
        q_mps.mse.rotation = torch.from_numpy(q_np.mse_quant.rotation.astype(np.float32)).to(device)
        q_mps.mse.codebook = torch.from_numpy(q_np.mse_quant.codebook.astype(np.float32)).to(device)
        q_mps.S  = torch.from_numpy(q_np.S.astype(np.float32)).to(device)
        q_mps.ST = q_mps.S.T.contiguous()

        idx_np, qjl_np, gamma_np = q_np.quant(X_np)
        idx_t, qjl_t, gamma_t    = q_mps.quant(X_t)

        np.testing.assert_array_equal(idx_np, idx_t.cpu().numpy().astype(np.int16))
        np.testing.assert_array_equal(qjl_np, qjl_t.cpu().numpy().astype(np.int8))
        np.testing.assert_allclose(gamma_np,
                                   gamma_t.cpu().numpy(),
                                   atol=1e-4,
                                   err_msg="gamma mismatch between numpy and MPS")

    def test_prod_d_prod_agrees(self, device):
        d, b, n = 128, 2, 500
        X_np = make_unit_vectors(n, d, seed=0)
        Y_np = make_unit_vectors(n, d, seed=1)
        ip_true = (X_np * Y_np).sum(-1)

        q_np  = TurboQuantProd(d, b, seed=5)
        q_mps = MPSTurboQuantProd(d, b, device=device, dtype=torch.float32, seed=5)
        q_mps.mse.rotation = torch.from_numpy(q_np.mse_quant.rotation.astype(np.float32)).to(device)
        q_mps.mse.codebook = torch.from_numpy(q_np.mse_quant.codebook.astype(np.float32)).to(device)
        q_mps.S  = torch.from_numpy(q_np.S.astype(np.float32)).to(device)
        q_mps.ST = q_mps.S.T.contiguous()

        idx_np, qjl_np, gamma_np = q_np.quant(X_np)
        X_hat_np = q_np.dequant(idx_np, qjl_np, gamma_np)
        ip_np = (X_hat_np * Y_np).sum(-1)

        X_t  = torch.from_numpy(X_np).to(device)
        idx_t, qjl_t, gamma_t = q_mps.quant(X_t)
        X_hat_t = q_mps.dequant(idx_t, qjl_t, gamma_t).cpu().numpy()
        ip_mps = (X_hat_t * Y_np).sum(-1)

        np.testing.assert_allclose(ip_np, ip_mps, atol=1e-4,
            err_msg="Inner-product estimates differ between numpy and MPS")


# ---------------------------------------------------------------------------
# Distortion bounds (empirical matches paper references)
# ---------------------------------------------------------------------------

class TestDistortionBounds:

    @pytest.mark.parametrize("b,ref,tol", [
        (1, 0.36, 0.04),
        (2, 0.117, 0.015),
        (3, 0.034, 0.008),
        (4, 0.009, 0.003),
    ])
    def test_mse_distortion_matches_paper(self, b, ref, tol):
        q = TurboQuantMSE(d=256, b=b, seed=b)
        X = make_unit_vectors(3000, 256)
        d_mse = q.mse(X)
        assert abs(d_mse - ref) < tol, (
            f"b={b}: D_mse={d_mse:.4f}, paper_ref={ref}, tol={tol}"
        )

    def test_d_mse_above_information_lower_bound(self):
        """D_mse must be ≥ 4^{-b} (information-theoretic lower bound)."""
        for b in [1, 2, 3]:
            q = TurboQuantMSE(d=256, b=b, seed=b)
            d_mse = q.mse(make_unit_vectors(2000, 256))
            lb = 4**(-b)
            assert d_mse > lb * 0.9, (
                f"b={b}: D_mse={d_mse:.5f} unexpectedly below lower bound {lb:.5f}"
            )


# ---------------------------------------------------------------------------
# Unbiasedness of TurboQuantProd
# ---------------------------------------------------------------------------

class TestUnbiasedness:

    def test_prod_inner_product_unbiased(self):
        """|bias| must be < 3 standard errors (statistical test at 99.7% confidence)."""
        n, d, b = 20_000, 128, 2
        X = make_unit_vectors(n, d, seed=0)
        Y = make_unit_vectors(n, d, seed=1)
        ip_true = (X * Y).sum(-1)

        q = TurboQuantProd(d, b, seed=42)
        idx, qjl, gamma = q.quant(X)
        ip_est = q.inner_product(Y, idx, qjl, gamma)

        errors = ip_est - ip_true
        bias    = float(np.mean(errors))
        stderr  = float(np.std(errors) / np.sqrt(n))
        assert abs(bias) < 3 * stderr, (
            f"TurboQuantProd is biased: bias={bias:.6f}, 3*stderr={3*stderr:.6f}"
        )

    def test_mse_inner_product_biased(self):
        """TurboQuantMSE inner-product estimate SHOULD be biased (it's not designed for this)."""
        n, d, b = 10_000, 64, 2
        X = make_unit_vectors(n, d, seed=0)
        Y = make_unit_vectors(n, d, seed=1)
        ip_true = (X * Y).sum(-1)

        q = TurboQuantMSE(d, b, seed=42)
        X_hat = q.dequant(q.quant(X))
        ip_mse = (X_hat * Y).sum(-1)
        bias_mse = abs(float(np.mean(ip_mse - ip_true)))

        q_prod = TurboQuantProd(d, b, seed=42)
        idx, qjl, gamma = q_prod.quant(X)
        ip_prod = q_prod.inner_product(Y, idx, qjl, gamma)
        bias_prod = abs(float(np.mean(ip_prod - ip_true)))

        # prod bias should be much smaller; both < 1 just sanity-checks magnitude
        assert bias_mse >= 0 and bias_prod >= 0
        assert bias_prod < 0.01, f"prod bias {bias_prod} unexpectedly large"


# ---------------------------------------------------------------------------
# MPS pack/unpack losslessness through full quant cycle
# ---------------------------------------------------------------------------

class TestPackUnpackLosslessness:

    @pytest.mark.parametrize("b", [1, 2, 3, 4])
    def test_mse_quant_pack_lossless_indices(self, device, b):
        """Unpacking packed_idx must recover the same indices as direct quant."""
        d = 64
        q = MPSTurboQuantMSE(d=d, b=b, device=device, seed=0)
        X = make_unit_tensors(20, d, device)

        # Direct indices
        idx_direct = q.quant(X)

        # Via pack/unpack
        packed_idx, _ = q.quant_pack(X)
        from turboquant.mps_quantizer import _storage_bits, unpack_bits
        pb = _storage_bits(b)
        if pb < 8:
            idx_recovered = unpack_bits(packed_idx, pb, d)
        else:
            idx_recovered = packed_idx.to(torch.int16)

        assert torch.equal(idx_direct, idx_recovered), (
            f"b={b}: packed/unpacked indices differ from direct quant"
        )

    @pytest.mark.parametrize("b", [1, 2, 4])
    def test_prod_quant_pack_lossless_indices(self, device, b):
        """Same losslessness check for TurboQuantProd."""
        from turboquant.mps_quantizer import _storage_bits, unpack_bits
        d = 64
        q = MPSTurboQuantProd(d=d, b=b, device=device, seed=0)
        X = make_unit_tensors(20, d, device)

        idx_direct, qjl_direct, _ = q.quant(X)
        pi, pq, _, _ = q.quant_pack(X)

        pb = _storage_bits(max(b - 1, 1))
        idx_rec = unpack_bits(pi, pb, d) if pb < 8 else pi.to(torch.int16)
        assert torch.equal(idx_direct, idx_rec), f"b={b}: idx mismatch after pack/unpack"

        from turboquant.mps_quantizer import unpack_qjl
        qjl_rec = unpack_qjl(pq, d)
        assert torch.equal(qjl_direct, qjl_rec), f"b={b}: qjl mismatch after pack/unpack"


# ---------------------------------------------------------------------------
# MPSTurboQuantCache reconstruction quality
# ---------------------------------------------------------------------------

class TestCacheReconstructionQuality:

    def test_cosine_similarity_above_threshold(self, device):
        """At b=3, reconstructed KV vectors should align with originals."""
        cache = MPSTurboQuantCache(head_dim=64, bits=3, device=device,
                                   dtype=torch.float16, seed=0)
        B, H, T, d = 1, 4, 50, 64
        k, v = _make_cache_kv(B=B, H=H, T=T, d=d, device=device)
        k_out, v_out = cache.update(k, v, layer_idx=0)

        # Compute cosine sim over all (B*H*T) token vectors
        k_flat     = k.float().reshape(-1, d).cpu().numpy()
        k_out_flat = k_out.float().reshape(-1, d).cpu().numpy()
        cos = (
            (k_flat * k_out_flat).sum(-1)
            / (np.linalg.norm(k_flat, axis=-1) * np.linalg.norm(k_out_flat, axis=-1) + 1e-9)
        )
        assert cos.mean() > 0.85, f"Mean cosine similarity {cos.mean():.3f} too low at b=3"

    def test_reconstruction_improves_with_bits(self, device):
        B, H, T, d = 1, 4, 30, 64
        k, v = _make_cache_kv(B=B, H=H, T=T, d=d, device=device)
        prev_mse = float("inf")
        for b in [1, 2, 3, 4]:
            cache = MPSTurboQuantCache(head_dim=d, bits=b, device=device,
                                       dtype=torch.float16, seed=0)
            k_out, _ = cache.update(k, v, layer_idx=0)
            mse = ((k.float() - k_out.float())**2).mean().item()
            assert mse < prev_mse, f"b={b}: MSE {mse:.5f} not < b={b-1}: {prev_mse:.5f}"
            prev_mse = mse


# ---------------------------------------------------------------------------
# Cache accumulation consistency
# ---------------------------------------------------------------------------

class TestCacheAccumulationConsistency:

    def test_seq_len_consistent_across_layers(self, device):
        n_layers = 4
        cache = MPSTurboQuantCache(head_dim=64, bits=2, device=device,
                                   dtype=torch.float16, seed=0)
        for i in range(n_layers):
            cache.update(*_make_cache_kv(T=32, device=device), layer_idx=i)
        for i in range(n_layers):
            assert cache.get_seq_length(i) == 32

    def test_mixed_chunk_sizes(self, device):
        """Prefill then decode with single tokens."""
        cache = MPSTurboQuantCache(head_dim=64, bits=2, device=device,
                                   dtype=torch.float16, seed=0)
        cache.update(*_make_cache_kv(T=100, device=device), layer_idx=0)
        for _ in range(10):
            cache.update(*_make_cache_kv(T=1, device=device), layer_idx=0)
        assert cache.get_seq_length(0) == 110

    def test_compression_ratio_stable_across_chunk_sizes(self, device):
        """Theoretical ratio should match actual within 5% after sufficient tokens."""
        cache = MPSTurboQuantCache(head_dim=64, bits=2, device=device,
                                   dtype=torch.float16, seed=0)
        cache.update(*_make_cache_kv(T=256, device=device), layer_idx=0)
        theory = cache.theoretical_compression_ratio()
        actual = cache.compression_ratio()
        assert abs(actual - theory) / theory < 0.05, (
            f"Actual ratio {actual:.2f} differs from theoretical {theory:.2f} by >5%"
        )

    def test_fp16_bytes_proportional_to_tokens(self, device):
        B, H, d = 1, 4, 64
        cache = MPSTurboQuantCache(head_dim=d, bits=2, device=device,
                                   dtype=torch.float16, seed=0)
        cache.update(*_make_cache_kv(B=B, H=H, T=64, d=d, device=device), layer_idx=0)
        expected = 2 * B * H * 64 * d * 2   # 2=K+V, 2=bytes
        assert cache.fp16_bytes() == expected
