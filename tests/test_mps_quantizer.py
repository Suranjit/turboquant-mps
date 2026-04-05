"""
Tests for turboquant/mps_quantizer.py

Covers:
  - _storage_bits: all branches
  - pack_bits / unpack_bits: round-trips, all bit widths, non-power-of-vpb dims,
                             invalid n_bits, sign extension handling
  - pack_qjl / unpack_qjl: round-trips, edge dims
  - MPSTurboQuantMSE: quant/dequant shapes+dtypes, round-trip distortion,
                      quant_pack returns int8, dequant_unpack correctness
  - MPSTurboQuantProd: b=1 special case, general case, pack/unpack round-trip,
                       compressed_bytes_per_token_head formula, compression_ratio,
                       invalid b, ST = S.T
"""
import math

import numpy as np
import pytest
import torch

from tests.conftest import make_unit_tensors
from turboquant.mps_quantizer import (
    MPSTurboQuantMSE,
    MPSTurboQuantProd,
    _storage_bits,
    pack_bits,
    pack_qjl,
    unpack_bits,
    unpack_qjl,
)


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


# ---------------------------------------------------------------------------
# _storage_bits
# ---------------------------------------------------------------------------

class TestStorageBits:
    def test_b1(self):  assert _storage_bits(1) == 1
    def test_b2(self):  assert _storage_bits(2) == 2
    def test_b3(self):  assert _storage_bits(3) == 4   # 3 bits → 4-bit slot
    def test_b4(self):  assert _storage_bits(4) == 4
    def test_b5(self):  assert _storage_bits(5) == 8   # >4 → full byte
    def test_b8(self):  assert _storage_bits(8) == 8


# ---------------------------------------------------------------------------
# pack_bits / unpack_bits
# ---------------------------------------------------------------------------

class TestPackUnpackBits:

    @pytest.mark.parametrize("n_bits", [1, 2, 4])
    @pytest.mark.parametrize("d", [1, 7, 8, 63, 64, 65, 128, 256])
    def test_round_trip(self, device, n_bits, d):
        max_val = (1 << n_bits) - 1
        orig = torch.randint(0, max_val + 1, (4, d), device=device, dtype=torch.int32)
        packed = pack_bits(orig.clone(), n_bits)
        recovered = unpack_bits(packed, n_bits, d).to(torch.int32)
        assert torch.equal(orig, recovered), (
            f"n_bits={n_bits} d={d}: max_diff={( orig - recovered).abs().max()}"
        )

    @pytest.mark.parametrize("n_bits", [1, 2, 4])
    def test_packed_dtype_is_int8(self, device, n_bits):
        t = torch.zeros(10, 64, device=device, dtype=torch.int32)
        packed = pack_bits(t, n_bits)
        assert packed.dtype == torch.int8

    @pytest.mark.parametrize("n_bits", [1, 2, 4])
    def test_packed_width_correct(self, device, n_bits):
        d = 64
        t = torch.zeros(1, d, device=device, dtype=torch.int32)
        packed = pack_bits(t, n_bits)
        vpb = 8 // n_bits
        expected_cols = math.ceil(d / vpb)
        assert packed.shape[-1] == expected_cols

    def test_invalid_n_bits_raises(self, device):
        t = torch.zeros(4, 64, device=device, dtype=torch.int32)
        with pytest.raises(ValueError, match="n_bits"):
            pack_bits(t, 3)

    def test_all_ones_round_trip(self, device):
        """Max value for each bit width should survive packing."""
        for n_bits in [1, 2, 4]:
            max_val = (1 << n_bits) - 1
            orig = torch.full((2, 32), max_val, device=device, dtype=torch.int32)
            packed = pack_bits(orig, n_bits)
            recovered = unpack_bits(packed, n_bits, 32).to(torch.int32)
            assert torch.equal(orig, recovered)

    def test_batch_dims_preserved(self, device):
        """pack/unpack should work for any leading batch shape."""
        orig = torch.randint(0, 4, (2, 3, 64), device=device, dtype=torch.int32)
        packed = pack_bits(orig, 2)
        assert packed.shape[:2] == (2, 3)
        recovered = unpack_bits(packed, 2, 64)
        assert recovered.shape == (2, 3, 64)

    def test_sign_extension_handled(self, device):
        """Unpack must mask away int8 sign extension (values 0x80-0xFF)."""
        # Manually pack a byte where the high bit is set (0xFF = 0b11111111)
        # For n_bits=2: should decode to (3, 3, 3, 3)
        packed = torch.tensor([[-1]], device=device, dtype=torch.int8)  # 0xFF as int8
        recovered = unpack_bits(packed, n_bits=2, d=4).to(torch.int32)
        expected = torch.full((1, 4), 3, device=device, dtype=torch.int32)
        assert torch.equal(recovered, expected)


# ---------------------------------------------------------------------------
# pack_qjl / unpack_qjl
# ---------------------------------------------------------------------------

class TestPackUnpackQjl:

    @pytest.mark.parametrize("d", [1, 7, 8, 63, 64, 128])
    def test_round_trip(self, device, d):
        signs = torch.where(
            torch.randint(0, 2, (3, d), device=device) == 1,
            torch.ones(3, d, device=device, dtype=torch.int8),
            -torch.ones(3, d, device=device, dtype=torch.int8),
        )
        packed = pack_qjl(signs)
        recovered = unpack_qjl(packed, d)
        assert torch.equal(signs, recovered), f"d={d}: mismatch"

    def test_packed_dtype_is_int8(self, device):
        signs = torch.ones(4, 64, device=device, dtype=torch.int8)
        assert pack_qjl(signs).dtype == torch.int8

    def test_all_plus1(self, device):
        signs = torch.ones(2, 16, device=device, dtype=torch.int8)
        packed = pack_qjl(signs)
        recovered = unpack_qjl(packed, 16)
        assert torch.equal(recovered, signs)

    def test_all_minus1(self, device):
        signs = -torch.ones(2, 16, device=device, dtype=torch.int8)
        packed = pack_qjl(signs)
        recovered = unpack_qjl(packed, 16)
        assert torch.equal(recovered, signs)

    def test_packed_size_is_ceil_d_over_8(self, device):
        for d in [8, 9, 15, 16, 64, 65]:
            signs = torch.ones(1, d, device=device, dtype=torch.int8)
            packed = pack_qjl(signs)
            assert packed.shape[-1] == math.ceil(d / 8), f"d={d}"


# ---------------------------------------------------------------------------
# MPSTurboQuantMSE
# ---------------------------------------------------------------------------

class TestMPSTurboQuantMSE:

    @pytest.fixture
    def q(self, device):
        return MPSTurboQuantMSE(d=64, b=2, device=device, seed=0)

    # --- Construction ---

    def test_rotation_on_device(self, q, device):
        assert str(q.rotation.device).startswith(device)

    def test_codebook_on_device(self, q, device):
        assert str(q.codebook.device).startswith(device)

    def test_codebook_sorted(self, q):
        cb = q.codebook.cpu().float().numpy()
        assert np.all(np.diff(cb) > 0)

    def test_codebook_length(self, device):
        for b in [1, 2, 3, 4]:
            q = MPSTurboQuantMSE(d=32, b=b, device=device, seed=0)
            assert len(q.codebook) == 2**b

    # --- quant ---

    @pytest.mark.parametrize("d,b", [(32, 1), (64, 2), (128, 3)])
    def test_quant_shape(self, device, d, b):
        q = MPSTurboQuantMSE(d=d, b=b, device=device, seed=0)
        X = make_unit_tensors(8, d, device)
        idx = q.quant(X)
        assert idx.shape == (8, d)

    def test_quant_dtype(self, q, device):
        X = make_unit_tensors(4, 64, device)
        assert q.quant(X).dtype == torch.int16

    def test_quant_index_range(self, q, device):
        X = make_unit_tensors(100, 64, device)
        idx = q.quant(X)
        assert idx.min().item() >= 0
        assert idx.max().item() <= 3   # b=2 → 4 levels

    # --- dequant ---

    def test_dequant_shape(self, q, device):
        X = make_unit_tensors(8, 64, device)
        idx = q.quant(X)
        X_hat = q.dequant(idx)
        assert X_hat.shape == (8, 64)

    def test_dequant_dtype_matches(self, q, device):
        X = make_unit_tensors(4, 64, device)
        idx = q.quant(X)
        assert q.dequant(idx).dtype == q.dtype

    def test_dequant_high_bits_close(self, device):
        q = MPSTurboQuantMSE(d=64, b=4, device=device, seed=0)
        X = make_unit_tensors(200, 64, device, dtype=torch.float32)
        idx = q.quant(X)
        X_hat = q.dequant(idx)
        d_mse = ((X - X_hat)**2).sum(-1).mean().item()
        assert d_mse < 0.02, f"b=4 D_mse={d_mse:.4f}"

    # --- quant_pack / dequant_unpack ---

    @pytest.mark.parametrize("b", [1, 2, 3, 4, 5])
    def test_quant_pack_idx_dtype_int8(self, device, b):
        q = MPSTurboQuantMSE(d=64, b=b, device=device, seed=0)
        X = make_unit_tensors(4, 64, device)
        packed_idx, _ = q.quant_pack(X)
        assert packed_idx.dtype == torch.int8

    def test_quant_pack_norms_dtype_float16(self, q, device):
        X = make_unit_tensors(4, 64, device)
        _, norms = q.quant_pack(X)
        assert norms.dtype == torch.float16

    def test_quant_pack_norms_shape(self, q, device):
        X = make_unit_tensors(10, 64, device)
        _, norms = q.quant_pack(X)
        assert norms.shape == (10,)

    def test_quant_pack_norms_positive(self, q, device):
        rng = np.random.default_rng(0)
        X = torch.from_numpy(rng.standard_normal((10, 64)).astype(np.float32) * 3).to(device)
        _, norms = q.quant_pack(X)
        assert (norms > 0).all()

    @pytest.mark.parametrize("b", [1, 2, 4])
    def test_dequant_unpack_round_trip_shape(self, device, b):
        q = MPSTurboQuantMSE(d=64, b=b, device=device, seed=0)
        X = make_unit_tensors(6, 64, device) * 2.5
        packed_idx, norms = q.quant_pack(X)
        X_rec = q.dequant_unpack(packed_idx, norms)
        assert X_rec.shape == (6, 64)

    def test_dequant_unpack_high_bits_close(self, device):
        q = MPSTurboQuantMSE(d=64, b=4, device=device, seed=0)
        rng = np.random.default_rng(1)
        X = torch.from_numpy(rng.standard_normal((100, 64)).astype(np.float32)).to(device)
        packed_idx, norms = q.quant_pack(X)
        X_rec = q.dequant_unpack(packed_idx, norms)
        rmse = ((X - X_rec)**2).sum(-1).sqrt().mean().item()
        x_scale = X.norm(dim=-1).mean().item()
        assert rmse / x_scale < 0.15, f"Relative RMSE={rmse/x_scale:.3f} too high"

    def test_packed_idx_width_formula(self, device):
        """packed_idx should have ceil(b*d/8) columns when b≤4."""
        for b in [1, 2, 4]:
            d = 64
            q = MPSTurboQuantMSE(d=d, b=b, device=device, seed=0)
            packed_idx, _ = q.quant_pack(make_unit_tensors(4, d, device))
            expected_cols = math.ceil(b * d / 8)
            assert packed_idx.shape[-1] == expected_cols, f"b={b}"

    def test_b5_uses_full_byte_storage(self, device):
        """b=5 → _storage_bits=8 → one int8 per value (no sub-byte packing)."""
        d, b = 64, 5
        q = MPSTurboQuantMSE(d=d, b=b, device=device, seed=0)
        assert q._pack_bits == 8
        X = make_unit_tensors(4, d, device)
        packed_idx, norms = q.quant_pack(X)
        assert packed_idx.dtype == torch.int8
        assert packed_idx.shape[-1] == d   # one byte per value
        X_rec = q.dequant_unpack(packed_idx, norms)
        assert X_rec.shape == (4, d)


# ---------------------------------------------------------------------------
# MPSTurboQuantProd
# ---------------------------------------------------------------------------

class TestMPSTurboQuantProd:

    @pytest.fixture
    def q(self, device):
        return MPSTurboQuantProd(d=64, b=2, device=device, seed=0)

    # --- Construction ---

    def test_b0_raises(self, device):
        with pytest.raises(ValueError):
            MPSTurboQuantProd(d=32, b=0, device=device)

    def test_ST_equals_ST(self, q):
        """self.ST must be S.T (transpose)."""
        torch.testing.assert_close(q.ST, q.S.T.contiguous())

    def test_qjl_scale_formula(self, device):
        d = 64
        q = MPSTurboQuantProd(d=d, b=2, device=device, seed=0)
        expected = math.sqrt(math.pi / 2) / d
        assert abs(q._qjl_scale - expected) < 1e-10

    # --- quant ---

    @pytest.mark.parametrize("d,b", [(32, 1), (64, 2), (128, 3)])
    def test_quant_output_shapes(self, device, d, b):
        q = MPSTurboQuantProd(d=d, b=b, device=device, seed=0)
        X = make_unit_tensors(5, d, device)
        idx, qjl, gamma = q.quant(X)
        assert idx.shape == (5, d)
        assert qjl.shape == (5, d)
        assert gamma.shape == (5,)

    def test_quant_qjl_values_pm1(self, q, device):
        X = make_unit_tensors(50, 64, device)
        _, qjl, _ = q.quant(X)
        assert qjl.dtype == torch.int8
        vals = qjl.unique().cpu().tolist()
        assert set(vals).issubset({-1, 1}), f"QJL has unexpected values: {vals}"

    def test_quant_gamma_non_negative(self, q, device):
        X = make_unit_tensors(50, 64, device)
        _, _, gamma = q.quant(X)
        assert (gamma >= 0).all()

    def test_b1_idx_all_zeros(self, device):
        q = MPSTurboQuantProd(d=32, b=1, device=device, seed=0)
        X = make_unit_tensors(10, 32, device)
        idx, _, _ = q.quant(X)
        assert torch.equal(idx, torch.zeros_like(idx))

    def test_b1_gamma_equals_one_for_unit_input(self, device):
        """b=1: residual = x (unit norm), so gamma should be ≈ 1."""
        q = MPSTurboQuantProd(d=32, b=1, device=device, seed=0)
        X = make_unit_tensors(20, 32, device)
        _, _, gamma = q.quant(X)
        torch.testing.assert_close(gamma, torch.ones_like(gamma), atol=1e-3, rtol=0)

    # --- dequant ---

    @pytest.mark.parametrize("d,b", [(32, 1), (64, 2), (128, 3)])
    def test_dequant_output_shape(self, device, d, b):
        q = MPSTurboQuantProd(d=d, b=b, device=device, seed=0)
        X = make_unit_tensors(5, d, device)
        idx, qjl, gamma = q.quant(X)
        assert q.dequant(idx, qjl, gamma).shape == (5, d)

    def test_dequant_dtype(self, q, device):
        X = make_unit_tensors(4, 64, device)
        idx, qjl, gamma = q.quant(X)
        assert q.dequant(idx, qjl, gamma).dtype == q.dtype

    # --- quant_pack / dequant_unpack ---

    def test_quant_pack_returns_four_tensors(self, q, device):
        X = make_unit_tensors(4, 64, device)
        result = q.quant_pack(X)
        assert len(result) == 4

    @pytest.mark.parametrize("b", [1, 2, 3, 4])
    def test_quant_pack_dtypes(self, device, b):
        q = MPSTurboQuantProd(d=64, b=b, device=device, seed=0)
        X = make_unit_tensors(4, 64, device)
        pi, pq, pg, pn = q.quant_pack(X)
        assert pi.dtype == torch.int8,    f"b={b}: packed_idx dtype {pi.dtype}"
        assert pq.dtype == torch.int8,    f"b={b}: packed_qjl dtype {pq.dtype}"
        assert pg.dtype == torch.float16, f"b={b}: gamma dtype {pg.dtype}"
        assert pn.dtype == torch.float16, f"b={b}: norms dtype {pn.dtype}"

    @pytest.mark.parametrize("b", [1, 2, 4])
    def test_dequant_unpack_round_trip_shape(self, device, b):
        q = MPSTurboQuantProd(d=64, b=b, device=device, seed=0)
        X = make_unit_tensors(6, 64, device) * 3.0
        pi, pq, pg, pn = q.quant_pack(X)
        X_rec = q.dequant_unpack(pi, pq, pg, pn)
        assert X_rec.shape == (6, 64)

    def test_dequant_unpack_preserves_scale(self, device):
        """dequant_unpack should restore magnitude ≈ input magnitude."""
        q = MPSTurboQuantProd(d=128, b=3, device=device, seed=0)
        rng = np.random.default_rng(0)
        scale = 4.0
        X = torch.from_numpy(
            rng.standard_normal((50, 128)).astype(np.float32)
        ).to(device) * scale
        pi, pq, pg, pn = q.quant_pack(X)
        X_rec = q.dequant_unpack(pi, pq, pg, pn)
        ratio = X_rec.norm(dim=-1) / X.norm(dim=-1)
        # Magnitude ratio should be close to 1 (within 30% mean)
        assert abs(ratio.mean().item() - 1.0) < 0.30

    def test_quant_pack_norms_match_input_norms(self, device):
        q = MPSTurboQuantProd(d=64, b=2, device=device, seed=0)
        rng = np.random.default_rng(2)
        X = torch.from_numpy(rng.standard_normal((10, 64)).astype(np.float32)).to(device)
        _, _, _, norms = q.quant_pack(X)
        input_norms = X.norm(dim=-1).to(torch.float16)
        torch.testing.assert_close(norms, input_norms, atol=1e-2, rtol=0)

    # --- compressed_bytes_per_token_head ---

    def test_compressed_bytes_formula(self, device):
        """Verify the formula: ceil(pb*d/8) + ceil(d/8) + 4 bytes."""
        d, b = 64, 2
        q = MPSTurboQuantProd(d=d, b=b, device=device, seed=0)
        pb = _storage_bits(max(b - 1, 1))
        expected = math.ceil(pb * d / 8) + math.ceil(d / 8) + 2 + 2  # 2=float16
        assert q.compressed_bytes_per_token_head() == expected

    @pytest.mark.parametrize("d,b", [(64, 1), (64, 2), (128, 2), (64, 4)])
    def test_compressed_less_than_fp16(self, device, d, b):
        q = MPSTurboQuantProd(d=d, b=b, device=device, seed=0)
        assert q.compressed_bytes_per_token_head() < q.fp16_bytes_per_token_head()

    def test_compression_ratio_greater_one(self, device):
        q = MPSTurboQuantProd(d=64, b=2, device=device, seed=0)
        assert q.compression_ratio() > 1.0

    def test_fp16_bytes_formula(self, device):
        d = 64
        q = MPSTurboQuantProd(d=d, b=2, device=device, seed=0)
        assert q.fp16_bytes_per_token_head() == d * 2

    def test_b5_compressed_bytes_full_byte_path(self, device):
        """b=5 → MSE uses b-1=4 bits → _storage_bits(4)=4 < 8; b=6 → _storage_bits(5)=8."""
        d = 32
        q = MPSTurboQuantProd(d=d, b=6, device=device, seed=0)
        # max(b-1, 1) = 5 → _storage_bits(5) = 8 → idx_bytes = d
        pb = _storage_bits(max(6 - 1, 1))
        assert pb == 8
        cb = q.compressed_bytes_per_token_head()
        assert cb == d + math.ceil(d / 8) + 4   # idx_bytes=d + qjl + 2 scalars
        assert q.compression_ratio() > 1.0

    # --- device placement ---

    def test_all_tensors_on_device(self, q, device):
        assert str(q.S.device).startswith(device)
        assert str(q.ST.device).startswith(device)
        assert str(q.mse.rotation.device).startswith(device)

    def test_output_tensors_on_device(self, q, device):
        X = make_unit_tensors(4, 64, device)
        idx, qjl, gamma = q.quant(X)
        assert str(idx.device).startswith(device)
        assert str(qjl.device).startswith(device)
        assert str(gamma.device).startswith(device)
