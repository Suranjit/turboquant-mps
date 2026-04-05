"""
Tests for turboquant/kv_cache.py

Covers:
  - TurboQuantLayer (mode='mse' and 'prod'):
      update, shapes, seq_len accumulation, get_mask_sizes,
      get_max_cache_shape, compressed_bytes, fp16_bytes
  - TurboQuantDynamicCache:
      multi-layer update, get_seq_length, get_max_length, seen_tokens,
      __len__, compression_ratio, memory_summary, invalid mode error,
      b=1 edge case that previously returned compressed_bytes=0
"""
import numpy as np
import pytest
import torch

from turboquant.kv_cache import TurboQuantDynamicCache, TurboQuantLayer
from turboquant.quantizer import TurboQuantMSE, TurboQuantProd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kv(B=1, H=4, T=32, d=64):
    rng = torch.Generator().manual_seed(0)
    k = torch.randn(B, H, T, d, generator=rng)
    v = torch.randn(B, H, T, d, generator=rng)
    return k, v


def _make_layer(d=64, b=2, mode="prod"):
    if mode == "mse":
        kq = TurboQuantMSE(d, b, seed=0)
        vq = TurboQuantMSE(d, b, seed=1)
    else:
        kq = TurboQuantProd(d, b, seed=0)
        vq = TurboQuantProd(d, b, seed=1)
    return TurboQuantLayer(kq, vq, mode)


# ---------------------------------------------------------------------------
# TurboQuantLayer
# ---------------------------------------------------------------------------

class TestTurboQuantLayer:

    @pytest.mark.parametrize("mode", ["mse", "prod"])
    def test_update_returns_correct_shape(self, mode):
        layer = _make_layer(d=64, b=2, mode=mode)
        k, v = _make_kv(B=1, H=4, T=32, d=64)
        k_out, v_out = layer.update(k, v)
        assert k_out.shape == (1, 4, 32, 64)
        assert v_out.shape == (1, 4, 32, 64)

    @pytest.mark.parametrize("mode", ["mse", "prod"])
    def test_update_output_device(self, mode):
        layer = _make_layer(d=64, b=2, mode=mode)
        k, v = _make_kv()
        k_out, v_out = layer.update(k, v)
        assert k_out.device == k.device
        assert v_out.device == v.device

    @pytest.mark.parametrize("mode", ["mse", "prod"])
    def test_update_output_dtype(self, mode):
        layer = _make_layer(d=64, b=2, mode=mode)
        k, v = _make_kv()
        k_out, v_out = layer.update(k, v)
        assert k_out.dtype == k.dtype
        assert v_out.dtype == v.dtype

    def test_seq_len_accumulates(self):
        layer = _make_layer(d=64, b=2, mode="prod")
        assert layer.get_seq_length() == 0
        k, v = _make_kv(T=10)
        layer.update(k, v)
        assert layer.get_seq_length() == 10
        layer.update(k, v)
        assert layer.get_seq_length() == 20

    def test_output_shape_grows_with_each_step(self):
        layer = _make_layer(d=64, b=2, mode="prod")
        k, v = _make_kv(T=16)
        k_out, _ = layer.update(k, v)
        assert k_out.shape[2] == 16
        k2, v2 = _make_kv(T=1)
        k_out2, _ = layer.update(k2, v2)
        assert k_out2.shape[2] == 17

    def test_multiple_updates_concatenate_correctly(self):
        """Returned full KV must contain T_total tokens in the seq dim."""
        layer = _make_layer(d=32, b=2, mode="prod")
        chunks = [8, 4, 1, 1, 1]
        total = 0
        for t in chunks:
            k, v = _make_kv(H=2, T=t, d=32)
            k_out, v_out = layer.update(k, v)
            total += t
            assert k_out.shape[2] == total

    def test_get_mask_sizes(self):
        layer = _make_layer(d=64, b=2, mode="prod")
        layer.update(*_make_kv(T=50))
        cache_pos = torch.arange(1)   # 1 new token
        kv_len, kv_off = layer.get_mask_sizes(cache_pos)
        assert kv_len == 51   # 50 cached + 1 new
        assert kv_off == 0

    def test_get_mask_sizes_prefill(self):
        layer = _make_layer(d=64, b=2, mode="prod")
        # Before any update, mask for 32-token prefill
        cache_pos = torch.arange(32)
        kv_len, kv_off = layer.get_mask_sizes(cache_pos)
        assert kv_len == 32
        assert kv_off == 0

    def test_get_max_cache_shape(self):
        layer = _make_layer()
        assert layer.get_max_cache_shape() == -1

    def test_is_sliding_false(self):
        assert TurboQuantLayer.__dict__.get("is_sliding") == False or \
               _make_layer().is_sliding == False

    # --- memory accounting ---

    @pytest.mark.parametrize("mode,b", [("mse", 2), ("prod", 2), ("prod", 1), ("prod", 3)])
    def test_compressed_bytes_positive(self, mode, b):
        layer = _make_layer(d=64, b=b, mode=mode)
        layer.update(*_make_kv(T=32))
        assert layer.compressed_bytes() > 0

    def test_compressed_bytes_b1_not_zero(self):
        """b=1 previously returned 0 due to log2(idx.max()) bug; now fixed."""
        layer = _make_layer(d=64, b=1, mode="prod")
        layer.update(*_make_kv(T=32))
        assert layer.compressed_bytes() > 0

    def test_compressed_bytes_less_than_fp16(self):
        layer = _make_layer(d=64, b=2, mode="prod")
        layer.update(*_make_kv(T=64))
        assert layer.compressed_bytes() < layer.fp16_bytes()

    def test_fp16_bytes_formula(self):
        """fp16_bytes = 2 (K+V) * B * H * T * d * 2 bytes."""
        B, H, T, d = 1, 4, 32, 64
        layer = _make_layer(d=d, b=2, mode="prod")
        layer.update(*_make_kv(B=B, H=H, T=T, d=d))
        expected = 2 * B * H * T * d * 2
        assert layer.fp16_bytes() == expected

    def test_fp16_bytes_zero_before_update(self):
        layer = _make_layer()
        assert layer.fp16_bytes() == 0

    def test_mse_mode_compressed_bytes_formula(self):
        """MSE mode: ceil(b*d/8) * B*H*T + 4*B*H*T (float32 norms)."""
        import math
        B, H, T, d, b = 1, 4, 32, 64, 2
        layer = _make_layer(d=d, b=b, mode="mse")
        layer.update(*_make_kv(B=B, H=H, T=T, d=d))
        # key + value, same formula
        expected = 2 * (math.ceil(b * d / 8) * B * H * T + 4 * B * H * T)
        assert layer.compressed_bytes() == expected

    def test_prod_mode_compressed_bytes_formula(self):
        """Prod mode: ceil((b-1)*d/8) + ceil(d/8) + 4 + 4 per B*H*T, for K+V."""
        import math
        B, H, T, d, b = 1, 4, 32, 64, 2
        layer = _make_layer(d=d, b=b, mode="prod")
        layer.update(*_make_kv(B=B, H=H, T=T, d=d))
        b_mse = max(b - 1, 1)
        per = (math.ceil(b_mse * d / 8) + math.ceil(d / 8) + 4 + 4)
        expected = 2 * per * B * H * T
        assert layer.compressed_bytes() == expected


# ---------------------------------------------------------------------------
# TurboQuantDynamicCache
# ---------------------------------------------------------------------------

class TestTurboQuantDynamicCache:

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            TurboQuantDynamicCache(head_dim=64, bits=2, mode="invalid")

    def test_initial_seq_len_zero(self):
        cache = TurboQuantDynamicCache(head_dim=64, bits=2)
        assert cache.get_seq_length() == 0

    def test_initial_len_zero(self):
        cache = TurboQuantDynamicCache(head_dim=64, bits=2)
        assert len(cache) == 0

    def test_update_creates_layer(self):
        cache = TurboQuantDynamicCache(head_dim=64, bits=2)
        cache.update(*_make_kv(T=16), layer_idx=0)
        assert len(cache) == 1

    def test_update_multi_layer(self):
        cache = TurboQuantDynamicCache(head_dim=64, bits=2)
        for i in range(4):
            cache.update(*_make_kv(T=16), layer_idx=i)
        assert len(cache) == 4

    def test_get_seq_length_after_update(self):
        cache = TurboQuantDynamicCache(head_dim=64, bits=2)
        cache.update(*_make_kv(T=32), layer_idx=0)
        cache.update(*_make_kv(T=32), layer_idx=1)
        assert cache.get_seq_length(layer_idx=0) == 32
        assert cache.get_seq_length(layer_idx=1) == 32

    def test_get_seq_length_nonexistent_layer(self):
        cache = TurboQuantDynamicCache(head_dim=64, bits=2)
        assert cache.get_seq_length(layer_idx=99) == 0

    def test_get_max_length_returns_none(self):
        cache = TurboQuantDynamicCache(head_dim=64, bits=2)
        assert cache.get_max_length() is None

    def test_seen_tokens(self):
        cache = TurboQuantDynamicCache(head_dim=64, bits=2)
        cache.update(*_make_kv(T=20), layer_idx=0)
        assert cache.seen_tokens == 20

    def test_update_returns_correct_shapes(self):
        cache = TurboQuantDynamicCache(head_dim=64, bits=2)
        k, v = _make_kv(B=1, H=4, T=32, d=64)
        k_out, v_out = cache.update(k, v, layer_idx=0)
        assert k_out.shape == (1, 4, 32, 64)
        assert v_out.shape == (1, 4, 32, 64)

    def test_multi_step_decode_shape_grows(self):
        cache = TurboQuantDynamicCache(head_dim=64, bits=2)
        for step in range(5):
            k, v = _make_kv(T=1, d=64)
            k_out, _ = cache.update(k, v, layer_idx=0)
            assert k_out.shape[2] == step + 1

    def test_get_mask_sizes_dispatch(self):
        cache = TurboQuantDynamicCache(head_dim=64, bits=2)
        cache.update(*_make_kv(T=30), layer_idx=0)
        kv_len, kv_off = cache.get_mask_sizes(torch.arange(1), layer_idx=0)
        assert kv_len == 31
        assert kv_off == 0

    def test_get_mask_sizes_no_layer(self):
        cache = TurboQuantDynamicCache(head_dim=64, bits=2)
        kv_len, kv_off = cache.get_mask_sizes(torch.arange(5), layer_idx=99)
        assert kv_len == 5
        assert kv_off == 0

    # --- memory reporting ---

    @pytest.mark.parametrize("b", [1, 2, 3])
    def test_compressed_bytes_positive(self, b):
        cache = TurboQuantDynamicCache(head_dim=64, bits=b)
        cache.update(*_make_kv(T=32), layer_idx=0)
        assert cache.compressed_bytes() > 0

    def test_compression_ratio_greater_than_one(self):
        cache = TurboQuantDynamicCache(head_dim=64, bits=2)
        cache.update(*_make_kv(T=64), layer_idx=0)
        assert cache.compression_ratio() > 1.0

    def test_compression_ratio_zero_empty(self):
        cache = TurboQuantDynamicCache(head_dim=64, bits=2)
        assert cache.compression_ratio() == 0.0

    def test_memory_summary_contains_ratio(self):
        cache = TurboQuantDynamicCache(head_dim=64, bits=2)
        cache.update(*_make_kv(T=32), layer_idx=0)
        s = cache.memory_summary()
        assert "Ratio" in s
        assert "KB" in s or "MB" in s

    @pytest.mark.parametrize("mode", ["mse", "prod"])
    def test_mode_roundtrip_output_close(self, mode):
        """Dequantized output should be close to input at b=4."""
        cache = TurboQuantDynamicCache(head_dim=64, bits=4, mode=mode)
        k, v = _make_kv(B=1, H=2, T=10, d=64)
        k_out, v_out = cache.update(k, v, layer_idx=0)
        # Normalise both then compare cosine similarity per token
        k_np = k.float().numpy().reshape(-1, 64)
        k_out_np = k_out.float().numpy().reshape(-1, 64)
        cosine = (
            (k_np * k_out_np).sum(-1)
            / (np.linalg.norm(k_np, axis=-1) * np.linalg.norm(k_out_np, axis=-1) + 1e-9)
        )
        assert cosine.mean() > 0.9, f"mode={mode}: mean cosine={cosine.mean():.3f}"
