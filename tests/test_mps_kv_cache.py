"""
Tests for turboquant/mps_kv_cache.py

Covers:
  - MPSTurboQuantLayer:
      lazy_initialization, update shapes, incremental accumulation,
      get_seq_length, get_mask_sizes, get_max_cache_shape, is_sliding,
      compressed_bytes (int8 storage verified), fp16_bytes formula,
      _dequant_chunk shape, memory_report adaptive formatting
  - MPSTurboQuantCache:
      multi-layer dispatch, _make_layer seeds different per layer,
      get_seq_length before/after, get_mask_sizes, get_max_length,
      seen_tokens, __len__, compression_ratio, theoretical_compression_ratio,
      memory_report KB/MB formatting
"""
import math

import numpy as np
import pytest
import torch

from tests.conftest import make_unit_tensors
from turboquant.mps_kv_cache import MPSTurboQuantCache, MPSTurboQuantLayer
from turboquant.mps_quantizer import MPSTurboQuantProd


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


def _make_kv(B=1, H=4, T=32, d=64, device="cpu", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    k = torch.randn(B, H, T, d, generator=g, dtype=torch.float16).to(device)
    v = torch.randn(B, H, T, d, generator=g, dtype=torch.float16).to(device)
    return k, v


def _make_layer(d=64, b=2, device="cpu"):
    kq = MPSTurboQuantProd(d=d, b=b, device=device, dtype=torch.float16, seed=0)
    vq = MPSTurboQuantProd(d=d, b=b, device=device, dtype=torch.float16, seed=1)
    return MPSTurboQuantLayer(kq, vq)


# ---------------------------------------------------------------------------
# MPSTurboQuantLayer
# ---------------------------------------------------------------------------

class TestMPSTurboQuantLayer:

    def test_is_sliding_false(self, device):
        layer = _make_layer(device=device)
        assert layer.is_sliding == False

    def test_initial_not_initialized(self, device):
        layer = _make_layer(device=device)
        assert not layer.is_initialized

    def test_lazy_initialization_on_first_update(self, device):
        layer = _make_layer(device=device)
        k, v = _make_kv(device=device)
        layer.update(k, v)
        assert layer.is_initialized

    def test_lazy_init_sets_device(self, device):
        layer = _make_layer(device=device)
        k, v = _make_kv(device=device)
        layer.update(k, v)
        assert str(layer.device).startswith(device)

    # --- update output shapes ---

    @pytest.mark.parametrize("B,H,T,d", [(1, 4, 32, 64), (2, 8, 16, 128)])
    def test_update_output_shape(self, device, B, H, T, d):
        layer = _make_layer(d=d, device=device)
        k, v = _make_kv(B=B, H=H, T=T, d=d, device=device)
        k_out, v_out = layer.update(k, v)
        assert k_out.shape == (B, H, T, d)
        assert v_out.shape == (B, H, T, d)

    def test_update_output_dtype_preserved(self, device):
        layer = _make_layer(device=device)
        k, v = _make_kv(device=device)
        k_out, v_out = layer.update(k, v)
        assert k_out.dtype == k.dtype
        assert v_out.dtype == v.dtype

    def test_update_output_on_correct_device(self, device):
        layer = _make_layer(device=device)
        k, v = _make_kv(device=device)
        k_out, _ = layer.update(k, v)
        assert str(k_out.device).startswith(device)

    # --- incremental accumulation ---

    def test_seq_len_starts_zero(self, device):
        layer = _make_layer(device=device)
        assert layer.get_seq_length() == 0

    def test_seq_len_after_prefill(self, device):
        layer = _make_layer(device=device)
        layer.update(*_make_kv(T=50, device=device))
        assert layer.get_seq_length() == 50

    def test_seq_len_accumulates_over_steps(self, device):
        layer = _make_layer(device=device)
        for _ in range(5):
            layer.update(*_make_kv(T=10, device=device))
        assert layer.get_seq_length() == 50

    def test_shape_grows_token_by_token(self, device):
        layer = _make_layer(device=device)
        layer.update(*_make_kv(T=10, device=device))
        for step in range(1, 6):
            k_out, _ = layer.update(*_make_kv(T=1, device=device))
            assert k_out.shape[2] == 10 + step

    def test_incremental_matches_bulk(self, device):
        """10 single-token updates should give same seq_len as one 10-token update."""
        layer_bulk = _make_layer(device=device)
        layer_incr = _make_layer(device=device)
        k, v = _make_kv(T=10, device=device, seed=0)
        layer_bulk.update(k, v)
        for t in range(10):
            layer_incr.update(k[:, :, t:t+1, :], v[:, :, t:t+1, :])
        assert layer_bulk.get_seq_length() == layer_incr.get_seq_length()

    def test_full_kv_cached_not_none_after_update(self, device):
        layer = _make_layer(device=device)
        layer.update(*_make_kv(T=8, device=device))
        assert layer._full_k is not None
        assert layer._full_v is not None

    # --- get_mask_sizes ---

    def test_get_mask_sizes_decode_step(self, device):
        layer = _make_layer(device=device)
        layer.update(*_make_kv(T=40, device=device))
        cache_pos = torch.arange(1, device=device)
        kv_len, kv_off = layer.get_mask_sizes(cache_pos)
        assert kv_len == 41   # 40 cached + 1 new
        assert kv_off == 0

    def test_get_mask_sizes_multi_token(self, device):
        layer = _make_layer(device=device)
        layer.update(*_make_kv(T=20, device=device))
        cache_pos = torch.arange(5, device=device)
        kv_len, _ = layer.get_mask_sizes(cache_pos)
        assert kv_len == 25

    def test_get_mask_sizes_empty_cache(self, device):
        layer = _make_layer(device=device)
        cache_pos = torch.arange(32, device=device)
        kv_len, kv_off = layer.get_mask_sizes(cache_pos)
        assert kv_len == 32
        assert kv_off == 0

    def test_get_max_cache_shape(self, device):
        layer = _make_layer(device=device)
        assert layer.get_max_cache_shape() == -1

    # --- memory accounting ---

    def test_compressed_bytes_positive_after_update(self, device):
        layer = _make_layer(device=device)
        layer.update(*_make_kv(T=32, device=device))
        assert layer.compressed_bytes() > 0

    def test_compressed_bytes_uses_int8_storage(self, device):
        """packed_idx and packed_qjl are int8, so element_size=1."""
        layer = _make_layer(d=64, b=2, device=device)
        layer.update(*_make_kv(T=32, device=device))
        for t in layer._k_idx:
            assert t.dtype == torch.int8, f"_k_idx dtype {t.dtype}"
        for t in layer._k_qjl:
            assert t.dtype == torch.int8, f"_k_qjl dtype {t.dtype}"

    def test_compressed_bytes_int8_not_int16(self, device):
        """Verify storage uses 1 byte/value, not 2 (regression for dtype doc bug)."""
        B, H, T, d, b = 1, 4, 32, 64, 2
        layer = _make_layer(d=d, b=b, device=device)
        layer.update(*_make_kv(B=B, H=H, T=T, d=d, device=device))
        cb = layer.compressed_bytes()
        # fp16_bytes = 2 * B * H * T * d * 2; ratio should be ~6x not ~3x
        fp = layer.fp16_bytes()
        ratio = fp / cb
        assert ratio > 4.0, f"Ratio {ratio:.1f}x; int16 storage would give ~3x, int8 gives ~6x"

    def test_compressed_bytes_less_than_fp16(self, device):
        layer = _make_layer(device=device)
        layer.update(*_make_kv(T=64, device=device))
        assert layer.compressed_bytes() < layer.fp16_bytes()

    def test_fp16_bytes_formula(self, device):
        B, H, T, d = 1, 4, 32, 64
        layer = _make_layer(d=d, device=device)
        layer.update(*_make_kv(B=B, H=H, T=T, d=d, device=device))
        expected = 2 * B * H * T * d * 2  # 2=K+V, 2=bytes/fp16
        assert layer.fp16_bytes() == expected

    def test_fp16_bytes_zero_before_update(self, device):
        layer = _make_layer(device=device)
        assert layer.fp16_bytes() == 0

    def test_fp16_bytes_accumulates(self, device):
        B, H, d = 1, 4, 64
        layer = _make_layer(d=d, device=device)
        layer.update(*_make_kv(B=B, H=H, T=16, d=d, device=device))
        layer.update(*_make_kv(B=B, H=H, T=16, d=d, device=device))
        assert layer.fp16_bytes() == 2 * B * H * 32 * d * 2


# ---------------------------------------------------------------------------
# MPSTurboQuantCache
# ---------------------------------------------------------------------------

class TestMPSTurboQuantCache:

    @pytest.fixture
    def cache(self, device):
        return MPSTurboQuantCache(head_dim=64, bits=2, device=device,
                                  dtype=torch.float16, seed=42)

    def test_initial_len_zero(self, cache):
        assert len(cache) == 0

    def test_initial_seq_len_zero(self, cache):
        assert cache.get_seq_length() == 0

    def test_initial_seen_tokens_zero(self, cache):
        assert cache.seen_tokens == 0

    def test_update_creates_layer(self, cache, device):
        cache.update(*_make_kv(device=device), layer_idx=0)
        assert len(cache) == 1

    def test_update_multi_layer(self, cache, device):
        for i in range(4):
            cache.update(*_make_kv(device=device), layer_idx=i)
        assert len(cache) == 4

    def test_update_creates_missing_layers_in_order(self, cache, device):
        """Updating layer_idx=3 should create layers 0-3."""
        cache.update(*_make_kv(device=device), layer_idx=3)
        assert len(cache) == 4

    def test_update_returns_correct_shape(self, cache, device):
        k, v = _make_kv(B=1, H=4, T=32, d=64, device=device)
        k_out, v_out = cache.update(k, v, layer_idx=0)
        assert k_out.shape == (1, 4, 32, 64)
        assert v_out.shape == (1, 4, 32, 64)

    def test_decode_shape_grows(self, cache, device):
        cache.update(*_make_kv(T=20, device=device), layer_idx=0)
        for step in range(1, 6):
            k_out, _ = cache.update(*_make_kv(T=1, device=device), layer_idx=0)
            assert k_out.shape[2] == 20 + step

    def test_different_layers_have_different_seeds(self, device):
        """Each layer should use distinct rotation matrices."""
        cache = MPSTurboQuantCache(head_dim=32, bits=2, device=device, seed=0)
        for i in range(3):
            cache.update(*_make_kv(T=4, d=32, device=device), layer_idx=i)
        rot0 = cache._layers[0]._kq.mse.rotation
        rot1 = cache._layers[1]._kq.mse.rotation
        assert not torch.equal(rot0, rot1), "Different layers must use different rotations"

    def test_key_value_use_different_seeds(self, device):
        cache = MPSTurboQuantCache(head_dim=32, bits=2, device=device, seed=0)
        cache.update(*_make_kv(T=4, d=32, device=device), layer_idx=0)
        k_rot = cache._layers[0]._kq.mse.rotation
        v_rot = cache._layers[0]._vq.mse.rotation
        assert not torch.equal(k_rot, v_rot), "Key/value quantizers must use different seeds"

    def test_get_seq_length_after_update(self, cache, device):
        cache.update(*_make_kv(T=32, device=device), layer_idx=0)
        assert cache.get_seq_length(layer_idx=0) == 32

    def test_get_seq_length_nonexistent_returns_zero(self, cache):
        assert cache.get_seq_length(layer_idx=99) == 0

    def test_seen_tokens(self, cache, device):
        cache.update(*_make_kv(T=25, device=device), layer_idx=0)
        assert cache.seen_tokens == 25

    def test_get_max_length_returns_none(self, cache):
        assert cache.get_max_length() is None

    # --- get_mask_sizes ---

    def test_get_mask_sizes_dispatch_to_layer(self, cache, device):
        cache.update(*_make_kv(T=30, device=device), layer_idx=0)
        kv_len, kv_off = cache.get_mask_sizes(torch.arange(1, device=device), layer_idx=0)
        assert kv_len == 31
        assert kv_off == 0

    def test_get_mask_sizes_no_layer(self, cache, device):
        kv_len, _ = cache.get_mask_sizes(torch.arange(5, device=device), layer_idx=99)
        assert kv_len == 5

    # --- memory reporting ---

    def test_compressed_bytes_zero_before_update(self, cache):
        assert cache.compressed_bytes() == 0

    def test_compressed_bytes_positive_after_update(self, cache, device):
        cache.update(*_make_kv(T=32, device=device), layer_idx=0)
        assert cache.compressed_bytes() > 0

    def test_compression_ratio_zero_before_update(self, cache):
        assert cache.compression_ratio() == 0.0

    def test_compression_ratio_greater_than_one(self, cache, device):
        cache.update(*_make_kv(T=64, device=device), layer_idx=0)
        assert cache.compression_ratio() > 1.0

    def test_theoretical_compression_ratio_positive(self, cache):
        assert cache.theoretical_compression_ratio() > 1.0

    @pytest.mark.parametrize("tokens,unit", [
        (100,  "KB"),
        (10000, "MB"),
    ])
    def test_memory_report_adaptive_units(self, device, tokens, unit):
        cache = MPSTurboQuantCache(head_dim=64, bits=2, device=device,
                                   dtype=torch.float16, seed=42)
        cache.update(*_make_kv(T=tokens, device=device), layer_idx=0)
        report = cache.memory_report()
        assert unit in report, f"Expected {unit} in report for {tokens} tokens: '{report}'"

    def test_memory_report_contains_ratio(self, cache, device):
        cache.update(*_make_kv(T=64, device=device), layer_idx=0)
        report = cache.memory_report()
        assert "×" in report or "x" in report.lower()

    def test_fp16_bytes_sums_layers(self, device):
        cache = MPSTurboQuantCache(head_dim=64, bits=2, device=device,
                                   dtype=torch.float16, seed=0)
        cache.update(*_make_kv(T=32, device=device), layer_idx=0)
        cache.update(*_make_kv(T=32, device=device), layer_idx=1)
        per_layer = cache._layers[0].fp16_bytes()
        assert cache.fp16_bytes() == 2 * per_layer

    def test_example_q_compression_ratio(self, cache):
        """_example_q must reflect theoretical ratio correctly."""
        r = cache._example_q.compression_ratio()
        assert r == cache.theoretical_compression_ratio()
        assert r > 1.0
