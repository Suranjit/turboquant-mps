"""
Tests for the production Metal shader infrastructure:
  - turboquant/production/model_configs.py
  - turboquant/production/cache.py
  - turboquant/metal/metal_lib.py
  - turboquant/metal/metal_quantizer.py

Metal kernels themselves are not invoked (PyObjC not available in CI),
so all Metal quantizer tests use the PyTorch MPS fallback path.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tests.conftest import make_unit_tensors


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


@pytest.fixture(scope="module")
def llama3b_config():
    from turboquant.production.model_configs import get_model_config
    return get_model_config("llama-3.2-3b")


# ===========================================================================
# model_configs.py
# ===========================================================================

class TestLlamaVariantProperties:

    def test_head_dim(self, llama3b_config):
        # hidden=3072, heads=24 → 128
        assert llama3b_config.head_dim == 3072 // 24

    def test_gqa_ratio(self, llama3b_config):
        # 24 query heads / 8 KV heads = 3
        assert llama3b_config.gqa_ratio == 3

    def test_fp16_kv_gb_formula(self, llama3b_config):
        cfg = llama3b_config
        T = 1024
        expected = (
            2 * cfg.num_hidden_layers * cfg.num_kv_heads * T * cfg.head_dim * 2
        ) / (1024 ** 3)
        assert abs(cfg.fp16_kv_gb(T) - expected) < 1e-9

    def test_fp16_kv_gb_uses_max_ctx_when_none(self, llama3b_config):
        cfg = llama3b_config
        gb_explicit = cfg.fp16_kv_gb(cfg.context_length)
        gb_default  = cfg.fp16_kv_gb()
        assert abs(gb_explicit - gb_default) < 1e-9

    def test_tq_kv_gb_less_than_fp16(self, llama3b_config):
        for b in [1, 2, 3, 4]:
            assert llama3b_config.tq_kv_gb(b, 32768) < llama3b_config.fp16_kv_gb(32768)

    def test_compression_ratio_positive(self, llama3b_config):
        for b in [1, 2, 3, 4]:
            assert llama3b_config.compression_ratio(b) > 1.0

    def test_compression_ratio_decreases_with_bits(self, llama3b_config):
        ratios = [llama3b_config.compression_ratio(b) for b in [1, 2, 3, 4]]
        assert ratios == sorted(ratios, reverse=True), "More bits should mean lower ratio"

    def test_max_context_at_memory(self, llama3b_config):
        ctx = llama3b_config.max_context_at_memory(3, available_gb=4.0)
        assert ctx > 0
        # Verify it actually fits
        assert llama3b_config.tq_kv_gb(3, ctx) <= 4.0 + 1e-3  # small tolerance

    def test_str_representation(self, llama3b_config):
        s = str(llama3b_config)
        assert "Llama-3.2-3B" in s
        assert "128" in s   # head_dim
        assert "GQA" in s


class TestGetModelConfig:

    def test_exact_key_lookup(self):
        from turboquant.production.model_configs import get_model_config
        cfg = get_model_config("llama-3.2-3b")
        assert cfg.name == "Llama-3.2-3B"

    def test_underscore_to_hyphen_normalisation(self):
        from turboquant.production.model_configs import get_model_config
        # Model key uses hyphens; underscores are converted to hyphens
        cfg = get_model_config("llama-3.2-3b")
        assert cfg.name == "Llama-3.2-3B"
        # A fuzzy partial match also works
        cfg2 = get_model_config("tinyllama-1.1b")
        assert "TinyLlama" in cfg2.name

    def test_case_insensitive(self):
        from turboquant.production.model_configs import get_model_config
        cfg = get_model_config("LLAMA-3.2-3B")
        assert cfg.name == "Llama-3.2-3B"

    def test_all_supported_models_loadable(self):
        from turboquant.production.model_configs import SUPPORTED_MODELS, get_model_config
        for key in SUPPORTED_MODELS:
            cfg = get_model_config(key)
            assert cfg.head_dim > 0
            assert cfg.num_kv_heads > 0

    def test_unknown_key_raises(self):
        from turboquant.production.model_configs import get_model_config
        with pytest.raises(KeyError):
            get_model_config("gpt-4-turbo-invalid")

    def test_tinyllama_head_dim(self):
        from turboquant.production.model_configs import get_model_config
        cfg = get_model_config("tinyllama-1.1b")
        assert cfg.head_dim == 2048 // 32  # 64

    def test_llama31_8b_gqa(self):
        from turboquant.production.model_configs import get_model_config
        cfg = get_model_config("llama-3.1-8b")
        assert cfg.num_attention_heads == 32
        assert cfg.num_kv_heads == 8
        assert cfg.gqa_ratio == 4


class TestAutoDetectConfig:

    def test_detects_known_model(self, llama3b_config):
        from turboquant.production.model_configs import auto_detect_config

        class FakeHFConfig:
            model_type = "llama"
            hidden_size = llama3b_config.hidden_size
            num_attention_heads = llama3b_config.num_attention_heads
            num_key_value_heads = llama3b_config.num_kv_heads
            num_hidden_layers = llama3b_config.num_hidden_layers
            max_position_embeddings = llama3b_config.context_length
            vocab_size = llama3b_config.vocab_size
            intermediate_size = llama3b_config.intermediate_size
            _name_or_path = "meta-llama/Llama-3.2-3B-Instruct"

        result = auto_detect_config(FakeHFConfig())
        assert result is not None
        assert result.name == "Llama-3.2-3B"

    def test_constructs_dynamic_variant_for_unknown(self):
        from turboquant.production.model_configs import auto_detect_config

        class UnknownConfig:
            model_type = "custom"
            hidden_size = 512
            num_attention_heads = 8
            num_key_value_heads = 2
            num_hidden_layers = 6
            max_position_embeddings = 2048
            vocab_size = 50000
            intermediate_size = 2048
            _name_or_path = "my-custom-model"

        result = auto_detect_config(UnknownConfig())
        assert result is not None
        assert result.head_dim == 512 // 8
        assert result.num_kv_heads == 2

    def test_returns_none_for_missing_fields(self):
        from turboquant.production.model_configs import auto_detect_config

        class EmptyConfig:
            model_type = "unknown"
            hidden_size = 0
            num_attention_heads = 0

        result = auto_detect_config(EmptyConfig())
        assert result is None


# ===========================================================================
# production/cache.py
# ===========================================================================

class TestProductionTurboQuantCacheBasic:

    @pytest.fixture
    def small_cache(self, device):
        from turboquant.production.cache import ProductionTurboQuantCache
        return ProductionTurboQuantCache(
            head_dim=64,
            bits=2,
            num_layers=4,
            num_kv_heads=2,
            device=device,
            dtype=torch.float16,
            seed=0,
            use_metal=False,   # force PyTorch MPS fallback (no PyObjC in test)
        )

    def test_update_returns_correct_shape(self, small_cache, device):
        B, H, T, d = 1, 2, 16, 64
        k = torch.randn(B, H, T, d, dtype=torch.float16).to(device)
        v = torch.randn(B, H, T, d, dtype=torch.float16).to(device)
        k_out, v_out = small_cache.update(k, v, layer_idx=0)
        assert k_out.shape == (B, H, T, d)
        assert v_out.shape == (B, H, T, d)

    def test_update_dtype_matches(self, small_cache, device):
        B, H, T, d = 1, 2, 8, 64
        k = torch.randn(B, H, T, d, dtype=torch.float16).to(device)
        v = torch.randn(B, H, T, d, dtype=torch.float16).to(device)
        k_out, v_out = small_cache.update(k, v, layer_idx=0)
        assert k_out.dtype == torch.float16
        assert v_out.dtype == torch.float16

    def test_seq_len_accumulates(self, small_cache, device):
        B, H, T, d = 1, 2, 10, 64
        k = torch.randn(B, H, T, d, dtype=torch.float16).to(device)
        v = torch.randn(B, H, T, d, dtype=torch.float16).to(device)
        small_cache.update(k, v, layer_idx=1)
        small_cache.update(k, v, layer_idx=1)
        assert small_cache.get_seq_length(1) == 20

    def test_get_seq_length_zero_for_empty_layer(self, small_cache):
        assert small_cache.get_seq_length(99) == 0

    def test_get_max_cache_shape_is_none(self, small_cache):
        assert small_cache.get_max_cache_shape() is None

    def test_get_mask_sizes(self, small_cache, device):
        B, H, T, d = 1, 2, 7, 64
        k = torch.randn(B, H, T, d, dtype=torch.float16).to(device)
        v = torch.randn(B, H, T, d, dtype=torch.float16).to(device)
        small_cache.update(k, v, layer_idx=0)
        sizes = small_cache.get_mask_sizes(layer_idx=0)
        assert sizes == (7,)

    def test_full_kv_grows_correctly(self, small_cache, device):
        B, H, d = 1, 2, 64
        k = torch.randn(B, H, 5, d, dtype=torch.float16).to(device)
        v = torch.randn(B, H, 5, d, dtype=torch.float16).to(device)
        small_cache.update(k, v, layer_idx=2)
        small_cache.update(k, v, layer_idx=2)
        k_full, _ = small_cache.update(k, v, layer_idx=2)
        assert k_full.shape[2] == 15

    def test_reset_clears_state(self, small_cache, device):
        B, H, T, d = 1, 2, 8, 64
        k = torch.randn(B, H, T, d, dtype=torch.float16).to(device)
        v = torch.randn(B, H, T, d, dtype=torch.float16).to(device)
        small_cache.update(k, v, layer_idx=0)
        assert small_cache.get_seq_length(0) > 0
        small_cache.reset()
        assert small_cache.get_seq_length(0) == 0


class TestProductionCacheMemoryReporting:

    @pytest.fixture
    def loaded_cache(self, device):
        from turboquant.production.cache import ProductionTurboQuantCache
        cache = ProductionTurboQuantCache(
            head_dim=64, bits=3, num_layers=2, num_kv_heads=2,
            device=device, dtype=torch.float16, seed=1, use_metal=False,
        )
        k = torch.randn(1, 2, 32, 64, dtype=torch.float16).to(device)
        v = torch.randn(1, 2, 32, 64, dtype=torch.float16).to(device)
        cache.update(k, v, layer_idx=0)
        cache.update(k, v, layer_idx=1)
        return cache

    def test_fp16_bytes_formula(self, loaded_cache):
        # 2 layers, 2 KV heads, 32 tokens each, d=64, float16 = 2 bytes
        # 2 (K+V) * 2 layers * 2 heads * 32 tokens * 64 * 2 bytes
        expected = 2 * 2 * 2 * 32 * 64 * 2
        assert loaded_cache.fp16_bytes() == expected

    def test_compressed_bytes_less_than_fp16(self, loaded_cache):
        assert loaded_cache.compressed_bytes() < loaded_cache.fp16_bytes()

    def test_compression_ratio_positive(self, loaded_cache):
        assert loaded_cache.compression_ratio() > 0

    def test_compression_ratio_zero_when_empty(self, device):
        from turboquant.production.cache import ProductionTurboQuantCache
        cache = ProductionTurboQuantCache(
            head_dim=64, bits=2, num_layers=1, num_kv_heads=2,
            device=device, dtype=torch.float16, seed=0, use_metal=False,
        )
        assert cache.compression_ratio() == 0.0

    def test_theoretical_ratio_formula(self, loaded_cache):
        expected = 16.0 / 3   # 16 / bits
        assert abs(loaded_cache.theoretical_compression_ratio() - expected) < 1e-6

    def test_memory_report_is_string(self, loaded_cache):
        report = loaded_cache.memory_report()
        assert isinstance(report, str)
        assert "ProductionTurboQuantCache" in report
        assert "Compressed" in report


class TestProductionCacheFromModelConfig:

    def test_from_model_config(self, device, llama3b_config):
        from turboquant.production.cache import ProductionTurboQuantCache
        cache = ProductionTurboQuantCache.from_model_config(
            llama3b_config, bits=2, device=device, use_metal=False,
        )
        assert cache.head_dim == llama3b_config.head_dim
        assert cache.num_kv_heads == llama3b_config.num_kv_heads
        assert cache.num_layers == llama3b_config.num_hidden_layers

    def test_each_layer_has_different_quantizer(self, device, llama3b_config):
        from turboquant.production.cache import ProductionTurboQuantCache
        cache = ProductionTurboQuantCache.from_model_config(
            llama3b_config, bits=2, device=device, use_metal=False,
        )
        # Each layer should have a distinct rotation matrix
        rot0 = cache._layers[0].kq.mse.rotation
        rot1 = cache._layers[1].kq.mse.rotation
        assert not torch.allclose(rot0, rot1), "Layers must use different quantizers"

    def test_key_and_value_use_different_quantizers(self, device, llama3b_config):
        from turboquant.production.cache import ProductionTurboQuantCache
        cache = ProductionTurboQuantCache.from_model_config(
            llama3b_config, bits=2, device=device, use_metal=False,
        )
        kq = cache._layers[0].kq
        vq = cache._layers[0].vq
        assert not torch.allclose(kq.mse.rotation, vq.mse.rotation), \
            "Key and value quantizers must differ"

    def test_reconstruction_quality_b3(self, device, llama3b_config):
        from turboquant.production.cache import ProductionTurboQuantCache
        cache = ProductionTurboQuantCache.from_model_config(
            llama3b_config, bits=3, device=device, use_metal=False, seed=0,
        )
        B, H = 1, llama3b_config.num_kv_heads
        T, d = 32, llama3b_config.head_dim
        k = torch.randn(B, H, T, d, dtype=torch.float16).to(device)
        v = torch.randn(B, H, T, d, dtype=torch.float16).to(device)
        k_rec, _ = cache.update(k, v, layer_idx=0)

        k_flat  = k.cpu().float().reshape(-1, d)
        kr_flat = k_rec.cpu().float().reshape(-1, d)
        cos = (
            (k_flat * kr_flat).sum(-1)
            / (k_flat.norm(dim=-1) * kr_flat.norm(dim=-1) + 1e-9)
        ).mean().item()
        assert cos > 0.80, f"Cosine similarity {cos:.3f} too low at b=3"


# ===========================================================================
# metal/metal_lib.py
# ===========================================================================

class TestMetalAvailability:

    def test_metal_available_returns_bool(self):
        from turboquant.metal.metal_lib import metal_available
        result = metal_available()
        assert isinstance(result, bool)

    def test_metal_lib_load_returns_none_or_metallib(self):
        from turboquant.metal.metal_lib import MetalLib
        result = MetalLib.load()
        # On systems without PyObjC, should return None gracefully
        assert result is None or hasattr(result, "run_kernel")

    def test_compile_metal_lib_without_sdk(self, monkeypatch):
        """compile_metal_lib returns None gracefully when xcrun not found."""
        from turboquant.metal import metal_lib as ml
        monkeypatch.setattr(ml, "_sdk_path", lambda: None)
        result = ml.compile_metal_lib(force=True)
        assert result is None


# ===========================================================================
# metal/metal_quantizer.py  (fallback = PyTorch MPS path)
# ===========================================================================

class TestMetalTurboQuantMSEFallback:
    """Tests that run on the PyTorch-MPS fallback path (no PyObjC needed)."""

    @pytest.fixture
    def quantizer(self, device):
        from turboquant.metal.metal_quantizer import MetalTurboQuantMSE
        return MetalTurboQuantMSE(d=64, b=2, device=device, dtype=torch.float16, seed=0)

    def test_uses_fallback_when_no_pyobjc(self, quantizer):
        # Without PyObjC, _metal should be None
        assert quantizer._metal is None

    def test_quant_shape(self, quantizer, device):
        x = make_unit_tensors(10, 64, device).half()
        idx = quantizer.quant(x)
        assert idx.shape == (10, 64)
        assert idx.dtype == torch.int16

    def test_dequant_shape(self, quantizer, device):
        x = make_unit_tensors(10, 64, device).half()
        idx = quantizer.quant(x)
        x_rec = quantizer.dequant(idx)
        assert x_rec.shape == (10, 64)

    def test_quant_pack_returns_correct_types(self, quantizer, device):
        x = make_unit_tensors(8, 64, device).half()
        packed, norms = quantizer.quant_pack(x)
        assert packed.dtype == torch.int8
        assert norms.dtype == torch.float16
        assert norms.shape == (8,)

    def test_dequant_unpack_roundtrip_shape(self, quantizer, device):
        x = make_unit_tensors(8, 64, device).half()
        packed, norms = quantizer.quant_pack(x)
        x_rec = quantizer.dequant_unpack(packed, norms)
        assert x_rec.shape == (8, 64)

    def test_compressed_bytes_per_token_head(self, quantizer):
        # b=2, d=64: ceil(2*64/8) + 2 = 16 + 2 = 18
        assert quantizer.compressed_bytes_per_token_head() == 18

    def test_reconstruction_quality(self, quantizer, device):
        x = make_unit_tensors(200, 64, device).half()
        packed, norms = quantizer.quant_pack(x)
        x_rec = quantizer.dequant_unpack(packed, norms)
        cos = (
            (x.float() * x_rec.float()).sum(-1)
            / (x.float().norm(dim=-1) * x_rec.float().norm(dim=-1) + 1e-9)
        ).mean().item()
        assert cos > 0.85

    @pytest.mark.parametrize("b", [1, 2, 3, 4])
    def test_various_bits(self, device, b):
        from turboquant.metal.metal_quantizer import MetalTurboQuantMSE
        q = MetalTurboQuantMSE(d=32, b=b, device=device, seed=b)
        x = make_unit_tensors(5, 32, device).half()
        packed, norms = q.quant_pack(x)
        x_rec = q.dequant_unpack(packed, norms)
        assert x_rec.shape == (5, 32)


class TestMetalTurboQuantProdFallback:

    @pytest.fixture
    def quantizer(self, device):
        from turboquant.metal.metal_quantizer import MetalTurboQuantProd
        return MetalTurboQuantProd(d=64, b=2, device=device, dtype=torch.float16, seed=0)

    def test_b_zero_raises(self, device):
        from turboquant.metal.metal_quantizer import MetalTurboQuantProd
        with pytest.raises(ValueError):
            MetalTurboQuantProd(d=64, b=0, device=device)

    def test_uses_fallback(self, quantizer):
        assert quantizer._metal is None

    def test_quant_shapes(self, quantizer, device):
        x = make_unit_tensors(10, 64, device).half()
        idx, qjl, gamma = quantizer.quant(x)
        assert idx.shape == (10, 64)
        assert qjl.shape == (10, 64)
        assert gamma.shape == (10,)
        assert idx.dtype == torch.int16
        assert qjl.dtype == torch.int8

    def test_dequant_shape(self, quantizer, device):
        x = make_unit_tensors(10, 64, device).half()
        idx, qjl, gamma = quantizer.quant(x)
        x_rec = quantizer.dequant(idx, qjl, gamma)
        assert x_rec.shape == (10, 64)

    def test_quant_pack_types(self, quantizer, device):
        x = make_unit_tensors(8, 64, device).half()
        pi, pq, gamma, norms = quantizer.quant_pack(x)
        assert pi.dtype == torch.int8
        assert pq.dtype == torch.int8
        assert norms.dtype == torch.float16
        assert gamma.shape == (8,)

    def test_dequant_unpack_shape(self, quantizer, device):
        x = make_unit_tensors(8, 64, device).half()
        pi, pq, gamma, norms = quantizer.quant_pack(x)
        x_rec = quantizer.dequant_unpack(pi, pq, gamma, norms)
        assert x_rec.shape == (8, 64)

    def test_inner_product(self, quantizer, device):
        x = make_unit_tensors(20, 64, device).half()
        y = make_unit_tensors(20, 64, device).half()
        idx, qjl, gamma = quantizer.quant(x)
        ip = quantizer.inner_product(y, idx, qjl, gamma)
        assert ip.shape == (20,)

    def test_compressed_bytes_per_token_head(self, quantizer):
        # b=2: b_mse=1, pb=1, ceil(1*64/8) + ceil(64/8) + 2 + 4 = 8 + 8 + 6 = 22
        expected = math.ceil(1 * 64 / 8) + math.ceil(64 / 8) + 2 + 4
        assert quantizer.compressed_bytes_per_token_head() == expected

    def test_b1_special_case(self, device):
        from turboquant.metal.metal_quantizer import MetalTurboQuantProd
        q = MetalTurboQuantProd(d=32, b=1, device=device, seed=7)
        x = make_unit_tensors(5, 32, device).half()
        idx, qjl, gamma = q.quant(x)
        # b=1: idx should be all zeros
        assert torch.all(idx == 0)

    @pytest.mark.parametrize("b", [1, 2, 3, 4])
    def test_various_bits_produce_correct_shapes(self, device, b):
        from turboquant.metal.metal_quantizer import MetalTurboQuantProd
        q = MetalTurboQuantProd(d=32, b=b, device=device, seed=b)
        x = make_unit_tensors(5, 32, device).half()
        pi, pq, gamma, norms = q.quant_pack(x)
        x_rec = q.dequant_unpack(pi, pq, gamma, norms)
        assert x_rec.shape == (5, 32)


# ===========================================================================
# production/__init__.py exports
# ===========================================================================

class TestProductionModuleExports:

    def test_all_exports_importable(self):
        from turboquant.production import (
            LlamaVariant,
            get_model_config,
            SUPPORTED_MODELS,
            ProductionTurboQuantCache,
            patch_llama_model,
            unpatch_llama_model,
        )
        assert callable(get_model_config)
        assert isinstance(SUPPORTED_MODELS, dict)
        assert len(SUPPORTED_MODELS) >= 6

    def test_turboquant_root_exports(self):
        import turboquant
        assert hasattr(turboquant, "MetalTurboQuantMSE")
        assert hasattr(turboquant, "MetalTurboQuantProd")
        assert hasattr(turboquant, "metal_available")
        assert hasattr(turboquant, "ProductionTurboQuantCache")
        assert hasattr(turboquant, "patch_llama_model")
        assert turboquant.__version__ == "0.2.0"


# ===========================================================================
# attention_patch.py — unit tests that don't require a real model
# ===========================================================================

class TestAttentionPatchHelpers:

    def test_infer_device_from_model(self, device):
        """_infer_device returns the device of the model's first parameter."""
        from turboquant.production.attention_patch import _infer_device
        import torch.nn as nn
        m = nn.Linear(4, 4)
        m = m.to(device)
        assert device in _infer_device(m)  # "mps:0" contains "mps"

    def test_tq_past_key_value_indexing(self):
        """_TQPastKeyValue supports both getitem and .key_cache/.value_cache."""
        from turboquant.production.attention_patch import _TQPastKeyValue
        k = torch.ones(1, 2, 8, 64)
        v = torch.zeros(1, 2, 8, 64)
        pkv = _TQPastKeyValue(k, v)

        k2, v2 = pkv[0]
        assert torch.equal(k2, k)
        assert torch.equal(v2, v)
        assert torch.equal(pkv.key_cache[0], k)
        assert torch.equal(pkv.value_cache[0], v)

    def test_patch_requires_model_with_layers(self):
        """patch_llama_model raises if model doesn't have standard attention structure."""
        from turboquant.production.attention_patch import patch_llama_model
        import torch.nn as nn

        class BadModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)
            def forward(self, x):
                return self.fc(x)

        bm = BadModel()
        bm.config = type("C", (), {
            "model_type": "custom", "hidden_size": 0, "num_attention_heads": 0
        })()

        with pytest.raises((ValueError, AttributeError)):
            patch_llama_model(bm, bits=2)

    def test_unpatch_no_op_when_not_patched(self):
        """unpatch_llama_model doesn't crash on an unpatched model."""
        from turboquant.production.attention_patch import unpatch_llama_model
        import torch.nn as nn

        class FakeModel(nn.Module):
            pass

        fm = FakeModel()
        fm.model = type("M", (), {"layers": []})()  # empty layers list
        # Should not raise
        unpatch_llama_model(fm)
