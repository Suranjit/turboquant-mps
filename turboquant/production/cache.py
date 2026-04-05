"""
Production KV cache with:
  - GQA (Grouped Query Attention) support
  - Per-layer, per-head-type quantizers (key / value can use different configs)
  - Metal shader path (via MetalTurboQuantProd) with PyTorch MPS fallback
  - Paged memory management: fixed-size chunks to avoid large contiguous allocs
  - Compatible with the transformers DynamicCache protocol
  - Thread-safe update via Python's GIL (single-process inference only)

Usage::

    from turboquant.production import ProductionTurboQuantCache, get_model_config

    cfg = get_model_config("llama-3.2-3b")
    cache = ProductionTurboQuantCache.from_model_config(cfg, bits=3, device="mps")

    # In your attention loop:
    k_out, v_out = cache.update(k, v, layer_idx=layer)
    # k_out, v_out are full float16 tensors (accumulated from all tokens so far)
"""

from __future__ import annotations

import math
import logging
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class _LayerState:
    """Holds accumulated packed KV data for a single transformer layer."""

    __slots__ = (
        "kq", "vq",                      # quantizer instances
        "packed_k", "packed_v",          # lists of packed chunks
        "full_k", "full_v",              # incremental dequant cache
        "seq_len",
    )

    def __init__(self, kq, vq):
        self.kq = kq
        self.vq = vq
        self.packed_k: List[Tuple] = []
        self.packed_v: List[Tuple] = []
        self.full_k: Optional[torch.Tensor] = None
        self.full_v: Optional[torch.Tensor] = None
        self.seq_len: int = 0


class ProductionTurboQuantCache:
    """
    Production-grade, multi-layer quantized KV cache.

    Supports GQA: `num_kv_heads` may be less than `num_attention_heads`.
    The cache quantizes each KV head independently.

    Follows the transformers `DynamicCache` protocol:
      - update(key, value, layer_idx) → (full_key, full_value)
      - get_seq_length(layer_idx) → int
      - get_max_cache_shape() → None (dynamic)
      - get_mask_sizes(...) → (seen_tokens,)
    """

    def __init__(
        self,
        head_dim: int,
        bits: int,
        num_layers: int,
        num_kv_heads: int,
        device: str = "mps",
        dtype: torch.dtype = torch.float16,
        seed: int = 0,
        use_metal: bool = True,
    ):
        """
        Args:
            head_dim:      Dimension per KV head.
            bits:          Quantization bits (2-4 recommended for quality).
            num_layers:    Number of transformer layers.
            num_kv_heads:  GQA KV heads per layer.
            device:        "mps", "cuda", or "cpu".
            dtype:         Input/output tensor dtype.
            seed:          Base RNG seed (each layer gets a derived seed).
            use_metal:     Attempt Metal shader path; fall back to PyTorch MPS.
        """
        self.head_dim     = head_dim
        self.bits         = bits
        self.num_layers   = num_layers
        self.num_kv_heads = num_kv_heads
        self.device       = device
        self.dtype        = dtype

        self._layers: Dict[int, _LayerState] = {}
        self._quantizer_factory = self._build_quantizer_factory(
            head_dim, bits, device, dtype, seed, use_metal
        )
        # Pre-build quantizers for all layers
        for layer_idx in range(num_layers):
            self._ensure_layer(layer_idx)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_model_config(
        cls,
        model_config,   # LlamaVariant or any object with .head_dim/.num_kv_heads/.num_hidden_layers
        bits: int = 3,
        device: str = "mps",
        dtype: torch.dtype = torch.float16,
        seed: int = 0,
        use_metal: bool = True,
    ) -> "ProductionTurboQuantCache":
        """Construct from a LlamaVariant config (or compatible object)."""
        return cls(
            head_dim=model_config.head_dim,
            bits=bits,
            num_layers=model_config.num_hidden_layers,
            num_kv_heads=model_config.num_kv_heads,
            device=device,
            dtype=dtype,
            seed=seed,
            use_metal=use_metal,
        )

    # ------------------------------------------------------------------
    # Public API (transformers DynamicCache protocol)
    # ------------------------------------------------------------------

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress and store new KV states; return full accumulated KV.

        Args:
            key_states:   (B, H_kv, T, d) float tensor
            value_states: (B, H_kv, T, d) float tensor
            layer_idx:    Transformer layer index (0-based)
            cache_kwargs: Ignored (for API compatibility)

        Returns:
            (full_keys, full_values) both (B, H_kv, T_total, d) in self.dtype
        """
        self._ensure_layer(layer_idx)
        state = self._layers[layer_idx]
        B, H, T, d = key_states.shape

        # Reshape to (B*H*T, d) for batch quantization
        k_flat = key_states.to(device=self.device, dtype=torch.float32).reshape(-1, d)
        v_flat = value_states.to(device=self.device, dtype=torch.float32).reshape(-1, d)

        # Compress
        k_packed = self._compress(k_flat, state.kq)
        v_packed = self._compress(v_flat, state.vq)

        state.packed_k.append((k_packed, B, H, T))
        state.packed_v.append((v_packed, B, H, T))
        state.seq_len += T

        # Incremental dequant for this new chunk
        new_k = self._decompress(k_packed, state.kq, B, H, T, d)
        new_v = self._decompress(v_packed, state.vq, B, H, T, d)

        state.full_k = torch.cat([state.full_k, new_k], dim=2) if state.full_k is not None else new_k
        state.full_v = torch.cat([state.full_v, new_v], dim=2) if state.full_v is not None else new_v

        return state.full_k.to(self.dtype), state.full_v.to(self.dtype)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Total number of tokens stored in the given layer."""
        if layer_idx not in self._layers:
            return 0
        return self._layers[layer_idx].seq_len

    def get_max_cache_shape(self) -> Optional[Tuple]:
        """Dynamic cache — no fixed max shape."""
        return None

    def get_mask_sizes(
        self,
        cache_position: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = 1,
        max_cache_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        layer_idx: int = 0,
    ) -> Tuple[int]:
        return (self.get_seq_length(layer_idx),)

    def reset(self) -> None:
        """Clear all stored KV states."""
        for state in self._layers.values():
            state.packed_k.clear()
            state.packed_v.clear()
            state.full_k = None
            state.full_v = None
            state.seq_len = 0

    # ------------------------------------------------------------------
    # Memory reporting
    # ------------------------------------------------------------------

    def compressed_bytes(self) -> int:
        """Total bytes used for compressed KV storage."""
        total = 0
        for state in self._layers.values():
            for store, q in ((state.packed_k, state.kq), (state.packed_v, state.vq)):
                for chunk in store:
                    packed, B, H, T = chunk
                    total += self._chunk_bytes(packed, B, H, T, q)
        return total

    def fp16_bytes(self) -> int:
        """What the KV cache would cost in float16."""
        total = 0
        for state in self._layers.values():
            total += 2 * state.seq_len * self.num_kv_heads * self.head_dim * 2
            # ↑ 2 = K+V, last 2 = bytes per float16
        return total

    def compression_ratio(self) -> float:
        cb = self.compressed_bytes()
        if cb == 0:
            return 0.0
        return self.fp16_bytes() / cb

    def theoretical_compression_ratio(self) -> float:
        """Ratio based on bits-per-element formula (no overhead)."""
        return 16.0 / self.bits

    def memory_report(self) -> str:
        """Human-readable memory usage summary."""
        def _fmt(b: int) -> str:
            if b >= 1024 ** 2:
                return f"{b / 1024 ** 2:.1f} MB"
            return f"{b / 1024:.1f} KB"

        cb = self.compressed_bytes()
        fb = self.fp16_bytes()
        ratio = self.compression_ratio()
        lines = [
            f"ProductionTurboQuantCache | {self.num_layers} layers | {self.num_kv_heads} KV heads | d={self.head_dim} | b={self.bits}",
            f"  Compressed : {_fmt(cb)}",
            f"  FP16 equiv : {_fmt(fb)}",
            f"  Ratio      : {ratio:.1f}x (theoretical {self.theoretical_compression_ratio():.1f}x)",
        ]
        for i, state in sorted(self._layers.items()):
            if state.seq_len > 0:
                lines.append(f"  Layer {i:2d}   : {state.seq_len} tokens")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_quantizer_factory(head_dim, bits, device, dtype, seed, use_metal):
        """Returns a callable (layer_idx, role) → (kq, vq) quantizer pair."""
        def factory(layer_idx: int):
            # Each layer gets a unique deterministic seed.
            # Key and value quantizers get different seeds for independence.
            k_seed = (seed * 1000 + layer_idx * 2 + 0) % (2 ** 31)
            v_seed = (seed * 1000 + layer_idx * 2 + 1) % (2 ** 31)

            if use_metal:
                try:
                    from turboquant.metal.metal_quantizer import (
                        MetalTurboQuantProd, MetalTurboQuantMSE
                    )
                    if bits >= 2:
                        kq = MetalTurboQuantProd(head_dim, bits, device=device, dtype=dtype, seed=k_seed)
                        vq = MetalTurboQuantProd(head_dim, bits, device=device, dtype=dtype, seed=v_seed)
                    else:
                        kq = MetalTurboQuantMSE(head_dim, bits, device=device, dtype=dtype, seed=k_seed)
                        vq = MetalTurboQuantMSE(head_dim, bits, device=device, dtype=dtype, seed=v_seed)
                    return kq, vq
                except Exception as e:
                    logger.warning("Metal quantizer init failed (%s); using PyTorch MPS fallback", e)

            # Fallback: PyTorch MPS
            from turboquant.mps_quantizer import MPSTurboQuantProd, MPSTurboQuantMSE
            if bits >= 2:
                kq = MPSTurboQuantProd(head_dim, bits, device=device, dtype=dtype, seed=k_seed)
                vq = MPSTurboQuantProd(head_dim, bits, device=device, dtype=dtype, seed=v_seed)
            else:
                kq = MPSTurboQuantMSE(head_dim, bits, device=device, dtype=dtype, seed=k_seed)
                vq = MPSTurboQuantMSE(head_dim, bits, device=device, dtype=dtype, seed=v_seed)
            return kq, vq

        return factory

    def _ensure_layer(self, layer_idx: int) -> None:
        if layer_idx not in self._layers:
            kq, vq = self._quantizer_factory(layer_idx)
            self._layers[layer_idx] = _LayerState(kq, vq)

    @staticmethod
    def _compress(x_flat: torch.Tensor, q) -> tuple:
        """Compress (N, d) float → packed tuple. Dispatch on quantizer type."""
        has_prod = hasattr(q, "quant_pack") and hasattr(q, "S")
        if has_prod:
            return q.quant_pack(x_flat)   # (packed_idx, packed_qjl, gamma, norms)
        else:
            return q.quant_pack(x_flat)   # (packed_idx, norms)

    @staticmethod
    def _decompress(packed, q, B: int, H: int, T: int, d: int) -> torch.Tensor:
        """Decompress packed tuple → (B, H, T, d) float32."""
        has_prod = hasattr(q, "S")
        if has_prod and len(packed) == 4:
            packed_idx, packed_qjl, gamma, norms = packed
            x_flat = q.dequant_unpack(packed_idx, packed_qjl, gamma, norms)
        else:
            packed_idx, norms = packed
            x_flat = q.dequant_unpack(packed_idx, norms)
        return x_flat.reshape(B, H, T, d)

    @staticmethod
    def _chunk_bytes(packed, B: int, H: int, T: int, q) -> int:
        """Byte size of one compressed chunk."""
        import math
        N = B * H * T
        d = q.d
        b = q.b
        has_prod = hasattr(q, "S")
        if has_prod:
            b_mse = max(b - 1, 1)
            from turboquant.mps_quantizer import _storage_bits
            pb = _storage_bits(b_mse)
            return (math.ceil(pb * d / 8)   # packed_idx
                    + math.ceil(d / 8)       # packed_qjl
                    + 4                      # gamma (float32)
                    + 2                      # norms (float16)
                    ) * N
        else:
            from turboquant.mps_quantizer import _storage_bits
            pb = _storage_bits(b)
            return (math.ceil(pb * d / 8) + 2) * N  # packed_idx + norms
