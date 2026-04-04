"""
MPS-native KV cache using TurboQuant bit-packed quantization.

All quantization computation runs on-device (MPS/CUDA/CPU).
Compressed state is stored as compact int16 tensors (bit-packed),
giving 4-5× memory reduction for b=2 bits at typical head dimensions.

Compressed layout per layer, per (B, H) pair, accumulated over T tokens:
  packed_idx:  (B, H, T, n_idx_bytes)  int16  — MSE index bits
  packed_qjl:  (B, H, T, n_qjl_bytes)  int16  — QJL sign bits
  qjl_gamma:   (B, H, T)               float16 — residual norms
  vec_norm:    (B, H, T)               float16 — original vector norms
"""

from __future__ import annotations

import math
import torch
from typing import Optional, Tuple

from .mps_quantizer import MPSTurboQuantProd


class MPSTurboQuantLayer:
    """
    Single-layer MPS-native quantized KV cache.
    Implements the same interface as transformers' DynamicLayer.
    """

    is_sliding = False

    def __init__(self, key_quant: MPSTurboQuantProd, val_quant: MPSTurboQuantProd):
        self._kq = key_quant
        self._vq = val_quant
        self.device = key_quant.device
        self.dtype  = key_quant.dtype
        self.d      = key_quant.d

        # Accumulated compressed state — each is a list of per-chunk tensors
        # that get concatenated along the T (seq) dimension
        self._k_idx:   list[torch.Tensor] = []
        self._k_qjl:   list[torch.Tensor] = []
        self._k_gamma: list[torch.Tensor] = []
        self._k_norm:  list[torch.Tensor] = []

        self._v_idx:   list[torch.Tensor] = []
        self._v_qjl:   list[torch.Tensor] = []
        self._v_gamma: list[torch.Tensor] = []
        self._v_norm:  list[torch.Tensor] = []

        # Incremental reconstruction cache: avoids re-decompressing all prior
        # tokens on every update step (O(T²) → O(T) total decompression work).
        self._full_k: Optional[torch.Tensor] = None
        self._full_v: Optional[torch.Tensor] = None

        self._seq_len: int = 0
        self.is_initialized = False

    # ---- DynamicLayer interface ----

    def lazy_initialization(self, key_states: torch.Tensor):
        self.device = key_states.device
        self.dtype  = key_states.dtype
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize and append new key/value states.
        Returns the full dequantized (key, value) for this layer.
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        # Reshape (B, H, T, d) → (B*H*T, d) for batch quantization
        B, H, T, d = key_states.shape
        k_flat = key_states.reshape(-1, d).to(self.dtype)
        v_flat = value_states.reshape(-1, d).to(self.dtype)

        # Quantize and pack on device
        ki, kq, kg, kn = self._kq.quant_pack(k_flat)   # compressed keys
        vi, vq, vg, vn = self._vq.quant_pack(v_flat)   # compressed values

        # Reshape back to (B, H, T, n_bytes) for storage
        n_idx = ki.shape[-1]
        n_qjl = kq.shape[-1]

        self._k_idx.append(ki.reshape(B, H, T, n_idx))
        self._k_qjl.append(kq.reshape(B, H, T, n_qjl))
        self._k_gamma.append(kg.reshape(B, H, T))
        self._k_norm.append(kn.reshape(B, H, T))

        n_idx_v = vi.shape[-1]
        n_qjl_v = vq.shape[-1]
        self._v_idx.append(vi.reshape(B, H, T, n_idx_v))
        self._v_qjl.append(vq.reshape(B, H, T, n_qjl_v))
        self._v_gamma.append(vg.reshape(B, H, T))
        self._v_norm.append(vn.reshape(B, H, T))

        self._seq_len += T

        # Decompress only the new chunk, then cat onto the cached reconstruction.
        # This reduces total decompression work from O(T²) to O(T).
        new_k = self._dequant_chunk(ki, kq, kg, kn, self._kq, B, H, T)
        new_v = self._dequant_chunk(vi, vq, vg, vn, self._vq, B, H, T)

        self._full_k = (
            torch.cat([self._full_k, new_k], dim=2)
            if self._full_k is not None else new_k
        )
        self._full_v = (
            torch.cat([self._full_v, new_v], dim=2)
            if self._full_v is not None else new_v
        )
        return self._full_k.to(key_states.dtype), self._full_v.to(value_states.dtype)

    def get_seq_length(self) -> int:
        return self._seq_len

    def get_mask_sizes(self, cache_position: torch.Tensor) -> Tuple[int, int]:
        query_length = cache_position.shape[0]
        kv_length = self._seq_len + query_length
        return kv_length, 0

    def get_max_cache_shape(self) -> int:
        return -1

    # ---- Reconstruction ----

    def _dequant_chunk(
        self,
        idx: torch.Tensor,
        qjl: torch.Tensor,
        gamma: torch.Tensor,
        norm: torch.Tensor,
        q,
        B: int,
        H: int,
        T: int,
    ) -> torch.Tensor:
        """Dequantize a single packed chunk; returns (B, H, T, d)."""
        flat_idx   = idx.reshape(-1, idx.shape[-1])
        flat_qjl   = qjl.reshape(-1, qjl.shape[-1])
        flat_gamma = gamma.reshape(-1)
        flat_norm  = norm.reshape(-1)
        x_flat = q.dequant_unpack(flat_idx, flat_qjl, flat_gamma, flat_norm)
        return x_flat.reshape(B, H, T, q.d)

    # ---- Memory accounting ----

    def compressed_bytes(self) -> int:
        """Actual bytes used by compressed storage tensors."""
        total = 0
        for lst in (
            self._k_idx, self._k_qjl, self._k_gamma, self._k_norm,
            self._v_idx, self._v_qjl, self._v_gamma, self._v_norm,
        ):
            for t in lst:
                total += t.element_size() * t.numel()
        return total

    def fp16_bytes(self) -> int:
        """What this layer would cost at full FP16."""
        if self._seq_len == 0:
            return 0
        # Reconstruct from stored shape of first chunk
        B, H = self._k_idx[0].shape[:2]
        return 2 * B * H * self._seq_len * self.d * 2   # 2=K+V, 2=bytes/FP16


# ---------------------------------------------------------------------------
# Full-model MPS-native TurboQuant cache
# ---------------------------------------------------------------------------

class MPSTurboQuantCache:
    """
    Drop-in replacement for transformers DynamicCache using MPS-native
    TurboQuant quantization with bit-packed storage.

    Usage:
        cache = MPSTurboQuantCache(head_dim=64, bits=2, device="mps")
        # Pass to model.generate() or model.forward()
        outputs = model.generate(..., past_key_values=cache, use_cache=True)
    """

    def __init__(
        self,
        head_dim: int,
        bits: int,
        device: str | torch.device = "mps",
        dtype: torch.dtype = torch.float16,
        mode: str = "prod",
        seed: int = 42,
    ):
        """
        Args:
            head_dim: Attention head dimension (d in the paper).
            bits:     Bits per coordinate (b in the paper). Use 2 for 4-5× compression.
            device:   Torch device ("mps", "cuda", or "cpu").
            dtype:    Working dtype for attention (float16 recommended for MPS).
            mode:     "prod" for unbiased inner-product (recommended).
            seed:     RNG seed for rotation/projection matrices.
        """
        self.head_dim = head_dim
        self.bits = bits
        self.device = torch.device(device)
        self.dtype = dtype
        self.mode = mode

        self._layers: list[MPSTurboQuantLayer] = []
        self._rng_seed = seed
        self._layer_seed_offset = 0

        # Pre-compute one quantizer pair to expose memory stats
        self._example_q = MPSTurboQuantProd(head_dim, bits, device=device, dtype=dtype, seed=seed)

    def _make_layer(self) -> MPSTurboQuantLayer:
        s = self._rng_seed + self._layer_seed_offset
        self._layer_seed_offset += 2
        kq = MPSTurboQuantProd(self.head_dim, self.bits, self.device, self.dtype, seed=s)
        vq = MPSTurboQuantProd(self.head_dim, self.bits, self.device, self.dtype, seed=s + 1)
        return MPSTurboQuantLayer(kq, vq)

    # ---- Cache interface (matches transformers DynamicCache) ----

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        while len(self._layers) <= layer_idx:
            self._layers.append(self._make_layer())
        return self._layers[layer_idx].update(key_states, value_states, cache_kwargs)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._layers):
            return 0
        return self._layers[layer_idx].get_seq_length()

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> Tuple[int, int]:
        if layer_idx >= len(self._layers):
            return cache_position.shape[0], 0
        return self._layers[layer_idx].get_mask_sizes(cache_position)

    def get_max_length(self) -> Optional[int]:
        return None

    @property
    def seen_tokens(self) -> int:
        return self.get_seq_length(0)

    def __len__(self) -> int:
        return len(self._layers)

    # ---- Memory reporting ----

    def compressed_bytes(self) -> int:
        return sum(layer.compressed_bytes() for layer in self._layers)

    def fp16_bytes(self) -> int:
        return sum(layer.fp16_bytes() for layer in self._layers)

    def compression_ratio(self) -> float:
        cb = self.compressed_bytes()
        return self.fp16_bytes() / cb if cb > 0 else 0.0

    def theoretical_compression_ratio(self) -> float:
        return self._example_q.compression_ratio()

    def memory_report(self) -> str:
        cb = self.compressed_bytes()
        fp = self.fp16_bytes()
        actual = fp / cb if cb > 0 else 0.0
        theory = self.theoretical_compression_ratio()
        return (
            f"Compressed: {cb/1024**2:.1f} MB  |  "
            f"FP16 equiv: {fp/1024**2:.1f} MB  |  "
            f"Actual {actual:.1f}× (theoretical {theory:.1f}×)"
        )
