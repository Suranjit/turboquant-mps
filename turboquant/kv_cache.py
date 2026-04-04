"""
TurboQuant KV-cache integration for HuggingFace Transformers (>= 4.38).

Provides TurboQuantDynamicCache, a drop-in replacement for DynamicCache that
transparently quantizes each key/value tensor using TurboQuant as it is
written and dequantizes it on read.

Compression reality check (per token, per head, head_dim=d):
  Full FP16:  d * 2  bytes
  TQ_mse @ b bits:  ceil(b*d/8) + 4 bytes (indices + norm scalar)
  TQ_prod @ b bits: ceil(b*d/8) + ceil(d/8) + 8 bytes (idx + QJL bits + 2 scalars)

For d=64, b=2:
  FP16   = 128 bytes
  TQ_mse = 16 + 4 = 20 bytes  → 6.4x
  TQ_prod= 16 + 8 + 8 = 32 bytes → 4x
"""

from __future__ import annotations

import math
import numpy as np
import torch
from typing import Optional, Tuple, Any

from .quantizer import TurboQuantMSE, TurboQuantProd

try:
    from transformers.cache_utils import DynamicLayer, DynamicCache
    _HAS_DYNAMIC_LAYER = True
except ImportError:
    _HAS_DYNAMIC_LAYER = False


# ---------------------------------------------------------------------------
# Per-layer quantized cache
# ---------------------------------------------------------------------------

class TurboQuantLayer:
    """
    A single-layer cache that quantizes key and value states on write and
    dequantizes on read.  Mimics the DynamicLayer interface.
    """

    is_sliding = False
    is_initialized = False

    def __init__(self, key_quant: TurboQuantMSE | TurboQuantProd,
                 val_quant: TurboQuantMSE | TurboQuantProd,
                 mode: str):
        self._kq = key_quant   # quantizer for keys
        self._vq = val_quant   # quantizer for values
        self.mode = mode

        self._key_store: list  = []   # list of per-chunk compressed tuples
        self._val_store: list  = []
        self._seq_len: int = 0

        self.device = None
        self.dtype  = None

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
        """Quantize new states, append to store, return full dequantized cache."""
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        # Compress and store new chunk
        self._key_store.append(self._compress(key_states, self._kq))
        self._val_store.append(self._compress(value_states, self._vq))
        self._seq_len += key_states.shape[-2]

        # Reconstruct and return full accumulated cache
        full_keys = self._decompress_all(self._key_store, self._kq)
        full_vals = self._decompress_all(self._val_store, self._vq)

        return (
            torch.from_numpy(full_keys).to(device=self.device, dtype=self.dtype),
            torch.from_numpy(full_vals).to(device=self.device, dtype=self.dtype),
        )

    def get_seq_length(self) -> int:
        return self._seq_len

    def get_mask_sizes(self, cache_position: torch.Tensor) -> Tuple[int, int]:
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self._seq_len + query_length
        return kv_length, kv_offset

    def get_max_cache_shape(self) -> int:
        return -1  # unlimited

    # ---- Compression helpers ----

    def _compress(self, tensor: torch.Tensor, q):
        """Quantize (B, H, T, d) tensor; return compressed tuple."""
        x = tensor.float().cpu().numpy()    # (B, H, T, d)
        orig_shape = x.shape
        x_flat = x.reshape(-1, q.d)         # (B*H*T, d)

        if self.mode == "mse":
            idx, norms = q.quant_with_norm(x_flat)
            return ("mse", idx, norms, orig_shape)
        else:
            idx, qjl, qjl_gamma, norms = q.quant_with_norm(x_flat)
            return ("prod", idx, qjl, qjl_gamma, norms, orig_shape)

    def _decompress_chunk(self, chunk, q) -> np.ndarray:
        """Dequantize a single compressed chunk back to float32 numpy."""
        if chunk[0] == "mse":
            _, idx, norms, orig_shape = chunk
            x_flat = q.dequant_with_norm(idx, norms)
        else:
            _, idx, qjl, qjl_gamma, norms, orig_shape = chunk
            x_flat = q.dequant_with_norm(idx, qjl, qjl_gamma, norms)
        return x_flat.reshape(orig_shape)

    def _decompress_all(self, store: list, q) -> np.ndarray:
        """Decompress all chunks and concatenate along the T dimension."""
        chunks = [self._decompress_chunk(c, q) for c in store]
        return np.concatenate(chunks, axis=-2)   # concat along seq dim

    # ---- Memory accounting ----

    def compressed_bytes(self) -> int:
        """
        Theoretical compressed size in bytes using optimal bit packing.

        Uses the actual bit-width from the quantizer (self._kq.b / self._vq.b)
        rather than inferring it from index values, which is unreliable when
        all indices happen to be 0.
        """
        total = 0
        for store, q in ((self._key_store, self._kq), (self._val_store, self._vq)):
            b = q.b
            d = q.d
            for chunk in store:
                orig_shape = chunk[-1]
                B, H, T, _ = orig_shape
                if chunk[0] == "mse":
                    # ceil(b*d/8) bytes for packed indices + 4 bytes (float32) per norm
                    total += math.ceil(b * d / 8) * B * H * T
                    total += 4 * B * H * T
                else:
                    # MSE part at (b-1) bits + QJL 1 bit + two float32 scalars
                    b_mse = max(b - 1, 1)
                    total += math.ceil(b_mse * d / 8) * B * H * T   # packed idx
                    total += math.ceil(d / 8) * B * H * T           # QJL bits
                    total += 4 * B * H * T   # qjl_gamma (float32)
                    total += 4 * B * H * T   # norms (float32)
        return total

    def fp16_bytes(self) -> int:
        """What this cache would cost at full FP16 precision."""
        total = 0
        for chunk in self._key_store + self._val_store:
            orig_shape = chunk[-1]
            B, H, T, d = orig_shape
            total += B * H * T * d * 2   # 2 bytes per FP16 element
        return total


# ---------------------------------------------------------------------------
# Full-model TurboQuant cache (all layers)
# ---------------------------------------------------------------------------

class TurboQuantDynamicCache:
    """
    Drop-in replacement for transformers DynamicCache that applies TurboQuant
    quantization to all KV tensors.

    The interface matches DynamicCache from transformers >= 4.38 / 4.57.

    Args:
        head_dim: Dimension of each attention head (d in the paper).
        bits:     Bits per coordinate (b in the paper).
        mode:     "mse" for MSE-optimal, "prod" for unbiased inner-product.
        seed:     RNG seed for reproducibility.
    """

    def __init__(
        self,
        head_dim: int,
        bits: int,
        mode: str = "prod",
        seed: int = 42,
    ):
        if mode not in ("mse", "prod"):
            raise ValueError(f"mode must be 'mse' or 'prod', got '{mode}'")
        self.head_dim = head_dim
        self.bits = bits
        self.mode = mode

        rng = np.random.default_rng(seed)
        self._rng = rng
        self._layers: list[TurboQuantLayer] = []

    def _make_layer(self) -> TurboQuantLayer:
        seed_k = int(self._rng.integers(0, 2**31))
        seed_v = int(self._rng.integers(0, 2**31))
        if self.mode == "mse":
            kq = TurboQuantMSE(self.head_dim, self.bits, seed=seed_k)
            vq = TurboQuantMSE(self.head_dim, self.bits, seed=seed_v)
        else:
            kq = TurboQuantProd(self.head_dim, self.bits, seed=seed_k)
            vq = TurboQuantProd(self.head_dim, self.bits, seed=seed_v)
        return TurboQuantLayer(kq, vq, self.mode)

    # ---- Cache interface ----

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize and append; return full dequantized (key, value) for the layer."""
        while len(self._layers) <= layer_idx:
            self._layers.append(self._make_layer())
        return self._layers[layer_idx].update(key_states, value_states, cache_kwargs)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._layers):
            return 0
        return self._layers[layer_idx].get_seq_length()

    def get_max_length(self) -> Optional[int]:
        return None

    def get_mask_sizes(self, cache_position: "torch.Tensor", layer_idx: int) -> tuple:
        """Return (kv_length, kv_offset) for attention mask construction."""
        if layer_idx >= len(self._layers):
            return cache_position.shape[0], 0
        return self._layers[layer_idx].get_mask_sizes(cache_position)

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
        if cb == 0:
            return 0.0
        return self.fp16_bytes() / cb

    def memory_summary(self) -> str:
        cb = self.compressed_bytes()
        fp = self.fp16_bytes()
        ratio = self.compression_ratio()
        return (
            f"Compressed: {cb/1024:.1f} KB  |  "
            f"FP16 equiv: {fp/1024:.1f} KB  |  "
            f"Ratio: {ratio:.2f}x"
        )
