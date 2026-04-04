"""
TurboQuant KV-cache integration for HuggingFace Transformers.

Drop-in replacement for DynamicCache that transparently quantizes each
key/value tensor as it is written and dequantizes on read.

Usage:
    from turboquant.kv_cache import TurboQuantKVCache
    cache = TurboQuantKVCache(head_dim=128, bits=2, mode="prod")
    outputs = model.generate(..., past_key_values=cache, ...)
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Optional, Tuple, List

from .quantizer import TurboQuantMSE, TurboQuantProd


class TurboQuantKVCache:
    """
    A HuggingFace-compatible KV cache that applies TurboQuant quantization.

    Stores:
      - For "mse"  mode: (idx, norm)   per key/value layer
      - For "prod" mode: (idx, qjl, qjl_gamma, norm)  per key/value layer

    The cache is a list of (key_state, value_state) tuples indexed by layer,
    following the HuggingFace DynamicCache interface used in transformers >= 4.38.
    """

    def __init__(
        self,
        head_dim: int,
        bits: int = 2,
        mode: str = "prod",
        seed: int = 42,
    ):
        """
        Args:
            head_dim: Dimension of each attention head (d in the paper).
            bits:     Bits per coordinate for quantization (b in the paper).
            mode:     "mse" for TurboQuantMSE, "prod" for TurboQuantProd.
            seed:     RNG seed for reproducibility.
        """
        if mode not in ("mse", "prod"):
            raise ValueError(f"mode must be 'mse' or 'prod', got '{mode}'")
        self.head_dim = head_dim
        self.bits = bits
        self.mode = mode

        if mode == "mse":
            self._quant = TurboQuantMSE(head_dim, bits, seed=seed)
        else:
            self._quant = TurboQuantProd(head_dim, bits, seed=seed)

        # Internal storage: list of dicts, one per layer
        # Each dict has keys "key" and "value", each holding a compressed tuple.
        self._key_cache: list[list] = []
        self._value_cache: list[list] = []

        # Keep track of how many tokens are cached per layer
        self._seen_tokens: int = 0

    # ------------------------------------------------------------------
    # HuggingFace Cache interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._key_cache)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if len(self._key_cache) <= layer_idx:
            return 0
        return self._key_cache[layer_idx][0].shape[1] if self.mode == "mse" else \
               self._key_cache[layer_idx][0].shape[1]

    def get_max_length(self) -> Optional[int]:
        return None  # unlimited

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append key/value states for layer_idx, quantize them, and return the
        full (accumulated) dequantized key/value tensors.

        key_states / value_states shape: (batch, num_heads, seq_len, head_dim)
        """
        device = key_states.device
        dtype = key_states.dtype

        # Extend storage list if needed
        while len(self._key_cache) <= layer_idx:
            self._key_cache.append(None)
            self._value_cache.append(None)

        # Quantize new tokens
        key_q = self._compress(key_states)    # quantized representation
        val_q = self._compress(value_states)

        # Accumulate
        if self._key_cache[layer_idx] is None:
            self._key_cache[layer_idx] = key_q
            self._value_cache[layer_idx] = val_q
        else:
            self._key_cache[layer_idx] = self._concat(
                self._key_cache[layer_idx], key_q
            )
            self._value_cache[layer_idx] = self._concat(
                self._value_cache[layer_idx], val_q
            )

        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Dequantize the full cache and return as torch tensors
        full_keys = self._decompress(self._key_cache[layer_idx], device, dtype)
        full_vals = self._decompress(self._value_cache[layer_idx], device, dtype)
        return full_keys, full_vals

    # ------------------------------------------------------------------
    # Compression helpers
    # ------------------------------------------------------------------

    def _compress(self, tensor: torch.Tensor):
        """
        Quantize a (batch, heads, seq, head_dim) tensor.
        Returns a tuple of numpy arrays (compressed representation).
        """
        # Move to CPU numpy for TurboQuant (no GPU dependency needed)
        x = tensor.float().cpu().numpy()           # (B, H, T, d)
        orig_shape = x.shape
        # Flatten to (B*H*T, d) for vectorised quantization
        x_flat = x.reshape(-1, self.head_dim)

        if self.mode == "mse":
            idx, norms = self._quant.quant_with_norm(x_flat)
            # Reshape back
            idx = idx.reshape(*orig_shape[:-1], self.head_dim)
            norms = norms.reshape(*orig_shape[:-1])
            return (idx, norms, orig_shape)
        else:
            idx, qjl, qjl_gamma, norms = self._quant.quant_with_norm(x_flat)
            idx = idx.reshape(*orig_shape[:-1], self.head_dim)
            qjl = qjl.reshape(*orig_shape[:-1], self.head_dim)
            qjl_gamma = qjl_gamma.reshape(*orig_shape[:-1])
            norms = norms.reshape(*orig_shape[:-1])
            return (idx, qjl, qjl_gamma, norms, orig_shape)

    def _decompress(self, compressed, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Dequantize and return a torch tensor on the given device."""
        if self.mode == "mse":
            idx, norms, orig_shape = compressed
            idx_flat = idx.reshape(-1, self.head_dim)
            norms_flat = norms.reshape(-1)
            x_flat = self._quant.dequant_with_norm(idx_flat, norms_flat)
            x = x_flat.reshape(orig_shape)
        else:
            idx, qjl, qjl_gamma, norms, orig_shape = compressed
            idx_flat = idx.reshape(-1, self.head_dim)
            qjl_flat = qjl.reshape(-1, self.head_dim)
            qjl_gamma_flat = qjl_gamma.reshape(-1)
            norms_flat = norms.reshape(-1)
            x_flat = self._quant.dequant_with_norm(idx_flat, qjl_flat, qjl_gamma_flat, norms_flat)
            x = x_flat.reshape(orig_shape)

        return torch.from_numpy(x).to(device=device, dtype=dtype)

    def _concat(self, existing, new_compressed):
        """Concatenate two compressed cache entries along the seq_len dimension (axis -2 / axis 2)."""
        if self.mode == "mse":
            idx1, norms1, shape1 = existing
            idx2, norms2, shape2 = new_compressed
            idx = np.concatenate([idx1, idx2], axis=-2)
            norms = np.concatenate([norms1, norms2], axis=-1)
            combined_shape = (*shape1[:-2], shape1[-2] + shape2[-2], shape1[-1])
            return (idx, norms, combined_shape)
        else:
            idx1, qjl1, g1, norms1, shape1 = existing
            idx2, qjl2, g2, norms2, shape2 = new_compressed
            idx = np.concatenate([idx1, idx2], axis=-2)
            qjl = np.concatenate([qjl1, qjl2], axis=-2)
            g = np.concatenate([g1, g2], axis=-1)
            norms = np.concatenate([norms1, norms2], axis=-1)
            combined_shape = (*shape1[:-2], shape1[-2] + shape2[-2], shape1[-1])
            return (idx, qjl, g, norms, combined_shape)

    # ------------------------------------------------------------------
    # Memory reporting
    # ------------------------------------------------------------------

    def memory_bytes(self) -> dict[str, int]:
        """Report compressed memory usage vs full FP16 equivalent."""
        if not self._key_cache or self._key_cache[0] is None:
            return {"compressed": 0, "fp16_equivalent": 0}

        compressed = 0
        for layer_k, layer_v in zip(self._key_cache, self._value_cache):
            for storage in (layer_k, layer_v):
                for arr in storage[:-1]:  # skip orig_shape tuple
                    if isinstance(arr, np.ndarray):
                        compressed += arr.nbytes

        # FP16 equivalent size
        n_layers = len(self._key_cache)
        # Reconstruct total token count from stored shape
        shape = self._key_cache[0][-1]          # orig_shape
        B, H, T, d = shape
        fp16 = n_layers * 2 * B * H * T * d * 2  # 2 for K+V, 2 bytes for FP16

        return {"compressed_bytes": compressed, "fp16_equivalent_bytes": fp16}

    def compression_ratio(self) -> float:
        """Ratio of FP16 size to compressed size (higher = better compression)."""
        mem = self.memory_bytes()
        if mem["compressed_bytes"] == 0:
            return 0.0
        return mem["fp16_equivalent_bytes"] / mem["compressed_bytes"]
