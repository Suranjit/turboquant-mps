"""
MPS-native TurboQuant quantizers.

All operations stay on the device (MPS/CUDA/CPU) using pure PyTorch:
  - Rotation / projection: torch.matmul (MPS-accelerated)
  - Centroid lookup: broadcast subtract + argmin (MPS-accelerated)
  - QJL sign: torch.sign (MPS-accelerated)
  - Bit packing for compact storage: integer bitwise ops

No numpy or CPU round-trips during quantization.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple

from .codebook import get_codebook


# ---------------------------------------------------------------------------
# Bit packing utilities (stay on device)
# ---------------------------------------------------------------------------

def _storage_bits(b: int) -> int:
    """
    Smallest valid packing width (1, 2, or 4) that can hold 2^b distinct values.
    For b > 4, we store one value per byte (return 8 → caller uses int8 directly).
    """
    if b <= 1: return 1
    if b <= 2: return 2
    if b <= 4: return 4
    return 8   # no sub-byte packing; caller stores as int8


def pack_bits(tensor: torch.Tensor, n_bits: int) -> torch.Tensor:
    """
    Pack integer values (range [0, 2^n_bits - 1]) in the last dimension into
    1-byte (int8) storage, with n_bits bits per value.
    n_bits must divide 8 (i.e. 1, 2, or 4).

    Args:
        tensor: (..., d) integer tensor with values in [0, 2^n_bits - 1].
        n_bits: Bits per value (1, 2, or 4).

    Returns:
        (..., ceil(d / (8//n_bits))) int8 tensor — 1 byte per packed word.
    """
    if n_bits not in (1, 2, 4):
        raise ValueError(f"n_bits must be 1, 2, or 4; got {n_bits}.")
    vpb = 8 // n_bits       # values per byte
    d = tensor.shape[-1]
    prefix = tensor.shape[:-1]

    pad = (-d) % vpb
    if pad:
        tensor = F.pad(tensor, (0, pad), value=0)

    # Work in int32 for full MPS bitwise support
    tensor = tensor.reshape(*prefix, -1, vpb).to(torch.int32)
    mask   = (1 << n_bits) - 1
    shifts = torch.arange(vpb, device=tensor.device, dtype=torch.int32) * n_bits

    packed = (tensor.bitwise_and(mask) << shifts).sum(dim=-1)  # (..., n_bytes) int32
    return packed.to(torch.int8)   # 1 byte per packed word


def unpack_bits(packed: torch.Tensor, n_bits: int, d: int) -> torch.Tensor:
    """
    Unpack int8 storage (written by pack_bits) back into integer values.

    Args:
        packed: (..., n_bytes) int8 tensor.
        n_bits: Bits per value (1, 2, or 4).
        d:      Original number of values (to trim padding).

    Returns:
        (..., d) int16 tensor with values in [0, 2^n_bits - 1].
    """
    vpb = 8 // n_bits
    mask = (1 << n_bits) - 1
    shifts = torch.arange(vpb, device=packed.device, dtype=torch.int32) * n_bits
    # int8 → int32, then mask away the sign extension with 0xFF
    p32 = packed.to(torch.int32).bitwise_and(0xFF)   # treat as unsigned byte
    result = (p32.unsqueeze(-1) >> shifts).bitwise_and(mask)   # (..., n_bytes, vpb)
    return result.reshape(*packed.shape[:-1], -1)[..., :d].to(torch.int16)


def pack_qjl(qjl: torch.Tensor) -> torch.Tensor:
    """
    Pack {-1, +1} sign bits: 1 bit per value, 8 values per int8 byte.
    +1 → bit=1, -1 → bit=0.

    Args:
        qjl: (..., d) int8 tensor of {-1, +1}.

    Returns:
        (..., ceil(d/8)) int8 tensor.
    """
    bits = (qjl > 0).to(torch.int32)
    return pack_bits(bits, n_bits=1)


def unpack_qjl(packed: torch.Tensor, d: int) -> torch.Tensor:
    """
    Unpack 1-bit storage back to {-1, +1} int8.

    Args:
        packed: (..., ceil(d/8)) int8 tensor.
        d:      Original number of sign values.

    Returns:
        (..., d) int8 tensor of {-1, +1}.
    """
    bits = unpack_bits(packed, n_bits=1, d=d)        # (..., d) values in {0,1}
    return (bits.to(torch.int8) * 2 - 1)             # → {-1,+1}


# ---------------------------------------------------------------------------
# MPS-native MSE quantizer
# ---------------------------------------------------------------------------

class MPSTurboQuantMSE:
    """
    TurboQuant_mse implemented with pure PyTorch operations.
    Runs on any torch device (MPS / CUDA / CPU) with no numpy round-trips
    during quantization.

    Memory layout of compressed KV state per token per head:
      - indices (b bits each, packed): ceil(b*d / 8) bytes
      - norm scalar (float16):          2 bytes
    Total at b=2, d=64: 16 + 2 = 18 bytes vs FP16's 128 bytes → 7× saving
    """

    def __init__(
        self,
        d: int,
        b: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float16,
        seed: int | None = None,
    ):
        self.d = d
        self.b = b
        self.device = torch.device(device)
        self.dtype = dtype

        rng = np.random.default_rng(seed)
        rot_np, _ = np.linalg.qr(rng.standard_normal((d, d)).astype(np.float32))
        self.rotation = torch.from_numpy(rot_np).to(dtype=dtype, device=self.device)

        cb = get_codebook(d, b).astype(np.float32)
        self.codebook = torch.from_numpy(cb).to(dtype=dtype, device=self.device)
        self._pack_bits = _storage_bits(b)   # 1, 2, 4, or 8

    # ---- Quantization ----

    @torch.no_grad()
    def quant(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., d)  →  idx: (..., d) int16  (values in [0, 2^b - 1])
        """
        y = x.to(self.dtype) @ self.rotation.T           # (..., d)
        # Nearest centroid: (..., d, 1) vs (2^b,)
        dists = (y.unsqueeze(-1) - self.codebook).abs()  # (..., d, 2^b)
        return dists.argmin(dim=-1).to(torch.int16)      # (..., d)

    @torch.no_grad()
    def dequant(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (..., d) int16  →  x̃: (..., d)"""
        ỹ = self.codebook[idx.long()]                    # (..., d)
        return (ỹ @ self.rotation).to(self.dtype)        # (..., d)

    @torch.no_grad()
    def quant_pack(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Normalise, quantize, and pack indices.

        Returns:
            packed_idx: (..., ceil(b*d/8)) int8  — bit-packed indices
            norms:      (...,) float16
        """
        norms = x.norm(dim=-1).clamp(min=1e-12)         # (...,)
        idx = self.quant(x / norms.unsqueeze(-1))        # (..., d)
        if self._pack_bits == 8:
            packed = idx.to(torch.int8)                  # (..., d) — 1 byte each
        else:
            packed = pack_bits(idx, self._pack_bits)     # (..., n_bytes)
        return packed.to(self.device), norms.to(torch.float16)

    @torch.no_grad()
    def dequant_unpack(
        self, packed: torch.Tensor, norms: torch.Tensor
    ) -> torch.Tensor:
        """Unpack indices and reconstruct."""
        if self._pack_bits == 8:
            idx = packed.to(torch.int16)
        else:
            idx = unpack_bits(packed.to(self.device), self._pack_bits, self.d)
        x̃ = self.dequant(idx)
        return x̃ * norms.to(self.dtype).unsqueeze(-1)


# ---------------------------------------------------------------------------
# MPS-native inner-product quantizer
# ---------------------------------------------------------------------------

class MPSTurboQuantProd:
    """
    TurboQuant_prod implemented with pure PyTorch operations.

    Memory layout per token per head at b bits:
      - MSE indices (b-1 bits packed): ceil((b-1)*d / 8) bytes
      - QJL bits (1 bit packed):        ceil(d / 8) bytes
      - qjl_gamma scalar (float16):     2 bytes
      - norm scalar (float16):          2 bytes
    Total at b=2, d=64: 16 + 8 + 2 + 2 = 28 bytes vs FP16's 128 → 4.6×
    Total at b=2, d=128: 32 + 16 + 2 + 2 = 52 bytes vs 256 → 4.9×
    """

    def __init__(
        self,
        d: int,
        b: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float16,
        seed: int | None = None,
    ):
        if b < 1:
            raise ValueError("b must be ≥ 1")
        self.d = d
        self.b = b
        self.device = torch.device(device)
        self.dtype = dtype
        self._qjl_scale = math.sqrt(math.pi / 2) / d

        rng = np.random.default_rng(seed)
        mse_seed = int(rng.integers(0, 2**31))
        self.mse = MPSTurboQuantMSE(
            d, max(b - 1, 1), device=device, dtype=dtype, seed=mse_seed
        )

        S_np = rng.standard_normal((d, d)).astype(np.float32)
        self.S  = torch.from_numpy(S_np).to(dtype=dtype, device=self.device)
        self.ST = self.S.T.contiguous()

    # ---- Quantization ----

    @torch.no_grad()
    def quant(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (..., d) unit-norm  →  (idx, qjl, gamma)
        """
        x = x.to(self.dtype)
        if self.b == 1:
            idx = torch.zeros(*x.shape, dtype=torch.int16, device=self.device)
            r = x
        else:
            idx = self.mse.quant(x)                          # (..., d)
            x̃_mse = self.mse.dequant(idx)
            r = x - x̃_mse                                    # residual

        gamma = r.norm(dim=-1)                               # (...,)
        proj = r @ self.ST                                   # (..., d)
        qjl = proj.sign().to(torch.int8)
        qjl[qjl == 0] = 1
        return idx, qjl, gamma

    @torch.no_grad()
    def dequant(
        self,
        idx: torch.Tensor,
        qjl: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """(idx, qjl, gamma)  →  x̃: (..., d)"""
        if self.b == 1:
            x̃_mse = torch.zeros(
                *idx.shape[:-1], self.d, dtype=self.dtype, device=self.device
            )
        else:
            x̃_mse = self.mse.dequant(idx)

        qjl_proj = qjl.to(self.dtype) @ self.S              # (..., d)
        x̃_qjl = self._qjl_scale * gamma.unsqueeze(-1) * qjl_proj
        return x̃_mse + x̃_qjl

    @torch.no_grad()
    def quant_pack(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalise, quantize, and pack into compact storage.

        Returns:
            packed_idx:   (..., ceil((b-1)*d/8)) int8   — bit-packed MSE indices
            packed_qjl:   (..., ceil(d/8))       int8   — bit-packed QJL signs
            qjl_gamma:    (...,)                 float16
            norms:        (...,)                 float16
        """
        norms = x.norm(dim=-1).clamp(min=1e-12)
        x_unit = x / norms.unsqueeze(-1)
        idx, qjl, qjl_gamma = self.quant(x_unit)
        pb = _storage_bits(max(self.b - 1, 1))
        packed_idx = pack_bits(idx, pb) if pb < 8 else idx.to(torch.int8)
        packed_qjl = pack_qjl(qjl)
        return (
            packed_idx.to(self.device),
            packed_qjl.to(self.device),
            qjl_gamma.to(torch.float16),
            norms.to(torch.float16),
        )

    @torch.no_grad()
    def dequant_unpack(
        self,
        packed_idx: torch.Tensor,
        packed_qjl: torch.Tensor,
        qjl_gamma: torch.Tensor,
        norms: torch.Tensor,
    ) -> torch.Tensor:
        """Unpack all components and reconstruct x."""
        pb = _storage_bits(max(self.b - 1, 1))
        idx = (unpack_bits(packed_idx.to(self.device), pb, self.d)
               if pb < 8 else packed_idx.to(torch.int16))
        qjl = unpack_qjl(packed_qjl.to(self.device), self.d)
        gamma = qjl_gamma.to(self.dtype).to(self.device)
        norms = norms.to(self.dtype).to(self.device)
        x̃_unit = self.dequant(idx, qjl, gamma)
        return x̃_unit * norms.unsqueeze(-1)

    # ---- Theoretical memory accounting ----

    def compressed_bytes_per_token_head(self) -> int:
        """Theoretical bytes needed per token per KV head (with bit packing)."""
        pb = _storage_bits(max(self.b - 1, 1))
        if pb < 8:
            idx_bytes = math.ceil(pb * self.d / 8)
        else:
            idx_bytes = self.d   # 1 byte per value (int8)
        qjl_bytes    = math.ceil(self.d / 8)
        scalar_bytes = 2 + 2    # qjl_gamma + norm (float16 each)
        return idx_bytes + qjl_bytes + scalar_bytes

    def fp16_bytes_per_token_head(self) -> int:
        """FP16 bytes per token per KV head."""
        return self.d * 2   # float16 = 2 bytes

    def compression_ratio(self) -> float:
        return self.fp16_bytes_per_token_head() / self.compressed_bytes_per_token_head()
