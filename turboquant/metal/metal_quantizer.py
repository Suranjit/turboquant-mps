"""
Production Metal quantizers: fused GPU kernels with PyTorch MPS fallback.

Both MetalTurboQuantMSE and MetalTurboQuantProd share the same API as their
MPS counterparts (MPSTurboQuantMSE / MPSTurboQuantProd) so they can be used
as drop-in replacements in model patching.

Execution strategy (automatic):
  1. If PyObjC Metal is available AND Metal library compiled → use Metal kernels
     (fused, no Python overhead per kernel call)
  2. Otherwise → fall back to MPSTurboQuantMSE / MPSTurboQuantProd (PyTorch)

Both paths produce identical numerical outputs (float32 precision).
"""

from __future__ import annotations

import math
import logging
import struct
from typing import Optional, Tuple

import numpy as np
import torch

from turboquant.codebook import get_codebook
from turboquant.metal.metal_lib import MetalLib, metal_available

logger = logging.getLogger(__name__)

# Lazy import of fallback MPS quantizers
def _get_mps_fallback():
    from turboquant.mps_quantizer import MPSTurboQuantMSE, MPSTurboQuantProd
    return MPSTurboQuantMSE, MPSTurboQuantProd


def _uint_constant_buffer(lib: MetalLib, value: int) -> object:
    """Create a 4-byte Metal buffer holding a uint32 constant."""
    import ctypes
    import Metal
    data = struct.pack("<I", value)
    buf = lib._device.newBufferWithBytes_length_options_(data, 4, 0)
    return buf


class MetalTurboQuantMSE:
    """
    MSE-optimal TurboQuant using fused Metal GPU kernels.

    API mirrors MPSTurboQuantMSE exactly. Falls back to PyTorch MPS if Metal
    library is unavailable (xcrun not installed, or PyObjC missing).

    Args:
        d:      Head dimension.
        b:      Bits per coordinate (1-8).
        device: torch device string, e.g. "mps" or "cpu".
        dtype:  Working dtype (torch.float16 or torch.float32).
        seed:   RNG seed for rotation matrix.
    """

    def __init__(
        self,
        d: int,
        b: int,
        device: str = "mps",
        dtype: torch.dtype = torch.float16,
        seed: Optional[int] = None,
    ):
        self.d = d
        self.b = b
        self.device = device
        self.dtype = dtype

        # Build rotation matrix (float32 for Metal; same as MPS variant)
        rng = np.random.default_rng(seed)
        raw = rng.standard_normal((d, d))
        rot_f64, _ = np.linalg.qr(raw)
        self.rotation_np = rot_f64.astype(np.float32)
        self.rotation = torch.from_numpy(self.rotation_np).to(device)

        codebook_np = get_codebook(d, b).astype(np.float32)
        self.codebook_np = codebook_np
        self.codebook = torch.from_numpy(codebook_np).to(device)

        # Attempt to load Metal library
        self._metal: Optional[MetalLib] = MetalLib.load()
        if self._metal is not None:  # pragma: no cover
            logger.info("MetalTurboQuantMSE[d=%d, b=%d]: using Metal kernel path", d, b)
            self._init_metal_buffers()
        else:
            logger.info("MetalTurboQuantMSE[d=%d, b=%d]: using PyTorch MPS fallback", d, b)
            self._fallback = self._build_fallback(seed)

    # ------------------------------------------------------------------
    # Metal buffer pre-loading (rotation + codebook stay on GPU)
    # ------------------------------------------------------------------

    def _init_metal_buffers(self):  # pragma: no cover
        lib = self._metal
        self._rot_buf  = lib.make_buffer_from_numpy(self.rotation_np)
        self._cb_buf   = lib.make_buffer_from_numpy(self.codebook_np)
        self._d_buf    = _uint_constant_buffer(lib, self.d)
        self._b_buf    = _uint_constant_buffer(lib, self.b)
        self._norm0_buf = _uint_constant_buffer(lib, 0)  # normalize=0
        self._norm1_buf = _uint_constant_buffer(lib, 1)  # normalize=1

    def _build_fallback(self, seed):
        MPSTurboQuantMSE, _ = _get_mps_fallback()
        fb = MPSTurboQuantMSE(self.d, self.b, device=self.device, dtype=self.dtype, seed=seed)
        fb.rotation = self.rotation.to(dtype=self.dtype)
        fb.codebook = self.codebook.to(dtype=self.dtype)
        return fb

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quant(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize unit-norm vectors.

        Args:
            x: (N, d) float tensor on self.device
        Returns:
            idx: (N, d) int16 tensor
        """
        if self._metal is not None:
            return self._metal_quant(x, normalize=False)
        return self._fallback.quant(x)

    def dequant(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct from integer indices.

        Args:
            idx: (N, d) int16
        Returns:
            (N, d) float32
        """
        if self._metal is not None:
            return self._metal_dequant(idx, has_norm=False)
        return self._fallback.dequant(idx)

    def quant_pack(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize, quantize, and bit-pack.

        Returns:
            packed_idx: (N, ceil(b*d/8)) int8
            norms:      (N,) float16
        """
        if self._metal is not None:
            return self._metal_quant_pack(x)
        return self._fallback.quant_pack(x)

    def dequant_unpack(self, packed_idx: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """
        Unpack, dequantize, and scale by norms.

        Args:
            packed_idx: (N, ceil(b*d/8)) int8
            norms:      (N,) float16
        Returns:
            (N, d) tensor in self.dtype
        """
        if self._metal is not None:
            return self._metal_dequant_unpack(packed_idx, norms)
        return self._fallback.dequant_unpack(packed_idx, norms)

    def compressed_bytes_per_token_head(self) -> int:
        """Bytes per (token, head) including norms."""
        from turboquant.mps_quantizer import _storage_bits
        pb = _storage_bits(self.b)
        return math.ceil(pb * self.d / 8) + 2  # 2 bytes for float16 norm

    # ------------------------------------------------------------------
    # Metal kernel dispatches
    # ------------------------------------------------------------------

    def _metal_quant(self, x: torch.Tensor, normalize: bool) -> torch.Tensor:  # pragma: no cover
        lib = self._metal
        N = x.shape[0]
        x_f32 = x.float().cpu().contiguous()
        bytes_per = (self.b * self.d + 7) // 8

        x_buf    = lib.make_buffer_from_numpy(x_f32.numpy())
        out_buf  = lib.make_output_buffer(N * bytes_per)
        norm_buf = lib.make_output_buffer(N * 4)  # float32

        lib.run_kernel("tq_mse_quantize_fused", buffers=[
            x_buf, self._rot_buf, self._cb_buf, out_buf, norm_buf,
            self._d_buf, self._b_buf,
            self._norm1_buf if normalize else self._norm0_buf,
        ], grid=N)

        packed = lib.buffer_to_numpy(out_buf, (N, bytes_per), np.uint8)
        return self._packed_to_idx(packed)

    def _packed_to_idx(self, packed: np.ndarray) -> torch.Tensor:
        """Unpack bytes → int16 indices via the existing unpack_bits utility."""
        from turboquant.mps_quantizer import _storage_bits, unpack_bits
        import torch
        pb = _storage_bits(self.b)
        t = torch.from_numpy(packed.view(np.int8)).to(self.device)
        if pb < 8:
            return unpack_bits(t, pb, self.d).to(torch.int16)
        return t.to(torch.int16)

    def _metal_dequant(self, idx: torch.Tensor, has_norm: bool, norms: Optional[torch.Tensor] = None) -> torch.Tensor:  # pragma: no cover
        lib = self._metal
        N = idx.shape[0]

        # Pack idx back to bytes for the Metal kernel
        packed_np = self._idx_to_packed_np(idx)
        bytes_per = packed_np.shape[-1]

        x_buf    = lib.make_buffer_from_numpy(packed_np)
        out_buf  = lib.make_output_buffer(N * self.d * 4)
        norms_buf = lib.make_buffer_from_numpy(
            norms.cpu().float().numpy() if norms is not None
            else np.ones(N, dtype=np.float32)
        )

        lib.run_kernel("tq_mse_dequantize_fused", buffers=[
            x_buf, self._rot_buf, self._cb_buf, norms_buf, out_buf,
            self._d_buf, self._b_buf,
            _uint_constant_buffer(lib, 1 if has_norm else 0),
        ], grid=N)

        out_np = lib.buffer_to_numpy(out_buf, (N, self.d), np.float32)
        return torch.from_numpy(out_np).to(device=self.device, dtype=self.dtype)

    def _idx_to_packed_np(self, idx: torch.Tensor) -> np.ndarray:
        """Pack int16 indices back to bytes for the Metal kernel input."""
        from turboquant.mps_quantizer import _storage_bits, pack_bits
        pb = _storage_bits(self.b)
        if pb < 8:
            packed = pack_bits(idx.cpu(), pb)
            return packed.numpy().view(np.uint8)
        return idx.cpu().numpy().astype(np.uint8)

    def _metal_quant_pack(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover
        lib = self._metal
        N = x.shape[0]
        x_f32 = x.float().cpu().contiguous()
        bytes_per = (self.b * self.d + 7) // 8

        x_buf    = lib.make_buffer_from_numpy(x_f32.numpy())
        out_buf  = lib.make_output_buffer(N * bytes_per)
        norm_buf = lib.make_output_buffer(N * 4)  # float32 norms

        lib.run_kernel("tq_mse_quantize_fused", buffers=[
            x_buf, self._rot_buf, self._cb_buf, out_buf, norm_buf,
            self._d_buf, self._b_buf, self._norm1_buf,
        ], grid=N)

        packed_np = lib.buffer_to_numpy(out_buf, (N, bytes_per), np.int8)
        norms_np  = lib.buffer_to_numpy(norm_buf, (N,), np.float32)

        packed = torch.from_numpy(packed_np).to(self.device)
        norms  = torch.from_numpy(norms_np).to(self.device, dtype=torch.float16)
        return packed, norms

    def _metal_dequant_unpack(self, packed: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        lib = self._metal
        N = packed.shape[0]

        packed_np = packed.cpu().numpy().view(np.uint8)
        norms_np  = norms.cpu().float().numpy()

        x_buf    = lib.make_buffer_from_numpy(packed_np)
        out_buf  = lib.make_output_buffer(N * self.d * 4)
        norm_buf = lib.make_buffer_from_numpy(norms_np)

        lib.run_kernel("tq_mse_dequantize_fused", buffers=[
            x_buf, self._rot_buf, self._cb_buf, norm_buf, out_buf,
            self._d_buf, self._b_buf, self._norm1_buf,
        ], grid=N)

        out_np = lib.buffer_to_numpy(out_buf, (N, self.d), np.float32)
        return torch.from_numpy(out_np).to(device=self.device, dtype=self.dtype)


class MetalTurboQuantProd:
    """
    Inner-product-optimal TurboQuant using fused Metal GPU kernels.

    Combines TurboQuantMSE(b-1) + 1-bit QJL on residual.
    Falls back to MPSTurboQuantProd when Metal is unavailable.
    """

    def __init__(
        self,
        d: int,
        b: int,
        device: str = "mps",
        dtype: torch.dtype = torch.float16,
        seed: Optional[int] = None,
    ):
        if b < 1:
            raise ValueError("b must be >= 1 for MetalTurboQuantProd")
        self.d = d
        self.b = b
        self.device = device
        self.dtype = dtype

        rng = np.random.default_rng(seed)
        mse_seed = int(rng.integers(0, 2**31))
        self.mse = MetalTurboQuantMSE(d, max(b - 1, 1), device=device, dtype=dtype, seed=mse_seed)

        # QJL projection matrix S (d×d)
        s_np = rng.standard_normal((d, d)).astype(np.float32)
        self.S_np  = s_np
        self.S     = torch.from_numpy(s_np).to(device)
        self.ST    = self.S.T.contiguous()
        self._qjl_scale = math.sqrt(math.pi / 2) / d

        # Metal library
        self._metal: Optional[MetalLib] = self.mse._metal
        if self._metal is not None:  # pragma: no cover
            logger.info("MetalTurboQuantProd[d=%d, b=%d]: using Metal kernel path", d, b)
            self._init_metal_buffers()
        else:
            logger.info("MetalTurboQuantProd[d=%d, b=%d]: using PyTorch MPS fallback", d, b)
            self._fallback = self._build_fallback(seed)

    def _init_metal_buffers(self):  # pragma: no cover
        lib = self._metal
        mse = self.mse
        b_mse = max(self.b - 1, 1)
        self._s_buf    = lib.make_buffer_from_numpy(self.S_np)
        self._d_buf    = _uint_constant_buffer(lib, self.d)
        self._b_buf    = _uint_constant_buffer(lib, self.b)
        self._bm1_buf  = _uint_constant_buffer(lib, b_mse)

    def _build_fallback(self, seed):
        _, MPSTurboQuantProd = _get_mps_fallback()
        fb = MPSTurboQuantProd(self.d, self.b, device=self.device, dtype=self.dtype, seed=seed)
        fb.mse.rotation = self.mse.rotation.to(dtype=self.dtype)
        fb.mse.codebook = self.mse.codebook.to(dtype=self.dtype)
        fb.S  = self.S.to(dtype=self.dtype)
        fb.ST = self.ST.to(dtype=self.dtype)
        return fb

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quant(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize unit-norm vectors.

        Returns:
            idx:   (N, d) int16 — MSE indices
            qjl:   (N, d) int8  — QJL bits {-1, +1}
            gamma: (N,)  float32 — residual norms
        """
        if self._metal is not None:
            return self._metal_quant(x)
        return self._fallback.quant(x)

    def dequant(
        self,
        idx: torch.Tensor,
        qjl: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct vectors from compressed representation.

        Returns:
            (N, d) tensor in self.dtype
        """
        if self._metal is not None:
            return self._metal_dequant(idx, qjl, gamma)
        return self._fallback.dequant(idx, qjl, gamma)

    def quant_pack(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize, quantize, and bit-pack.

        Returns:
            packed_idx:  (N, ceil((b-1)*d/8)) int8
            packed_qjl:  (N, ceil(d/8))       int8
            gamma:       (N,)                  float32
            norms:       (N,)                  float16
        """
        if self._metal is not None:
            return self._metal_quant_pack(x)
        return self._fallback.quant_pack(x)

    def dequant_unpack(
        self,
        packed_idx: torch.Tensor,
        packed_qjl: torch.Tensor,
        gamma: torch.Tensor,
        norms: torch.Tensor,
    ) -> torch.Tensor:
        """
        Unpack, dequantize, and scale by norms.
        """
        if self._metal is not None:
            return self._metal_dequant_unpack(packed_idx, packed_qjl, gamma, norms)
        return self._fallback.dequant_unpack(packed_idx, packed_qjl, gamma, norms)

    def inner_product(
        self,
        y: torch.Tensor,
        idx: torch.Tensor,
        qjl: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fast inner-product estimate without full reconstruction.
        Uses fused Metal kernel if available.
        """
        if self._metal is not None:
            return self._metal_inner_product(y, idx, qjl, gamma)
        # PyTorch MPS fallback: implement directly (MPSTurboQuantProd lacks this method)
        return self._torch_inner_product(y, idx, qjl, gamma)

    def compressed_bytes_per_token_head(self) -> int:
        from turboquant.mps_quantizer import _storage_bits
        b_mse = max(self.b - 1, 1)
        pb = _storage_bits(b_mse)
        return math.ceil(pb * self.d / 8) + math.ceil(self.d / 8) + 2 + 4

    # ------------------------------------------------------------------
    # PyTorch fallback inner-product
    # ------------------------------------------------------------------

    def _torch_inner_product(
        self,
        y: torch.Tensor,
        idx: torch.Tensor,
        qjl: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """Pure-PyTorch inner product estimate (used when Metal not available)."""
        fb = self._fallback
        if self.b > 1:
            x_mse = fb.mse.dequant(idx)
            ip_mse = (x_mse.float() * y.float()).sum(-1)
        else:
            ip_mse = torch.zeros(idx.shape[0], device=self.device)

        y_f = y.float()
        if y_f.ndim == 1:
            Sy = self.S.float() @ y_f
        else:
            Sy = y_f @ self.ST.float()
        ip_qjl = (qjl.float() * Sy).sum(-1) * gamma.float() * self._qjl_scale
        return (ip_mse + ip_qjl).to(torch.float32)

    # ------------------------------------------------------------------
    # Metal kernel dispatches
    # ------------------------------------------------------------------

    def _metal_quant(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lib = self._metal
        N = x.shape[0]
        b_mse = max(self.b - 1, 1)
        idx_bytes = (b_mse * self.d + 7) // 8
        qjl_bytes = (self.d + 7) // 8

        x_np = x.float().cpu().contiguous().numpy()
        x_buf    = lib.make_buffer_from_numpy(x_np)
        idx_buf  = lib.make_output_buffer(N * idx_bytes)
        qjl_buf  = lib.make_output_buffer(N * qjl_bytes)
        gam_buf  = lib.make_output_buffer(N * 4)  # float32

        lib.run_kernel("tq_prod_quantize_fused", buffers=[
            x_buf,
            self.mse._rot_buf,
            self.mse._cb_buf,
            self._s_buf,
            idx_buf, qjl_buf, gam_buf,
            self._d_buf, self._b_buf,
        ], grid=N)

        idx_np  = lib.buffer_to_numpy(idx_buf, (N, idx_bytes), np.uint8)
        qjl_np  = lib.buffer_to_numpy(qjl_buf, (N, qjl_bytes), np.uint8)
        gam_np  = lib.buffer_to_numpy(gam_buf, (N,), np.float32)

        # Unpack to int16 / int8
        idx = self.mse._packed_to_idx(torch.from_numpy(idx_np.view(np.int8)))

        # QJL: unpack 1-bit → {-1, +1}
        from turboquant.mps_quantizer import unpack_qjl
        qjl = unpack_qjl(
            torch.from_numpy(qjl_np.view(np.int8)).to(self.device), self.d
        )

        gamma = torch.from_numpy(gam_np).to(self.device)
        return idx, qjl, gamma

    def _metal_dequant(
        self,
        idx: torch.Tensor,
        qjl: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        lib = self._metal
        N = idx.shape[0]
        b_mse = max(self.b - 1, 1)
        qjl_bytes = (self.d + 7) // 8

        idx_packed = self.mse._idx_to_packed_np(idx)
        # Pack qjl back: 1 bit per coord
        from turboquant.mps_quantizer import pack_qjl
        qjl_packed = pack_qjl(qjl.cpu()).numpy().view(np.uint8)

        idx_buf  = lib.make_buffer_from_numpy(idx_packed)
        qjl_buf  = lib.make_buffer_from_numpy(qjl_packed)
        gam_buf  = lib.make_buffer_from_numpy(gamma.cpu().float().numpy())
        out_buf  = lib.make_output_buffer(N * self.d * 4)

        lib.run_kernel("tq_prod_dequantize_fused", buffers=[
            idx_buf, qjl_buf, gam_buf,
            self.mse._rot_buf, self.mse._cb_buf, self._s_buf,
            out_buf, self._d_buf, self._b_buf,
        ], grid=N)

        out_np = lib.buffer_to_numpy(out_buf, (N, self.d), np.float32)
        return torch.from_numpy(out_np).to(device=self.device, dtype=self.dtype)

    def _metal_quant_pack(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Normalize, then quant
        x_f32  = x.float()
        norms  = x_f32.norm(dim=-1).clamp(min=1e-12)
        x_unit = x_f32 / norms.unsqueeze(-1)
        idx, qjl, gamma = self._metal_quant(x_unit)

        # Repack idx and qjl
        from turboquant.mps_quantizer import pack_bits, pack_qjl, _storage_bits
        b_mse = max(self.b - 1, 1)
        pb = _storage_bits(b_mse)
        packed_idx = pack_bits(idx.cpu(), pb).to(self.device) if pb < 8 else idx.to(torch.int8).to(self.device)
        packed_qjl = pack_qjl(qjl.cpu()).to(self.device)
        return packed_idx, packed_qjl, gamma, norms.to(torch.float16)

    def _metal_dequant_unpack(
        self,
        packed_idx: torch.Tensor,
        packed_qjl: torch.Tensor,
        gamma: torch.Tensor,
        norms: torch.Tensor,
    ) -> torch.Tensor:
        from turboquant.mps_quantizer import _storage_bits, unpack_bits, unpack_qjl
        b_mse = max(self.b - 1, 1)
        pb = _storage_bits(b_mse)
        idx = unpack_bits(packed_idx, pb, self.d) if pb < 8 else packed_idx.to(torch.int16)
        qjl = unpack_qjl(packed_qjl, self.d)
        x_unit = self._metal_dequant(idx, qjl, gamma)
        return x_unit * norms.float().unsqueeze(-1).to(self.device)

    def _metal_inner_product(
        self,
        y: torch.Tensor,
        idx: torch.Tensor,
        qjl: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        lib = self._metal
        N = idx.shape[0]
        b_mse = max(self.b - 1, 1)

        y_np      = y.float().cpu().contiguous().numpy()
        idx_np    = self.mse._idx_to_packed_np(idx)
        from turboquant.mps_quantizer import pack_qjl
        qjl_np    = pack_qjl(qjl.cpu()).numpy().view(np.uint8)
        gamma_np  = gamma.cpu().float().numpy()

        y_buf   = lib.make_buffer_from_numpy(y_np)
        idx_buf = lib.make_buffer_from_numpy(idx_np)
        qjl_buf = lib.make_buffer_from_numpy(qjl_np)
        gam_buf = lib.make_buffer_from_numpy(gamma_np)
        out_buf = lib.make_output_buffer(N * 4)

        lib.run_kernel("tq_inner_product", buffers=[
            y_buf, idx_buf, qjl_buf, gam_buf,
            self.mse._rot_buf, self.mse._cb_buf, self._s_buf,
            out_buf, self._d_buf, self._b_buf,
        ], grid=N)

        out_np = lib.buffer_to_numpy(out_buf, (N,), np.float32)
        return torch.from_numpy(out_np).to(self.device)
