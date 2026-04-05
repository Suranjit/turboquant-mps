"""
TurboQuant quantizers as described in the paper.

Two variants:
  - TurboQuantMSE   (Algorithm 1): minimises mean-squared error
  - TurboQuantProd  (Algorithm 2): provides unbiased inner-product estimates

Both accept batches of vectors (shape [..., d]) for vectorised operation.
"""

from __future__ import annotations

import math
import numpy as np
from .codebook import get_codebook


class TurboQuantMSE:
    """
    MSE-optimal TurboQuant (Algorithm 1).

    Setup (shared state, computed once):
      - Random rotation matrix Π ∈ ℝ^{d×d}  (via QR of random normal matrix)
      - Codebook c_1 ≤ c_2 ≤ … ≤ c_{2^b}   (Lloyd-Max on Beta distribution)

    QUANT(x):
      y  = Π · x
      idx_j = argmin_k |y_j - c_k|  for each j

    DEQUANT(idx):
      ỹ_j = c_{idx_j}
      x̃   = Πᵀ · ỹ
    """

    def __init__(self, d: int, b: int, seed: int | None = None):
        """
        Args:
            d:    Dimension of the vectors to quantize.
            b:    Bits per coordinate (total budget = b·d bits per vector).
            seed: Optional RNG seed for the rotation matrix (for reproducibility).
        """
        self.d = d
        self.b = b

        rng = np.random.default_rng(seed)
        # Generate orthogonal rotation matrix via QR decomposition.
        # QR is computed in float64 for numerical stability, then stored as
        # float32 to match the MPS quantizer and keep memory usage consistent.
        raw = rng.standard_normal((d, d))
        rotation_f64, _ = np.linalg.qr(raw)
        self.rotation = rotation_f64.astype(np.float32)  # Π : shape (d, d)

        # Precomputed Lloyd-Max codebook (float32)
        self.codebook = get_codebook(d, b).astype(np.float32)  # shape (2^b,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quant(self, x: np.ndarray) -> np.ndarray:
        """
        Quantize a batch of vectors.

        Args:
            x: Vectors of shape (..., d).  Should be unit-norm; if not, call
               quant_with_norm() which handles normalisation automatically.

        Returns:
            idx: Integer indices of shape (..., d), dtype int16.
        """
        x = np.asarray(x, dtype=np.float32)
        y = x @ self.rotation.T          # (..., d)  —  rotate: y = Π · x
        idx = self._nearest_centroid(y)  # (..., d)
        return idx

    def dequant(self, idx: np.ndarray) -> np.ndarray:
        """
        Reconstruct vectors from their quantized indices.

        Args:
            idx: Integer indices of shape (..., d).

        Returns:
            x̃: Reconstructed vectors of shape (..., d).
        """
        ỹ = self.codebook[idx]           # (..., d)  look up centroids
        x̃ = ỹ @ self.rotation           # (..., d)  rotate back: x̃ = Πᵀ · ỹ
        return x̃

    def quant_with_norm(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Normalise x, quantize the unit-norm version, and return (idx, norm).
        Use dequant_with_norm() to reconstruct.
        """
        x = np.asarray(x, dtype=np.float32)
        norms = np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-12)
        idx = self.quant(x / norms)
        return idx, norms.squeeze(-1)

    def dequant_with_norm(self, idx: np.ndarray, norms: np.ndarray) -> np.ndarray:
        """Reconstruct vectors from (idx, norm)."""
        x̃ = self.dequant(idx)
        return x̃ * norms[..., np.newaxis]

    def mse(self, x: np.ndarray) -> float:
        """
        Empirical D_mse = E[‖x − x̃‖²] on a batch of vectors.

        This matches the paper's definition: mean over vectors of the full
        squared reconstruction error (sum over all d coordinates), NOT the
        per-element mean.
        """
        x = np.asarray(x, dtype=np.float32)
        idx = self.quant(x)
        x̃ = self.dequant(idx)
        return float(np.mean(np.sum((x - x̃) ** 2, axis=-1)))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _nearest_centroid(self, y: np.ndarray) -> np.ndarray:
        """
        Vectorised nearest-centroid lookup using binary search (O(log 2^b) = O(b)).

        Args:
            y: Rotated coordinates of shape (..., d).

        Returns:
            idx: Centroid indices of shape (..., d), dtype int16.
        """
        # np.searchsorted gives the insertion point; we check left and right
        # neighbours to find the nearest centroid.
        flat = y.reshape(-1)
        # boundaries are midpoints between codebook entries
        boundaries = (self.codebook[:-1] + self.codebook[1:]) / 2.0
        idx_flat = np.searchsorted(boundaries, flat)   # values in [0, 2^b - 1]
        return idx_flat.reshape(y.shape).astype(np.int16)


class TurboQuantProd:
    """
    Inner-product-optimal TurboQuant (Algorithm 2).

    Combines TurboQuantMSE(b-1 bits) with a 1-bit QJL layer on the residual.
    The result is an *unbiased* inner-product estimator with near-optimal
    inner-product distortion.

    QUANT(x):
      idx       = QUANT_mse(x)              # b-1 bits
      r         = x - DEQUANT_mse(idx)      # residual
      qjl       = sign(S · r)               # 1-bit QJL, shape (d,)
      γ         = ||r||₂                    # scalar norm of residual

    DEQUANT(idx, qjl, γ):
      x̃_mse  = DEQUANT_mse(idx)
      x̃_qjl  = √(π/2)/d · γ · Sᵀ · qjl
      output  = x̃_mse + x̃_qjl
    """

    def __init__(self, d: int, b: int, seed: int | None = None):
        """
        Args:
            d:    Dimension.
            b:    Total bits per coordinate (must be ≥ 1).
            seed: Optional RNG seed.
        """
        if b < 1:
            raise ValueError("b must be ≥ 1 for TurboQuantProd")
        self.d = d
        self.b = b

        rng = np.random.default_rng(seed)
        # Instantiate MSE quantizer with b-1 bits
        mse_seed = int(rng.integers(0, 2**31))
        self.mse_quant = TurboQuantMSE(d, max(b - 1, 1), seed=mse_seed)

        # Random projection matrix S ∈ ℝ^{d×d}, entries ~ N(0, 1)
        self.S = rng.standard_normal((d, d)).astype(np.float32)  # shape (d, d)

        # Precompute Sᵀ for fast dequantisation
        self.ST = self.S.T.copy()

        # Scale factor for QJL dequantisation: √(π/2) / d
        self._qjl_scale = math.sqrt(math.pi / 2) / d

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quant(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Quantize a batch of unit-norm vectors.

        Args:
            x: Vectors of shape (..., d), should be unit-norm.

        Returns:
            idx:  MSE indices, shape (..., d), dtype int16.
            qjl:  QJL bits as int8 {-1, +1}, shape (..., d).
            gamma: L2 norm of the residual, shape (...,).
        """
        x = np.asarray(x, dtype=np.float32)
        # MSE quantize at b-1 bits
        if self.b == 1:
            # Special case: skip MSE stage, only QJL
            idx = np.zeros(x.shape, dtype=np.int16)
            r = x
        else:
            idx = self.mse_quant.quant(x)
            x̃_mse = self.mse_quant.dequant(idx)
            r = x - x̃_mse                         # residual, shape (..., d)

        gamma = np.linalg.norm(r, axis=-1)         # shape (...,)

        # QJL: sign(S · r) — apply projection and take sign
        # For batched input: r @ Sᵀ gives (..., d)
        proj = r @ self.ST                          # (..., d)
        qjl = np.sign(proj).astype(np.int8)
        qjl[qjl == 0] = 1                          # avoid zero (rare)

        return idx, qjl, gamma

    def dequant(
        self,
        idx: np.ndarray,
        qjl: np.ndarray,
        gamma: np.ndarray,
    ) -> np.ndarray:
        """
        Reconstruct vectors from their quantized representation.

        Args:
            idx:   MSE indices, shape (..., d).
            qjl:   QJL bits {-1, +1}, shape (..., d).
            gamma: Residual norms, shape (...,).

        Returns:
            x̃: Reconstructed vectors, shape (..., d).
        """
        if self.b == 1:
            x̃_mse = np.zeros((*idx.shape[:-1], self.d), dtype=np.float32)
        else:
            x̃_mse = self.mse_quant.dequant(idx)   # (..., d)

        # x̃_qjl = √(π/2)/d · γ · Sᵀ · qjl
        # qjl @ S  gives (..., d)
        qjl_proj = qjl.astype(np.float32) @ self.S  # (..., d)
        g = gamma[..., np.newaxis]                   # (..., 1)
        x̃_qjl = self._qjl_scale * g * qjl_proj     # (..., d)

        return x̃_mse + x̃_qjl

    def quant_with_norm(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalise, quantize, and return (idx, qjl, qjl_gamma, vec_norm).
        Use dequant_with_norm() to reconstruct.
        """
        x = np.asarray(x, dtype=np.float32)
        norms = np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-12)
        x_unit = x / norms
        idx, qjl, qjl_gamma = self.quant(x_unit)
        return idx, qjl, qjl_gamma, norms.squeeze(-1)

    def dequant_with_norm(
        self,
        idx: np.ndarray,
        qjl: np.ndarray,
        qjl_gamma: np.ndarray,
        vec_norm: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct from (idx, qjl, qjl_gamma, vec_norm)."""
        x̃_unit = self.dequant(idx, qjl, qjl_gamma)
        return x̃_unit * vec_norm[..., np.newaxis]

    def inner_product(
        self,
        y: np.ndarray,
        idx: np.ndarray,
        qjl: np.ndarray,
        gamma: np.ndarray,
    ) -> np.ndarray:
        """
        Compute unbiased inner product ⟨y, x⟩ estimate from the compressed
        representation of x, WITHOUT fully reconstructing x.

        This is the fast path for nearest-neighbour search:
          ⟨y, x̃⟩ ≈ ⟨y, x̃_mse⟩ + ||r||₂ · √(π/2)/d · ⟨y, Sᵀ·qjl⟩

        Args:
            y:     Query vector, shape (d,) or (..., d).
            idx:   MSE indices for x, shape (n, d).
            qjl:   QJL bits for residual of x, shape (n, d).
            gamma: Residual norms for x, shape (n,).

        Returns:
            Inner product estimates, shape (n,) or matching batch shape.
        """
        y = np.asarray(y, dtype=np.float32)
        x̃_mse = self.mse_quant.dequant(idx)         # (n, d)

        # Support both single query y=(d,) and batched y=(n,d)
        if y.ndim == 1:
            ip_mse = (x̃_mse * y).sum(-1)            # (n,)
            Sy = self.S @ y                           # (d,)
        else:
            ip_mse = (x̃_mse * y).sum(-1)            # (n,)
            # For batched y, compute per-sample S·y_i  → shape (n, d)
            Sy = y @ self.ST                          # (n, d)

        ip_qjl = (qjl.astype(np.float32) * Sy).sum(-1)  # (n,)
        ip_qjl *= gamma * self._qjl_scale

        return ip_mse + ip_qjl

    def inner_product_distortion(self, x: np.ndarray, y: np.ndarray) -> float:
        """Empirical D_prod on batches x and y (both unit-norm)."""
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        idx, qjl, gamma = self.quant(x)
        ip_true = (x * y).sum(-1)
        ip_est = self.inner_product(y, idx, qjl, gamma)
        return float(np.mean((ip_true - ip_est) ** 2))
