"""
TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.

Python implementation of the algorithms from:
  "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
  Zandieh, Daliri, Hadian, Mirrokni — arXiv 2504.19874, 2025.

Quick start:
    from turboquant import TurboQuantMSE, TurboQuantProd

    # MSE-optimal quantizer at 2 bits per coordinate, dimension 128
    q = TurboQuantMSE(d=128, b=2)
    idx = q.quant(x)       # compress
    x_hat = q.dequant(idx) # reconstruct

    # Inner-product-optimal quantizer
    q2 = TurboQuantProd(d=128, b=2)
    idx, qjl, gamma = q2.quant(x)
    x_hat = q2.dequant(idx, qjl, gamma)
"""

from .quantizer import TurboQuantMSE, TurboQuantProd
from .kv_cache import TurboQuantDynamicCache
from .mps_quantizer import MPSTurboQuantMSE, MPSTurboQuantProd
from .mps_kv_cache import MPSTurboQuantCache

__all__ = [
    "TurboQuantMSE",
    "TurboQuantProd",
    "TurboQuantDynamicCache",
    "MPSTurboQuantMSE",
    "MPSTurboQuantProd",
    "MPSTurboQuantCache",
]
__version__ = "0.1.0"
