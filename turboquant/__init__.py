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

# Production Metal shader backend
from .metal import MetalTurboQuantMSE, MetalTurboQuantProd, metal_available
from .production import (
    ProductionTurboQuantCache,
    LlamaVariant,
    get_model_config,
    SUPPORTED_MODELS,
    patch_llama_model,
    unpatch_llama_model,
)

__all__ = [
    # Core numpy quantizers
    "TurboQuantMSE",
    "TurboQuantProd",
    "TurboQuantDynamicCache",
    # PyTorch MPS quantizers
    "MPSTurboQuantMSE",
    "MPSTurboQuantProd",
    "MPSTurboQuantCache",
    # Metal shader quantizers (with PyTorch MPS fallback)
    "MetalTurboQuantMSE",
    "MetalTurboQuantProd",
    "metal_available",
    # Production multi-model cache
    "ProductionTurboQuantCache",
    "LlamaVariant",
    "get_model_config",
    "SUPPORTED_MODELS",
    "patch_llama_model",
    "unpatch_llama_model",
]
__version__ = "0.2.0"
