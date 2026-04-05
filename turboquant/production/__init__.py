"""Production-ready TurboQuant for multi-model Llama inference."""

from .model_configs import LlamaVariant, get_model_config, SUPPORTED_MODELS
from .cache import ProductionTurboQuantCache
from .attention_patch import patch_llama_model, unpatch_llama_model

__all__ = [
    "LlamaVariant",
    "get_model_config",
    "SUPPORTED_MODELS",
    "ProductionTurboQuantCache",
    "patch_llama_model",
    "unpatch_llama_model",
]
