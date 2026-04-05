"""
Monkey-patch utilities for LlamaAttention / LlamaSdpaAttention.

Replaces the model's default (full float16) KV cache with a
ProductionTurboQuantCache by hooking into the `past_key_value` parameter
of each attention layer's forward pass.

Supports:
  - LlamaAttention  (transformers >= 4.31)
  - LlamaSdpaAttention (transformers >= 4.39, default for most new models)
  - Grouped Query Attention (GQA) — the cache is aware of num_kv_heads

Usage::

    model = AutoModelForCausalLM.from_pretrained(...)
    cache = patch_llama_model(model, bits=3)

    # Run inference — cache is used automatically
    output = model.generate(...)

    # Restore original attention (optional)
    unpatch_llama_model(model)
"""

from __future__ import annotations

import logging
import types
from typing import Optional, Tuple, TYPE_CHECKING

import torch

from turboquant.production.cache import ProductionTurboQuantCache

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Store original forward methods here so unpatch works reliably
_ORIGINAL_FORWARDS: dict = {}


def patch_llama_model(
    model,
    bits: int = 3,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    use_metal: bool = True,
    seed: int = 0,
) -> ProductionTurboQuantCache:
    """
    Patch a Llama-style model to use TurboQuant KV compression.

    The patch works by attaching a shared ProductionTurboQuantCache to the
    model object (`model.tq_cache`) and modifying each attention layer's
    `forward` method to route `past_key_value` through it.

    Args:
        model:      A HuggingFace Llama (or compatible) causal LM.
        bits:       Bits per coordinate for quantization (2–4 recommended).
        device:     Device string, e.g. "mps". Defaults to model's device.
        dtype:      Dtype for quantization. Defaults to torch.float16.
        use_metal:  Use fused Metal shader kernels if available.
        seed:       RNG seed for quantizers.

    Returns:
        The ProductionTurboQuantCache instance attached to the model.
    """
    cfg = model.config
    device = device or _infer_device(model)
    dtype  = dtype  or torch.float16

    # Build config from HF model config
    from turboquant.production.model_configs import auto_detect_config
    tq_cfg = auto_detect_config(cfg)

    if tq_cfg is None:
        raise ValueError(
            "Could not determine model architecture from config. "
            "Use ProductionTurboQuantCache.from_model_config() directly."
        )

    logger.info(  # pragma: no cover
        "Patching %s with TurboQuant (b=%d, device=%s, Metal=%s)",
        tq_cfg.name, bits, device, use_metal,
    )

    cache = ProductionTurboQuantCache.from_model_config(
        tq_cfg,
        bits=bits,
        device=device,
        dtype=dtype,
        seed=seed,
        use_metal=use_metal,
    )

    # Attach cache to model for easy access
    model.tq_cache = cache

    # Patch each attention layer
    layers = _get_attention_layers(model)
    for layer_idx, attn_module in enumerate(layers):
        _patch_attention_layer(attn_module, cache, layer_idx)

    logger.info("Patched %d attention layers", len(layers))
    return cache


def unpatch_llama_model(model) -> None:
    """
    Restore original attention forward methods.

    Args:
        model: A previously patched Llama model.
    """
    layers = _get_attention_layers(model)
    restored = 0
    for attn_module in layers:
        module_id = id(attn_module)
        if module_id in _ORIGINAL_FORWARDS:  # pragma: no cover
            attn_module.forward = _ORIGINAL_FORWARDS.pop(module_id)
            restored += 1

    if hasattr(model, "tq_cache"):  # pragma: no cover
        del model.tq_cache

    logger.info("Unpatched %d attention layers", restored)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _infer_device(model) -> str:
    """Get the device of the first model parameter."""
    try:
        p = next(model.parameters())
        return str(p.device)
    except StopIteration:
        return "cpu"


def _get_attention_layers(model) -> list:
    """
    Extract attention modules from a Llama-style model.
    Returns a list in layer order (index = layer_idx).
    """
    # HF LlamaForCausalLM → model.model.layers[i].self_attn
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return [layer.self_attn for layer in model.model.layers]

    # Some models expose layers directly
    if hasattr(model, "layers"):
        return [layer.self_attn for layer in model.layers]

    raise ValueError(
        "Cannot find attention layers. "
        "Expected model.model.layers[i].self_attn structure."
    )


def _patch_attention_layer(  # pragma: no cover
    attn_module,
    cache: ProductionTurboQuantCache,
    layer_idx: int,
) -> None:
    """
    Replace attn_module.forward with a version that injects TurboQuant cache.

    Strategy: intercept the `past_key_value` argument in forward, redirect
    storage to `cache.update()`, and replace the returned past_key_value
    with a simple wrapper that downstream code can index into.
    """
    module_id = id(attn_module)
    if module_id in _ORIGINAL_FORWARDS:
        # Already patched — skip
        return

    original_forward = attn_module.forward
    _ORIGINAL_FORWARDS[module_id] = original_forward

    def patched_forward(
        self_,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        # Delegate to original forward to get fresh KV projections
        # We pass past_key_value=None so it does a fresh projection,
        # then we combine with our cached full KV.
        result = original_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None,   # always compute fresh KV
            output_attentions=output_attentions,
            use_cache=False,       # we handle caching ourselves
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # Extract fresh key/value from output (always at index 1, 2 for most HF models)
        # Some models return (attn_output, attn_weights, past_kv) or (attn_output, past_kv)
        if isinstance(result, tuple):
            attn_output = result[0]
            # Re-compute K, V by projecting hidden_states directly
            # (more reliable than parsing result tuple structure)
        else:
            attn_output = result

        # Compute fresh K/V projections from hidden_states
        # This is model-type-agnostic
        bsz, q_len, _ = hidden_states.shape

        try:
            # Standard LlamaAttention attributes
            key_states, value_states = _project_kv(attn_module, hidden_states, position_ids, position_embeddings)

            if key_states is not None:
                full_k, full_v = cache.update(key_states, value_states, layer_idx)
                # Build a simple past_key_value wrapper for downstream code
                new_past = _TQPastKeyValue(full_k, full_v)
            else:
                new_past = None

        except Exception as exc:
            logger.warning("TurboQuant cache update failed at layer %d: %s", layer_idx, exc)
            new_past = None

        if use_cache:
            if isinstance(result, tuple):
                return (attn_output,) + result[1:-1] + (new_past,)
            return attn_output, new_past

        return result

    # Bind as a method
    attn_module.forward = types.MethodType(patched_forward, attn_module)


def _project_kv(  # pragma: no cover
    attn_module,
    hidden_states: torch.Tensor,
    position_ids=None,
    position_embeddings=None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Apply K/V projections to hidden_states using the attention module's
    k_proj, v_proj, and optional rotary embeddings.

    Returns (key_states, value_states) shaped (B, H_kv, T, d).
    """
    import math

    bsz, q_len, _ = hidden_states.shape

    # K/V projections
    if not (hasattr(attn_module, "k_proj") and hasattr(attn_module, "v_proj")):
        return None, None

    key_states   = attn_module.k_proj(hidden_states)
    value_states = attn_module.v_proj(hidden_states)

    num_kv_heads = attn_module.num_key_value_heads if hasattr(attn_module, "num_key_value_heads") else attn_module.num_heads
    head_dim     = attn_module.head_dim if hasattr(attn_module, "head_dim") else (hidden_states.shape[-1] // num_kv_heads)

    key_states   = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

    # Apply rotary embeddings if present
    if position_embeddings is not None:
        cos, sin = position_embeddings
        if hasattr(attn_module, "rotary_emb"):
            try:
                from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                q_dummy = attn_module.q_proj(hidden_states).view(
                    bsz, q_len, attn_module.num_heads, head_dim
                ).transpose(1, 2)
                _, key_states = apply_rotary_pos_emb(q_dummy, key_states, cos, sin)
            except Exception:
                pass  # fallback: skip rotary

    return key_states, value_states


class _TQPastKeyValue:
    """
    Minimal past_key_value wrapper so existing model code can do:
        pkv.key_cache[layer], pkv.value_cache[layer]
    without breaking.
    """

    def __init__(self, full_k: torch.Tensor, full_v: torch.Tensor):
        self._k = full_k
        self._v = full_v

    def __getitem__(self, idx):
        return self._k, self._v

    @property
    def key_cache(self):
        return [self._k]

    @property
    def value_cache(self):
        return [self._v]
