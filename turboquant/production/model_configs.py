"""
Llama model configuration registry for TurboQuant.

Covers TinyLlama-1.1B, Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B,
Llama-3.1-70B, and the Qwen2 family (commonly used via Ollama on Apple Silicon).

Key fields used by TurboQuant:
  - head_dim       : d per KV head  (hidden_size // num_attention_heads)
  - num_kv_heads   : GQA KV heads   (may be < num_attention_heads)
  - num_layers     : number of transformer blocks
  - context_length : native training context
  - fp16_kv_gb     : memory required for full-precision KV cache at max context

The KV cache memory for a model is:
    2  (K + V)
  × num_layers
  × num_kv_heads
  × context_length
  × head_dim
  × 2 bytes (float16)
= bytes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class LlamaVariant:
    """Static configuration for one Llama model variant."""
    name: str
    hf_model_id: str              # HuggingFace model identifier
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int             # GQA: < num_attention_heads when grouped
    num_hidden_layers: int
    context_length: int           # max position embeddings / rope scaling
    vocab_size: int
    intermediate_size: int        # feed-forward hidden size

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def gqa_ratio(self) -> int:
        """Query heads per KV head (1 = MHA, >1 = GQA)."""
        return self.num_attention_heads // self.num_kv_heads

    def fp16_kv_gb(self, seq_len: Optional[int] = None, batch_size: int = 1) -> float:
        """
        Full float16 KV cache memory in GB for the given sequence length.
        Uses max context_length if seq_len is None.
        """
        T = seq_len or self.context_length
        bytes_ = (
            2  # K + V
            * self.num_hidden_layers
            * self.num_kv_heads
            * T
            * self.head_dim
            * 2  # float16
            * batch_size
        )
        return bytes_ / (1024 ** 3)

    def tq_kv_gb(self, bits: int, seq_len: Optional[int] = None, batch_size: int = 1) -> float:
        """
        TurboQuant-compressed KV cache memory in GB.

        Prod variant (bits >= 2):
          bytes_per_token_head = ceil((b-1)*d/8) + ceil(d/8) + 2 + 4
        MSE variant (bits == 1):
          bytes_per_token_head = ceil(1*d/8) + 2
        """
        import math
        T = seq_len or self.context_length
        d = self.head_dim
        if bits >= 2:
            b_mse = bits - 1
            pb = _storage_bits(b_mse)
            bph = math.ceil(pb * d / 8) + math.ceil(d / 8) + 2 + 4
        else:
            bph = math.ceil(d / 8) + 2
        bytes_ = (
            2  # K + V
            * self.num_hidden_layers
            * self.num_kv_heads
            * T
            * bph
            * batch_size
        )
        return bytes_ / (1024 ** 3)

    def compression_ratio(self, bits: int) -> float:
        """Theoretical memory reduction vs float16."""
        fp = self.fp16_kv_gb()
        tq = self.tq_kv_gb(bits)
        return fp / tq if tq > 0 else float("inf")

    def max_context_at_memory(self, bits: int, available_gb: float, batch_size: int = 1) -> int:
        """
        Maximum context length achievable within available_gb of KV cache memory.
        """
        import math
        d = self.head_dim
        if bits >= 2:
            b_mse = bits - 1
            pb = _storage_bits(b_mse)
            bph = math.ceil(pb * d / 8) + math.ceil(d / 8) + 2 + 4
        else:
            bph = math.ceil(d / 8) + 2

        bytes_available = available_gb * (1024 ** 3)
        per_token = 2 * self.num_hidden_layers * self.num_kv_heads * bph * batch_size
        return int(bytes_available // per_token)

    def __str__(self) -> str:
        return (
            f"{self.name} | d={self.head_dim} | "
            f"{self.num_kv_heads} KV heads (GQA {self.gqa_ratio}x) | "
            f"{self.num_hidden_layers} layers | ctx={self.context_length:,}"
        )


def _storage_bits(b: int) -> int:
    """Mirror of mps_quantizer._storage_bits (avoid circular import)."""
    if b <= 1: return 1
    if b <= 2: return 2
    if b <= 4: return 4
    return 8


# ---------------------------------------------------------------------------
# Pre-defined model configurations
# ---------------------------------------------------------------------------

SUPPORTED_MODELS: Dict[str, LlamaVariant] = {
    # ---- TinyLlama --------------------------------------------------------
    "tinyllama-1.1b": LlamaVariant(
        name="TinyLlama-1.1B",
        hf_model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        hidden_size=2048,
        num_attention_heads=32,
        num_kv_heads=4,            # GQA 8x
        num_hidden_layers=22,
        context_length=2048,
        vocab_size=32000,
        intermediate_size=5632,
    ),
    # ---- Llama 3.2 1B -----------------------------------------------------
    "llama-3.2-1b": LlamaVariant(
        name="Llama-3.2-1B",
        hf_model_id="meta-llama/Llama-3.2-1B-Instruct",
        hidden_size=2048,
        num_attention_heads=32,
        num_kv_heads=8,            # GQA 4x
        num_hidden_layers=16,
        context_length=131072,
        vocab_size=128256,
        intermediate_size=8192,
    ),
    # ---- Llama 3.2 3B -----------------------------------------------------
    "llama-3.2-3b": LlamaVariant(
        name="Llama-3.2-3B",
        hf_model_id="meta-llama/Llama-3.2-3B-Instruct",
        hidden_size=3072,
        num_attention_heads=24,
        num_kv_heads=8,            # GQA 3x
        num_hidden_layers=28,
        context_length=131072,
        vocab_size=128256,
        intermediate_size=8192,
    ),
    # ---- Llama 3.1 8B -----------------------------------------------------
    "llama-3.1-8b": LlamaVariant(
        name="Llama-3.1-8B",
        hf_model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=8,            # GQA 4x
        num_hidden_layers=32,
        context_length=131072,
        vocab_size=128256,
        intermediate_size=14336,
    ),
    # ---- Llama 3.1 70B ----------------------------------------------------
    "llama-3.1-70b": LlamaVariant(
        name="Llama-3.1-70B",
        hf_model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
        hidden_size=8192,
        num_attention_heads=64,
        num_kv_heads=8,            # GQA 8x
        num_hidden_layers=80,
        context_length=131072,
        vocab_size=128256,
        intermediate_size=28672,
    ),
    # ---- Llama 3 8B (older, different head count) -------------------------
    "llama-3-8b": LlamaVariant(
        name="Llama-3-8B",
        hf_model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=8,
        num_hidden_layers=32,
        context_length=8192,
        vocab_size=128256,
        intermediate_size=14336,
    ),
    # ---- Qwen2.5 0.5B (very small, good for testing) ----------------------
    "qwen2.5-0.5b": LlamaVariant(
        name="Qwen2.5-0.5B",
        hf_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hidden_size=896,
        num_attention_heads=14,
        num_kv_heads=2,            # GQA 7x
        num_hidden_layers=24,
        context_length=32768,
        vocab_size=151936,
        intermediate_size=4864,
    ),
    # ---- Qwen2.5 7B -------------------------------------------------------
    "qwen2.5-7b": LlamaVariant(
        name="Qwen2.5-7B",
        hf_model_id="Qwen/Qwen2.5-7B-Instruct",
        hidden_size=3584,
        num_attention_heads=28,
        num_kv_heads=4,            # GQA 7x
        num_hidden_layers=28,
        context_length=131072,
        vocab_size=151936,
        intermediate_size=18944,
    ),
}


def get_model_config(model_key: str) -> LlamaVariant:
    """
    Look up a model config by key (case-insensitive, hyphens/underscores interchangeable).

    Examples::
        get_model_config("llama-3.2-3b")
        get_model_config("TinyLlama-1.1B")
        get_model_config("llama_3_1_8b")
    """
    normalised = model_key.lower().replace("_", "-")
    if normalised in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[normalised]
    # Fuzzy: check if key is a substring of any known key
    matches = [k for k in SUPPORTED_MODELS if normalised in k or k in normalised]
    if len(matches) == 1:
        return SUPPORTED_MODELS[matches[0]]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous model key '{model_key}'. Matches: {matches}"
        )
    raise KeyError(
        f"Unknown model '{model_key}'. "
        f"Supported: {list(SUPPORTED_MODELS.keys())}"
    )


def auto_detect_config(hf_config) -> Optional[LlamaVariant]:
    """
    Attempt to match a loaded HuggingFace config to a known LlamaVariant.
    Falls back to constructing a LlamaVariant dynamically.
    """
    model_type = getattr(hf_config, "model_type", "")
    h = getattr(hf_config, "hidden_size", 0)
    nh = getattr(hf_config, "num_attention_heads", 0)
    nkv = getattr(hf_config, "num_key_value_heads", nh)
    nl = getattr(hf_model_config := hf_config, "num_hidden_layers", 0)
    ctx = getattr(hf_config, "max_position_embeddings", 4096)
    vs = getattr(hf_config, "vocab_size", 32000)
    inter = getattr(hf_config, "intermediate_size", h * 4)
    hf_id = getattr(hf_config, "_name_or_path", "unknown")

    # Try exact match first
    for variant in SUPPORTED_MODELS.values():
        if (variant.hidden_size == h
                and variant.num_attention_heads == nh
                and variant.num_kv_heads == nkv
                and variant.num_hidden_layers == nl):
            return variant

    # Construct a dynamic variant
    if h > 0 and nh > 0:
        return LlamaVariant(
            name=f"Custom-{model_type}-{h}",
            hf_model_id=hf_id,
            hidden_size=h,
            num_attention_heads=nh,
            num_kv_heads=nkv,
            num_hidden_layers=nl,
            context_length=ctx,
            vocab_size=vs,
            intermediate_size=inter,
        )
    return None


def print_memory_table(bits_list: list[int] = (1, 2, 3, 4)) -> None:
    """
    Print a comparison table of KV cache memory for all supported models.
    """
    cols = ["Model", "FP16 (1K ctx)", "FP16 (32K ctx)"] + [
        f"TQ-{b}b (32K)" for b in bits_list
    ] + [f"Ratio ({bits_list[-1]}b)"]

    header = " | ".join(f"{c:<22}" for c in cols)
    print(header)
    print("-" * len(header))

    for key, cfg in SUPPORTED_MODELS.items():
        row = [
            f"{cfg.name:<22}",
            f"{cfg.fp16_kv_gb(1024):.3f} GB          ",
            f"{cfg.fp16_kv_gb(32768):.2f} GB         ",
        ]
        for b in bits_list:
            row.append(f"{cfg.tq_kv_gb(b, 32768):.3f} GB          ")
        row.append(f"{cfg.compression_ratio(bits_list[-1]):.1f}x           ")
        print(" | ".join(row))
