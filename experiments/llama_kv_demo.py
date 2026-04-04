"""
Experiment 2: TurboQuant KV Cache on a Llama Model (Mac-friendly).

Demonstrates TurboQuant applied to the KV cache of a Llama model, showing:
  - How much memory the KV cache is compressed
  - That generation quality is maintained at 2-bit and 3-bit quantization
  - Side-by-side comparison of full-precision vs. TurboQuant outputs

Supported models (set via --model):
  - meta-llama/Llama-3.2-1B-Instruct  (default, ~2.5 GB, fits on 8 GB Mac)
  - meta-llama/Llama-3.2-3B-Instruct  (~6 GB)
  - meta-llama/Llama-3.1-8B-Instruct  (~16 GB, needs 32 GB Mac)

Requires HuggingFace access token for Llama models (set HF_TOKEN env var),
or use --model with a publicly accessible model.

Run:
    # Quick test with a public model:
    python experiments/llama_kv_demo.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

    # With Llama 3.2 (requires HF access):
    HF_TOKEN=hf_xxx python experiments/llama_kv_demo.py --model meta-llama/Llama-3.2-1B-Instruct

    # Custom prompt and bit-width:
    python experiments/llama_kv_demo.py --bits 2 3 --prompt "Explain quantum computing"
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import argparse
import textwrap
import torch
import numpy as np
from typing import Optional

# Transformers imports
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.cache_utils import DynamicCache
except ImportError:
    print("ERROR: transformers not installed. Run: pip install transformers")
    sys.exit(1)

from turboquant.kv_cache import TurboQuantKVCache


# ---------------------------------------------------------------------------
# TurboQuant cache wrapper compatible with transformers DynamicCache API
# ---------------------------------------------------------------------------

class TurboQuantDynamicCache(DynamicCache):
    """
    Subclass of DynamicCache that quantizes stored K/V tensors.

    At each .update() call we:
      1. Append new key/value states (standard DynamicCache behaviour)
      2. Re-quantize the full cache and store compressed representations
      3. Return dequantized tensors so the attention mechanism works normally
    """

    def __init__(self, head_dim: int, bits: int, mode: str = "prod", seed: int = 42):
        super().__init__()
        self._tq = TurboQuantKVCache(head_dim=head_dim, bits=bits, mode=mode, seed=seed)
        self.bits = bits
        self.mode = mode
        # Override internal storage with TurboQuant's
        self._tq_initialized = False

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ):
        """Quantize-aware cache update."""
        return self._tq.update(key_states, value_states, layer_idx, cache_kwargs)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._tq.get_seq_length(layer_idx)

    def get_max_length(self) -> Optional[int]:
        return None

    @property
    def seen_tokens(self) -> int:
        return self._tq._seen_tokens

    def compression_ratio(self) -> float:
        return self._tq.compression_ratio()

    def memory_bytes(self) -> dict:
        return self._tq.memory_bytes()


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: str):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    print(f"  Device: {device}")

    token = os.environ.get("HF_TOKEN", None)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use float16 for memory efficiency on Mac MPS; float32 fallback for CPU
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        token=token,
        low_cpu_mem_usage=True,
    )
    model.eval()

    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    return model, tokenizer


def get_head_dim(model) -> int:
    """Extract attention head dimension from model config."""
    cfg = model.config
    # Handle both standard and grouped-query attention
    num_heads = getattr(cfg, "num_attention_heads", 32)
    hidden_size = getattr(cfg, "hidden_size", 4096)
    return hidden_size // num_heads


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    device: str = "cpu",
    past_key_values=None,
    temperature: float = 0.0,
) -> tuple[str, float, any]:
    """
    Generate text with optional custom KV cache.

    Returns:
        (generated_text, time_seconds, cache_object)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        t0 = time.time()
        if temperature == 0.0:
            # Greedy decoding — deterministic, best for quality comparison
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                past_key_values=past_key_values,
                use_cache=True,
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                past_key_values=past_key_values,
                use_cache=True,
            )
        elapsed = time.time() - t0

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0, input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text, elapsed, past_key_values


def compute_perplexity(model, tokenizer, text: str, device: str) -> float:
    """Compute perplexity of a text string under the model."""
    enc = tokenizer(text, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    if input_ids.shape[1] < 2:
        return float("nan")
    with torch.no_grad():
        outputs = model(**enc, labels=input_ids)
    return float(torch.exp(outputs.loss))


def compare_outputs(text_fp: str, text_tq: str, n_words: int = 60) -> float:
    """Simple token-level overlap metric between two generated texts."""
    words_fp = set(text_fp.lower().split())
    words_tq = set(text_tq.lower().split())
    if not words_fp:
        return 0.0
    overlap = len(words_fp & words_tq) / len(words_fp)
    return overlap


def print_section(title: str, width: int = 70):
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def wrap_text(text: str, width: int = 70) -> str:
    return "\n".join(textwrap.fill(line, width) for line in text.split("\n"))


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def run_demo(args):
    # Determine device
    if torch.backends.mps.is_available() and not args.cpu:
        device = "mps"
        print("Apple Silicon MPS detected — using GPU acceleration")
    elif torch.cuda.is_available() and not args.cpu:
        device = "cuda"
    else:
        device = "cpu"
        print("Using CPU (this will be slow for large models)")

    # Load model
    model, tokenizer = load_model(args.model, device)
    head_dim = get_head_dim(model)
    print(f"  Attention head dim: {head_dim}")

    # Format prompt (chat-style if model supports it)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": args.prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted_prompt = args.prompt

    print_section("Prompt")
    print(wrap_text(args.prompt))

    results = {}

    # --- Full-precision baseline ---
    print_section("Full-precision KV Cache (baseline)")
    text_fp, t_fp, _ = generate_text(
        model, tokenizer, formatted_prompt,
        max_new_tokens=args.max_new_tokens, device=device
    )
    print(wrap_text(text_fp))
    print(f"\n  Time: {t_fp:.2f}s | Tokens/s: {args.max_new_tokens/t_fp:.1f}")
    results["full_precision"] = {"text": text_fp, "time": t_fp, "bits": 16}

    # --- TurboQuant at each requested bit-width ---
    for b in args.bits:
        for mode in args.modes:
            label = f"TurboQuant_{mode} @ {b}-bit"
            print_section(label)

            cache = TurboQuantDynamicCache(
                head_dim=head_dim, bits=b, mode=mode, seed=42
            )
            text_tq, t_tq, cache = generate_text(
                model, tokenizer, formatted_prompt,
                max_new_tokens=args.max_new_tokens,
                device=device,
                past_key_values=cache,
            )
            print(wrap_text(text_tq))

            ratio = cache.compression_ratio()
            mem = cache.memory_bytes()
            overlap = compare_outputs(text_fp, text_tq)

            print(f"\n  Time:               {t_tq:.2f}s | Tokens/s: {args.max_new_tokens/t_tq:.1f}")
            print(f"  Compression ratio:  {ratio:.2f}x  (vs FP16 KV cache)")
            if mem["compressed_bytes"] > 0:
                print(f"  Compressed size:    {mem['compressed_bytes']/1024:.1f} KB")
                print(f"  FP16 equivalent:    {mem['fp16_equivalent_bytes']/1024:.1f} KB")
            print(f"  Word overlap w/ FP: {overlap*100:.1f}%")

            results[f"{mode}_{b}bit"] = {
                "text": text_tq,
                "time": t_tq,
                "bits": b,
                "mode": mode,
                "compression_ratio": ratio,
                "word_overlap": overlap,
            }

    # --- Summary table ---
    print_section("Summary")
    print(f"{'Method':<30} {'Bits':>5} {'Time (s)':>10} {'Compression':>13} {'Word Overlap':>13}")
    print("-" * 75)
    print(f"{'Full Precision':<30} {'16':>5} {t_fp:>10.2f} {'1.0x':>13} {'100.0%':>13}")
    for key, r in results.items():
        if key == "full_precision":
            continue
        name = f"TurboQuant_{r['mode']} {r['bits']}-bit"
        ratio_str = f"{r.get('compression_ratio', 0):.1f}x"
        overlap_str = f"{r.get('word_overlap', 0)*100:.1f}%"
        print(f"  {name:<28} {r['bits']:>5} {r['time']:>10.2f} {ratio_str:>13} {overlap_str:>13}")

    # --- Quantization distortion on a mini synthetic dataset ---
    print_section("Quantization Distortion (theoretical validation)")
    d = head_dim
    rng = np.random.default_rng(99)
    n_test = 2000
    X_test = rng.standard_normal((n_test, d)).astype(np.float32)
    X_test /= np.linalg.norm(X_test, axis=1, keepdims=True)
    Y_test = rng.standard_normal((n_test, d)).astype(np.float32)
    Y_test /= np.linalg.norm(Y_test, axis=1, keepdims=True)

    import math
    print(f"\n  d = {d}  (Llama attention head dim)")
    print(f"\n  {'Method':<30} {'b':>3} {'D_mse':>10} {'D_prod':>10} {'Theory UB (MSE)':>16}")
    print("  " + "-" * 75)
    for b in args.bits:
        from turboquant import TurboQuantMSE, TurboQuantProd
        q_mse = TurboQuantMSE(d=d, b=b, seed=7)
        idx = q_mse.quant(X_test)
        X_hat = q_mse.dequant(idx)
        # D_mse = E[||x - x̃||²₂]
        d_mse = float(np.mean(np.sum((X_test - X_hat)**2, axis=-1)))
        theory_ub = math.sqrt(3 * math.pi) / 2 * 4**(-b)

        q_prod = TurboQuantProd(d=d, b=b, seed=8)
        idx2, qjl, gamma = q_prod.quant(X_test)
        ip_true = (X_test * Y_test).sum(-1)
        ip_est = q_prod.inner_product(Y_test, idx2, qjl, gamma)
        d_prod = float(np.mean((ip_true - ip_est)**2))

        print(f"  {'TurboQuantMSE':<30} {b:>3} {d_mse:>10.5f} {'—':>10} {theory_ub:>16.5f}")
        print(f"  {'TurboQuantProd':<30} {b:>3} {'—':>10} {d_prod:>10.6f} {'':>16}")

    print("\nDemo complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TurboQuant KV Cache demo on a Llama model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model name (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Explain the key differences between classical and quantum computing, "
            "and describe three potential applications of quantum computers."
        ),
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--bits", type=int, nargs="+", default=[2, 3],
        help="Bit-widths to test (default: 2 3)",
    )
    parser.add_argument(
        "--modes", nargs="+", default=["prod"],
        choices=["mse", "prod"],
        help="TurboQuant mode(s) to test: 'mse' and/or 'prod' (default: prod)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=150,
        help="Maximum tokens to generate (default: 150)",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU even if MPS/CUDA is available",
    )
    args = parser.parse_args()

    run_demo(args)
