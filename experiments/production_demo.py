"""
TurboQuant Production Demo — Metal Shader KV Cache

Demonstrates:
  1. Memory projections for all supported Llama model variants
  2. Extended context headroom with TurboQuant compression
  3. End-to-end inference with a real Llama model using the
     ProductionTurboQuantCache (Metal shader → MPS fallback)
  4. Compression ratio and reconstruction quality at each bit depth

Usage:
    python experiments/production_demo.py                        # memory table only
    python experiments/production_demo.py --model llama-3.2-1b  # run with real model
    python experiments/production_demo.py --model tinyllama-1.1b --bits 2
    python experiments/production_demo.py --model llama-3.2-3b --bits 3 --ctx 32768
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Optional

import numpy as np
import torch


def _fmt_bytes(b: int) -> str:
    if b >= 1024 ** 3:
        return f"{b / 1024 ** 3:.2f} GB"
    if b >= 1024 ** 2:
        return f"{b / 1024 ** 2:.1f} MB"
    return f"{b / 1024:.1f} KB"


# ---------------------------------------------------------------------------
# Section 1: Memory projection table
# ---------------------------------------------------------------------------

def show_memory_table(bits: int = 3) -> None:
    from turboquant.production.model_configs import SUPPORTED_MODELS

    col_w = 20
    print("\n" + "=" * 90)
    print("  TurboQuant KV Cache Memory Projections")
    print("=" * 90)

    header = (
        f"{'Model':<{col_w}} | "
        f"{'FP16 @ 4K ctx':>14} | "
        f"{'FP16 @ 32K ctx':>14} | "
        f"{'TQ @ 32K ctx':>12} | "
        f"{'Ratio':>8} | "
        f"{'Max ctx (8GB)':>13}"
    )
    print(header)
    print("-" * len(header))

    for key, cfg in SUPPORTED_MODELS.items():
        fp16_4k  = cfg.fp16_kv_gb(4096)
        fp16_32k = cfg.fp16_kv_gb(32768)
        tq_32k   = cfg.tq_kv_gb(bits, 32768)
        ratio    = cfg.compression_ratio(bits)
        max_ctx  = cfg.max_context_at_memory(bits, available_gb=8.0)

        print(
            f"{cfg.name:<{col_w}} | "
            f"{fp16_4k:>12.3f} GB | "
            f"{fp16_32k:>12.3f} GB | "
            f"{tq_32k:>10.3f} GB | "
            f"{ratio:>7.1f}x | "
            f"{max_ctx:>12,}"
        )

    print()


# ---------------------------------------------------------------------------
# Section 2: Standalone cache quality benchmark (no model download needed)
# ---------------------------------------------------------------------------

def benchmark_cache_quality(bits_range=(1, 2, 3, 4)) -> None:
    from turboquant.production import ProductionTurboQuantCache, get_model_config

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg    = get_model_config("llama-3.2-3b")

    print(f"\n{'='*70}")
    print(f"  Cache Quality Benchmark — Llama-3.2-3B config, device={device}")
    print(f"{'='*70}")
    print(f"  {'Bits':>5} | {'Ratio':>7} | {'MSE':>10} | {'Cos Sim':>9} | {'Compressed':>12} | {'Saved vs FP16':>13}")
    print("  " + "-" * 65)

    B, H, T, d = 1, cfg.num_kv_heads, 64, cfg.head_dim
    rng = torch.Generator(device="cpu").manual_seed(42)
    k_orig = torch.randn(B, H, T, d, generator=rng, dtype=torch.float16)
    v_orig = torch.randn(B, H, T, d, generator=rng, dtype=torch.float16)

    for b in bits_range:
        cache = ProductionTurboQuantCache.from_model_config(
            cfg, bits=b, device=device, dtype=torch.float16, seed=0, use_metal=True
        )
        k_rec, v_rec = cache.update(k_orig.to(device), v_orig.to(device), layer_idx=0)
        k_rec = k_rec.cpu().float()
        k_f   = k_orig.float()

        mse    = ((k_f - k_rec) ** 2).mean().item()
        norm_k = k_f.reshape(-1, d)
        norm_r = k_rec.reshape(-1, d)
        cos    = (
            (norm_k * norm_r).sum(-1)
            / (norm_k.norm(dim=-1) * norm_r.norm(dim=-1) + 1e-9)
        ).mean().item()

        cb = cache.compressed_bytes()
        fb = cache.fp16_bytes()
        ratio  = cache.compression_ratio()
        saving = (1 - cb / fb) * 100 if fb > 0 else 0

        print(f"  {b:>5} | {ratio:>7.1f}x | {mse:>10.5f} | {cos:>9.4f} | "
              f"{_fmt_bytes(cb):>12} | {saving:>12.1f}%")

    print()


# ---------------------------------------------------------------------------
# Section 3: Metal kernel availability check
# ---------------------------------------------------------------------------

def check_metal_status() -> None:
    from turboquant.metal.metal_lib import metal_available, compile_metal_lib

    print(f"\n{'='*50}")
    print("  Metal Shader Status")
    print(f"{'='*50}")

    pyobjc = metal_available()
    print(f"  PyObjC Metal bindings : {'available' if pyobjc else 'not available (using PyTorch MPS)'}")

    if pyobjc:
        lib_path = compile_metal_lib()
        if lib_path:
            print(f"  Metal library         : compiled → {lib_path.name}")
        else:
            print("  Metal library         : compilation failed (xcrun issue?)")
    else:
        print("  Metal library         : skipped (PyObjC not installed)")
        print("  Install with: pip install pyobjc-framework-Metal")

    mps_ok = torch.backends.mps.is_available()
    print(f"  PyTorch MPS           : {'available' if mps_ok else 'not available'}")
    print()


# ---------------------------------------------------------------------------
# Section 4: Extended context demo with real model
# ---------------------------------------------------------------------------

def run_extended_context_demo(
    model_key: str,
    bits: int = 3,
    target_ctx: Optional[int] = None,
    device_str: Optional[str] = None,
) -> None:
    print(f"\n{'='*70}")
    print(f"  Extended Context Demo — {model_key}, b={bits}")
    print(f"{'='*70}")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("  transformers not installed. Run: pip install transformers accelerate")
        return

    from turboquant.production import (
        ProductionTurboQuantCache,
        get_model_config,
        patch_llama_model,
    )

    cfg = get_model_config(model_key)
    device = device_str or ("mps" if torch.backends.mps.is_available() else "cpu")

    # Estimate memory requirements
    available_gb = 8.0   # conservative estimate for 16 GB Mac
    model_gb = _estimate_model_gb(cfg)
    kv_budget_gb = max(available_gb - model_gb - 1.0, 1.0)

    fp16_max_ctx = int(kv_budget_gb * 1024 ** 3 /
                       (2 * cfg.num_hidden_layers * cfg.num_kv_heads * cfg.head_dim * 2))
    tq_max_ctx   = cfg.max_context_at_memory(bits, kv_budget_gb)

    print(f"\n  Model         : {cfg.name}")
    print(f"  Model size    : ~{model_gb:.1f} GB (float16)")
    print(f"  KV budget     : {kv_budget_gb:.1f} GB")
    print(f"  Max ctx FP16  : {fp16_max_ctx:,} tokens")
    print(f"  Max ctx TQ-{bits}b : {tq_max_ctx:,} tokens  ({tq_max_ctx / max(fp16_max_ctx,1):.1f}x more)")

    target_ctx = target_ctx or min(tq_max_ctx // 2, 8192)
    print(f"  Demo ctx      : {target_ctx:,} tokens")

    print(f"\n  Loading {cfg.hf_model_id}...")
    t0 = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.hf_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.hf_model_id,
            torch_dtype=torch.float16,
            device_map=device,
        )
    except Exception as exc:
        print(f"\n  Could not load model: {exc}")
        print("  To run the demo, ensure you have model access and run:")
        print(f"    huggingface-cli login")
        print(f"    python experiments/production_demo.py --model {model_key}")
        return

    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    # Patch with TurboQuant cache
    cache = patch_llama_model(model, bits=bits, device=device, use_metal=True)

    # Build a long prompt to stress-test context
    prompt_text = (
        "Summarize the key points of the following very long document: "
        + " ".join([
            "The quick brown fox jumps over the lazy dog." * 10
            for _ in range(target_ctx // 100)
        ])[:target_ctx * 4]  # approx 4 chars/token
    )

    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                       max_length=target_ctx).to(device)
    input_len = inputs["input_ids"].shape[-1]
    print(f"\n  Prompt tokens : {input_len:,}")

    print("  Running inference with TurboQuant KV cache...")
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            past_key_values=cache,
        )
    gen_time = time.time() - t0
    new_tokens = outputs.shape[-1] - input_len

    decoded = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)
    print(f"  Generated     : {new_tokens} tokens in {gen_time:.2f}s ({new_tokens/gen_time:.1f} tok/s)")
    print(f"  Output        : {decoded[:200]}...")
    print()
    print(cache.memory_report())


# ---------------------------------------------------------------------------
# Section 5: Context throughput profiling (no model needed)
# ---------------------------------------------------------------------------

def profile_cache_throughput(bits: int = 3) -> None:
    """
    Profile cache update/retrieval throughput for realistic decode workloads.
    Runs prefill (bulk) + decode (single-token) simulation.
    """
    from turboquant.production import ProductionTurboQuantCache, get_model_config

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg    = get_model_config("llama-3.2-3b")

    cache = ProductionTurboQuantCache.from_model_config(
        cfg, bits=bits, device=device, dtype=torch.float16, seed=0, use_metal=True
    )

    B, H, d = 1, cfg.num_kv_heads, cfg.head_dim

    print(f"\n{'='*60}")
    print(f"  Throughput Profile — {cfg.name}, b={bits}, device={device}")
    print(f"{'='*60}")

    # ---- Prefill ----------------------------------------------------------
    T_prefill = 1024
    k = torch.randn(B, H, T_prefill, d, dtype=torch.float16).to(device)
    v = torch.randn(B, H, T_prefill, d, dtype=torch.float16).to(device)

    t0 = time.perf_counter()
    for layer_idx in range(cfg.num_hidden_layers):
        cache.update(k, v, layer_idx)
    dt = time.perf_counter() - t0
    prefill_tps = (T_prefill * cfg.num_hidden_layers) / dt
    print(f"  Prefill ({T_prefill} tokens × {cfg.num_hidden_layers} layers) : {dt*1000:.1f}ms  ({prefill_tps:.0f} token-layers/s)")

    # ---- Decode -----------------------------------------------------------
    T_decode = 100
    times = []
    for _ in range(T_decode):
        k1 = torch.randn(B, H, 1, d, dtype=torch.float16).to(device)
        v1 = torch.randn(B, H, 1, d, dtype=torch.float16).to(device)
        t0 = time.perf_counter()
        for layer_idx in range(cfg.num_hidden_layers):
            cache.update(k1, v1, layer_idx)
        times.append(time.perf_counter() - t0)

    avg_ms = np.mean(times) * 1000
    p95_ms = np.percentile(times, 95) * 1000
    decode_tps = cfg.num_hidden_layers / np.mean(times)
    print(f"  Decode   (1 token × {T_decode} steps × {cfg.num_hidden_layers} layers) :")
    print(f"    avg={avg_ms:.2f}ms/step  p95={p95_ms:.2f}ms  ({decode_tps:.0f} layer-updates/s)")

    # ---- Memory -----------------------------------------------------------
    print(f"\n  Memory after {cache.get_seq_length(0)} tokens/layer:")
    print(f"  " + "\n  ".join(cache.memory_report().split("\n")[1:]))
    print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_model_gb(cfg) -> float:
    """Rough float16 model weight estimate."""
    h = cfg.hidden_size
    nl = cfg.num_hidden_layers
    ffn = cfg.intermediate_size
    vs = cfg.vocab_size
    params = nl * (4 * h * h + 2 * h * ffn) + vs * h
    return params * 2 / 1024 ** 3


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TurboQuant Production Demo")
    parser.add_argument(
        "--model",
        default=None,
        help="Model key to run live demo (e.g. 'llama-3.2-1b', 'tinyllama-1.1b'). "
             "Omit to show memory table only.",
    )
    parser.add_argument("--bits", type=int, default=3, help="Quantization bits (1-4)")
    parser.add_argument("--ctx",  type=int, default=None, help="Target context length for demo")
    parser.add_argument("--device", default=None, help="Device: mps, cuda, cpu")
    parser.add_argument("--no-profile", action="store_true", help="Skip throughput profiling")
    args = parser.parse_args()

    # Always show these
    check_metal_status()
    show_memory_table(bits=args.bits)
    benchmark_cache_quality()

    if not args.no_profile:
        profile_cache_throughput(bits=args.bits)

    if args.model:
        run_extended_context_demo(
            model_key=args.model,
            bits=args.bits,
            target_ctx=args.ctx,
            device_str=args.device,
        )
    else:
        print("  Tip: run with --model llama-3.2-1b to run live inference demo")
        print("       run with --model llama-3.2-3b --bits 3 --ctx 16384 for extended context demo")


if __name__ == "__main__":
    main()
