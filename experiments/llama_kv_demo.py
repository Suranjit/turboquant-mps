"""
Experiment 2: TurboQuant KV Cache on a Llama Model (Mac-friendly).

Run:
    # Quick test (no HF token needed):
    python experiments/llama_kv_demo.py

    # With Llama 3.2 (requires HF access):
    HF_TOKEN=hf_xxx python experiments/llama_kv_demo.py \\
        --model meta-llama/Llama-3.2-1B-Instruct --bits 2 3

    # Custom options:
    python experiments/llama_kv_demo.py --bits 2 3 --modes prod mse \\
        --prompt "Explain quantum entanglement"
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import argparse
import textwrap
import math
import torch
import numpy as np
from typing import Optional

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("ERROR: transformers not installed. Run: pip install transformers accelerate")
    sys.exit(1)

from turboquant.kv_cache import TurboQuantDynamicCache
from turboquant import TurboQuantMSE, TurboQuantProd


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: str):
    print(f"Loading {model_name}  (device={device})")
    token = os.environ.get("HF_TOKEN", None)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device,
        token=token,
        low_cpu_mem_usage=True,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  {n_params:.2f}B parameters loaded")
    return model, tokenizer


def get_head_dim(model) -> int:
    cfg = model.config
    num_heads = getattr(cfg, "num_attention_heads", 32)
    hidden_size = getattr(cfg, "hidden_size", 4096)
    return hidden_size // num_heads


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def make_prompt(tokenizer, text: str) -> str:
    """Apply chat template if available, else return text as-is."""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": text}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return text


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str,
    past_key_values=None,
) -> tuple[str, float, object]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        t0 = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,       # greedy — deterministic for comparison
            past_key_values=past_key_values,
            use_cache=True,
        )
        elapsed = time.time() - t0
    input_len = inputs["input_ids"].shape[1]
    text = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)
    return text, elapsed, past_key_values


def word_overlap(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    return len(wa & wb) / max(len(wa), 1)


def wrap(text: str, width: int = 70) -> str:
    return "\n".join(textwrap.fill(ln, width) for ln in text.split("\n"))


def section(title: str, width: int = 68):
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def run_demo(args):
    # Device selection
    if torch.backends.mps.is_available() and not args.cpu:
        device = "mps"
        print("Apple Silicon MPS detected")
    elif torch.cuda.is_available() and not args.cpu:
        device = "cuda"
    else:
        device = "cpu"

    model, tokenizer = load_model(args.model, device)
    head_dim = get_head_dim(model)
    print(f"  head_dim = {head_dim}")

    prompt = make_prompt(tokenizer, args.prompt)

    section("Prompt")
    print(wrap(args.prompt))

    results = {}

    # ---- Full-precision baseline ----
    section("Full-Precision KV Cache (baseline)")
    text_fp, t_fp, _ = generate(model, tokenizer, prompt, args.max_new_tokens, device)
    print(wrap(text_fp))
    print(f"\n  Time: {t_fp:.2f}s  ({args.max_new_tokens/t_fp:.1f} tok/s)")
    results["fp"] = {"label": "Full Precision (FP16)", "text": text_fp, "time": t_fp}

    # ---- TurboQuant variants ----
    for b in args.bits:
        for mode in args.modes:
            label = f"TurboQuant_{mode} @ {b}-bit"
            section(label)

            cache = TurboQuantDynamicCache(head_dim=head_dim, bits=b, mode=mode, seed=42)
            text_tq, t_tq, cache = generate(
                model, tokenizer, prompt, args.max_new_tokens, device,
                past_key_values=cache,
            )
            print(wrap(text_tq))

            ratio  = cache.compression_ratio()
            cb     = cache.compressed_bytes()
            fp_b   = cache.fp16_bytes()
            olap   = word_overlap(text_fp, text_tq)

            print(f"\n  Time: {t_tq:.2f}s  ({args.max_new_tokens/t_tq:.1f} tok/s)")
            print(f"  KV cache: {cb/1024:.0f} KB compressed  vs  {fp_b/1024:.0f} KB FP16  "
                  f"→ {ratio:.1f}x compression")
            print(f"  Word overlap with baseline: {olap*100:.0f}%")

            key = f"{mode}_{b}b"
            results[key] = {
                "label": label, "text": text_tq, "time": t_tq,
                "bits": b, "mode": mode, "ratio": ratio, "overlap": olap,
                "cb": cb, "fp_b": fp_b,
            }

    # ---- Summary table ----
    section("Summary")
    fmt = "  {:<32} {:>5}  {:>8}  {:>13}  {:>12}"
    print(fmt.format("Method", "Bits", "Time(s)", "Compression", "Word Overlap"))
    print("  " + "-" * 75)
    r = results["fp"]
    print(fmt.format(r["label"], "16", f"{r['time']:.1f}", "1.0x (ref)", "100%"))
    for key, r in results.items():
        if key == "fp":
            continue
        ratio_s  = f"{r['ratio']:.1f}x"
        olap_s   = f"{r['overlap']*100:.0f}%"
        print(fmt.format(r["label"], str(r["bits"]), f"{r['time']:.1f}", ratio_s, olap_s))

    # ---- Theoretical distortion at this head_dim ----
    section("Quantization Distortion (head_dim vectors, unit-norm)")
    d = head_dim
    rng = np.random.default_rng(99)
    n = 2000
    X = rng.standard_normal((n, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Y = rng.standard_normal((n, d)).astype(np.float32)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)

    print(f"\n  d = {d}")
    hdr = "  {:<18} {:>3}  {:>10}  {:>10}  {:>12}"
    print(hdr.format("Method", "b", "D_mse", "D_prod", "Theory UB (mse)"))
    print("  " + "-" * 58)
    for b in args.bits:
        q_mse = TurboQuantMSE(d=d, b=b, seed=7)
        idx = q_mse.quant(X)
        X_hat = q_mse.dequant(idx)
        d_mse = float(np.mean(np.sum((X - X_hat)**2, axis=-1)))
        ub = math.sqrt(3 * math.pi) / 2 * 4**(-b)

        q_prod = TurboQuantProd(d=d, b=b, seed=8)
        idx2, qjl, gamma = q_prod.quant(X)
        ip_true = (X * Y).sum(-1)
        ip_est  = q_prod.inner_product(Y, idx2, qjl, gamma)
        d_prod  = float(np.mean((ip_true - ip_est)**2))

        print(hdr.format("TurboQuantMSE", b, f"{d_mse:.5f}", "—", f"{ub:.5f}"))
        print(hdr.format("TurboQuantProd", b, "—", f"{d_prod:.6f}", ""))

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TurboQuant KV cache demo on Llama")
    parser.add_argument(
        "--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)"
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Explain the key differences between classical and quantum computing, "
            "and describe three potential applications of quantum computers."
        ),
    )
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--modes", nargs="+", default=["prod"], choices=["mse", "prod"])
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    run_demo(args)
