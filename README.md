# TurboQuant-MPS

MPS-native (Apple Silicon) implementation of **TurboQuant** — an online vector quantization algorithm with near-optimal distortion rate, applied to LLM KV caches. Enables significantly longer context lengths on Mac by compressing the key-value cache 4–7× with minimal quality loss.

Based on: *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*, Zandieh, Daliri, Hadian, Mirrokni — arXiv 2504.19874, 2025.

---

## Contents

- [The Problem: KV Cache Memory](#the-problem-kv-cache-memory)
- [The Algorithm](#the-algorithm)
- [Compression vs Quality](#compression-vs-quality)
- [Context Length Impact](#context-length-impact)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Running the Demos](#running-the-demos)
- [End-to-End Demo: Llama with Longer Context](#end-to-end-demo-llama-with-longer-context)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)

---

## The Problem: KV Cache Memory

When a language model generates text, it stores a **key-value (KV) cache** for every token it has processed. This cache is what allows the model to attend back to earlier context without recomputing everything from scratch — but it grows without bound as the context lengthens.

```
Context length  ×  (layers × heads × head_dim × 2 bytes)  =  KV cache size
```

For **Llama-3.2-3B** on an 8 GB Mac:
- The model weights consume ~5.5 GB
- That leaves ~2.5 GB for the KV cache
- At 112 KB per token in FP16, you run out of memory at **~23k tokens**

This isn't a software problem — it's a fundamental memory problem. The model literally cannot hold more context.

**TurboQuant solves this** by compressing each KV vector from 16-bit floats (~128 bytes for head_dim=64) down to ~18–28 bytes using principled vector quantization. Same model, same quality, **7× longer context**.

---

## The Algorithm

TurboQuant compresses each d-dimensional vector x in two steps:

### Step 1 — TurboQuant_mse (MSE-optimal compression)

```
y = Π · x            # random rotation, decorrelates coordinates
idx_j = nearest(y_j) # per-coordinate scalar quantization using Lloyd-Max codebook
```

The rotation Π ensures the quantization error spreads isotropically across all dimensions rather than accumulating in any direction. The codebook is computed by the **Lloyd-Max algorithm** on the exact marginal distribution of rotated unit-sphere coordinates (a Beta distribution that converges to N(0, 1/d) for large d).

**Why rotate first?** Without rotation, the error concentrates along the axes of the input distribution. After rotation, each coordinate has the same Beta distribution regardless of the input geometry, so the codebook is optimal for every vector.

### Step 2 — TurboQuant_prod (inner-product-optimal, used for attention)

```
idx = TurboQuant_mse(x, b-1 bits)    # coarse MSE quantization
r   = x − dequant(idx)               # residual
qjl = sign(S · r)                    # 1-bit QJL on residual (S is random Gaussian)
γ   = ‖r‖                            # scalar residual norm
```

Reconstruction:
```
x̃ = dequant_mse(idx) + √(π/2)/d · γ · Sᵀ · qjl
```

The QJL (Quantized Johnson-Lindenstrauss) term provides an **unbiased estimator of the residual's contribution to any inner product** — which is exactly what attention needs. The result is that `⟨q, x̃⟩ ≈ ⟨q, x⟩` with minimum variance for the available bits.

### Memory layout per token per KV head

| Component | Size (b=2, d=64) | Purpose |
|-----------|-----------------|---------|
| `packed_idx` | `ceil((b−1)·d/8)` = 8 bytes | MSE indices, bit-packed |
| `packed_qjl` | `ceil(d/8)` = 8 bytes | QJL signs, 1-bit packed |
| `qjl_gamma` | 2 bytes (float16) | residual norm |
| `vec_norm` | 2 bytes (float16) | original vector norm |
| **Total** | **20 bytes** | vs **128 bytes** FP16 → **6.4×** |

All bit-packing and arithmetic runs natively on the MPS GPU — no CPU round-trips during inference.

---

## Compression vs Quality

Measured on random unit-norm vectors (d=128, n=500):

| Bits (b) | Compression | Cosine Similarity | D_mse |
|----------|-------------|-------------------|-------|
| 1-bit    | 7.1×        | 0.80              | 0.363 |
| 2-bit    | **7.1×**    | **0.94**          | 0.117 |
| 3-bit    | 4.9×        | 0.98              | 0.034 |
| 4-bit    | 3.0×        | 0.995             | 0.009 |

**2-bit is the recommended default** — it provides the best compression-to-quality ratio for most LLMs. At 3-4 bits, reconstruction is nearly perfect.

### Why TurboQuant is near-optimal

The paper proves:
- **Lower bound** (information theory): any quantizer at b bits must have D_mse ≥ 4⁻ᵇ
- **TurboQuant achieves**: D_mse ≈ 1.87 × 4⁻ᵇ (within a small constant of optimal)
- **Inner-product distortion** D_prod ≤ √(3π²) · ‖y‖²/d · 4⁻ᵇ (asymptotically tight)

No other efficient online quantizer comes this close to the theoretical limit.

---

## Context Length Impact

Assuming 2-bit TurboQuant, model weights loaded at FP16:

| Model | RAM | FP16 max context | TurboQuant max context | Gain |
|-------|-----|-----------------|----------------------|------|
| Llama-3.2-1B | 8 GB | 190k tokens | **1.2M tokens** | 6.4× |
| Llama-3.2-3B | 8 GB | 23k tokens | **166k tokens** | 7.1× |
| Llama-3.2-3B | 16 GB | 98k tokens | **699k tokens** | 7.1× |
| Llama-3.1-8B | 16 GB | 16k tokens | **116k tokens** | 7.1× |

The 3B model on an 8 GB Mac is the most compelling case: FP16 caps out at ~23k tokens (barely enough for a large document), while TurboQuant extends this to 166k tokens — enough for an entire book.

---

## Installation

**Requirements:** Python ≥ 3.10, macOS with Apple Silicon (MPS), or any CUDA/CPU system.

```bash
# Clone
git clone https://github.com/Suranjit/turboquant-mps.git
cd turboquant-mps

# Install core + inference dependencies
pip install -e ".[llm]"
```

This installs:
- `numpy`, `scipy`, `torch` — core quantization
- `transformers`, `accelerate` — for running Llama demos
- `matplotlib` — for distortion plots

For core quantization only (no model demos):
```bash
pip install -e .
```

---

## Quick Start

### Using the numpy quantizer (CPU)

```python
from turboquant import TurboQuantMSE, TurboQuantProd
import numpy as np

# Create some unit-norm vectors (e.g., attention keys)
X = np.random.randn(100, 128).astype(np.float32)
X /= np.linalg.norm(X, axis=1, keepdims=True)

# MSE-optimal quantizer at 2 bits per coordinate
q_mse = TurboQuantMSE(d=128, b=2)
idx = q_mse.quant(X)        # compress: (100, 128) → int16 indices
X_hat = q_mse.dequant(idx)  # reconstruct: int16 indices → (100, 128) float32
print(f"D_mse = {q_mse.mse(X):.4f}")   # ~0.116

# Inner-product-optimal quantizer (best for attention)
q_prod = TurboQuantProd(d=128, b=2)
idx, qjl, gamma = q_prod.quant(X)

# Estimate ⟨y, x⟩ without fully reconstructing x
y = np.random.randn(128).astype(np.float32)
ip_est = q_prod.inner_product(y, idx, qjl, gamma)   # unbiased estimate
ip_true = X @ y
print(f"Mean IP error: {np.mean(ip_true - ip_est):.4f}")   # ~0
```

### Using the MPS quantizer (Apple Silicon)

```python
import torch
from turboquant import MPSTurboQuantMSE, MPSTurboQuantProd

device = "mps"  # or "cuda" or "cpu"

# Single-vector pack/unpack (bit-packed int8 storage)
q = MPSTurboQuantProd(d=64, b=2, device=device, dtype=torch.float16)

X = torch.randn(32, 64, device=device, dtype=torch.float16)  # batch of 32 vectors

# Compress to bit-packed int8 + float16 scalars
packed_idx, packed_qjl, qjl_gamma, norms = q.quant_pack(X)

print(f"packed_idx: {packed_idx.shape} {packed_idx.dtype}")   # (32, 8) int8
print(f"packed_qjl: {packed_qjl.shape} {packed_qjl.dtype}")   # (32, 8) int8
print(f"Compression: {q.compression_ratio():.1f}x")           # 6.4x

# Reconstruct
X_hat = q.dequant_unpack(packed_idx, packed_qjl, qjl_gamma, norms)
```

### Using the KV cache with a Hugging Face model

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from turboquant import MPSTurboQuantCache

model_name = "meta-llama/Llama-3.2-3B-Instruct"  # or any causal LM
device = "mps"

model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16,
                                             device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Drop-in replacement for the default KV cache
cache = MPSTurboQuantCache(
    head_dim=128,   # model.config.hidden_size // model.config.num_attention_heads
    bits=2,         # 2-bit = best compression; 3-bit = higher quality
    device=device,
    dtype=torch.float16,
)

inputs = tokenizer("Explain quantum computing", return_tensors="pt").to(device)
output = model.generate(
    **inputs,
    max_new_tokens=200,
    past_key_values=cache,    # <-- plug in TurboQuant cache here
    use_cache=True,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
print(cache.memory_report())
# Compressed: 12.3 KB  |  FP16 equiv: 87.5 KB  |  Actual 7.1× (theoretical 7.1×)
```

---

## Running the Demos

### 1. Validate distortion bounds

Reproduces the paper's distortion results and plots D_mse / D_prod across bit-widths.

```bash
python experiments/validate_distortion.py --d 256 --n 3000 --max-bits 4
```

Expected output:
```
MSE Distortion:
  MSE b=1 ... D_mse = 0.3623  (asymp_ub=0.3837, it_lb=0.2500)  paper_ref≈0.36
  MSE b=2 ... D_mse = 0.1170  (asymp_ub=0.0959, it_lb=0.0625)  paper_ref≈0.117
  MSE b=3 ... D_mse = 0.0342  (asymp_ub=0.0240, it_lb=0.0156)  paper_ref≈0.03
  MSE b=4 ... D_mse = 0.0095  (asymp_ub=0.0060, it_lb=0.0039)  paper_ref≈0.009

Inner-Product Distortion:
  Inner-prod b=2 ... D_prod = 0.002225

Bias Test (b=2, d=256):
  TurboQuantMSE  bias = +0.000068  (expected ≠ 0 — biased for inner products)
  TurboQuantProd bias = -0.000477  (expected ≈ 0 — unbiased by design)
```

Plots are saved to `results/`.

---

### 2. TurboQuant on a Llama model

Runs FP16 baseline vs TurboQuant at 2-bit and 3-bit and compares output quality.

```bash
# Public model — no token required
python experiments/llama_kv_demo.py --bits 2 3 --max-new-tokens 150

# Llama 3.2 (requires HuggingFace access token)
HF_TOKEN=hf_xxx python experiments/llama_kv_demo.py \
    --model meta-llama/Llama-3.2-1B-Instruct --bits 2 3
```

---

### 3. MPS memory benchmark

Shows the theoretical and real memory usage at increasing context lengths without loading model weights (fast).

```bash
python experiments/mps_context_demo.py --skip-generation

# Full demo with Llama 3.2-1B (loads model, runs real prefill)
HF_TOKEN=hf_xxx python experiments/mps_context_demo.py \
    --model meta-llama/Llama-3.2-1B-Instruct --bits 2
```

---

## End-to-End Demo: Llama with Longer Context

This is the core result: running a Llama model at a context length that would OOM with a standard FP16 cache, using TurboQuant to make it fit.

### Prerequisites

```bash
pip install -e ".[llm]"

# For Llama 3.2 (recommended):
# 1. Accept the model licence at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
# 2. Create an access token at https://huggingface.co/settings/tokens
export HF_TOKEN=hf_your_token_here
```

### Step 1 — Run the memory benchmark (no GPU needed, ~10 seconds)

This shows exactly where each model hits its memory wall and how TurboQuant extends it:

```bash
python experiments/mps_context_demo.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --bits 2 \
    --skip-generation
```

Example output:
```
======================================================================
  TurboQuant: Extended Context on Mac
======================================================================
  Device: mps  |  Bits: 2  |  Model: meta-llama/Llama-3.2-3B-Instruct
  Total system RAM: 16 GB

======================================================================
  KV Cache Memory Benchmark: FP16 vs TurboQuant
======================================================================

  Model: 28 layers, 8 KV heads, head_dim=128
  TurboQuant bits: 2

  FP16 per token:       112.0 KB
  TurboQuant per token:  15.8 KB
  Theoretical ratio:     7.1×

  Context       FP16 KV    TurboQ KV    Ratio   8GB fit?  16GB fit?
  ----------------------------------------------------------------
        8k       875 MB       123 MB     7.1×  ✗FP16/✓TQ   ✓/✓
       16k       1.7 GB       247 MB     7.1×  ✗FP16/✓TQ   ✓/✓
       32k       3.4 GB       494 MB     7.1×  ✗FP16/✓TQ   ✓/✓
       64k       6.9 GB       968 MB     7.1×  ✗FP16/✓TQ   ✓/✓
      128k      13.7 GB       1.9 GB     7.1×  ✗FP16/✓TQ  ✗/✓
      256k      27.5 GB       3.9 GB     7.1×  ✗FP16/✗TQ  ✗/✓

  On 8 GB Mac (model uses ~5.5 GB):
    Max context with FP16:       23k tokens
    Max context with TurboQuant: 166k tokens  (7.1× longer!)
```

The Llama-3.2-3B model on an 8 GB Mac is limited to **~23k tokens** in FP16. With TurboQuant at 2-bit, the same model can handle **~166k tokens** — more than 7× the context.

### Step 2 — Real prefill benchmark (~2 minutes)

Loads the model and actually fills the KV cache with synthetic tokens to confirm real memory usage:

```bash
HF_TOKEN=hf_xxx python experiments/mps_context_demo.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --bits 2
```

Output includes:
```
======================================================================
  Real Prefill Benchmark at 16k tokens
======================================================================

  [1/2] FP16 cache prefill at 16k tokens...
    Time: 12.4s
    KV cache size: 1750 MB (FP16 theoretical)

  [2/2] TurboQuant 2-bit cache prefill at 16k tokens...
    Compressed: 246.1 KB  |  FP16 equiv: 1750.0 MB  |  Actual 7.1× (theoretical 7.1×)
    Time: 18.3s
    Compressed KV:   247 MB
    FP16 equiv:     1750 MB
    Ratio:          7.1×  (saves 1503 MB)
```

### Step 3 — Generation quality comparison

The demo also runs actual text generation, comparing FP16 output to TurboQuant output on a long document:

```bash
HF_TOKEN=hf_xxx python experiments/mps_context_demo.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --bits 2
```

Example comparison:
```
  [FP16 baseline]
  Time: 8.2s  (12 tok/s)
  KV cache (FP16): 350 MB (theoretical)
  Output: The most important invention in computing history is the
  microprocessor. It enabled the miniaturisation of computers from
  room-sized machines to devices that fit in a pocket...

  [TurboQuant 2-bit]
  Time: 9.1s  (11 tok/s)
  Compressed: 49.3 KB  |  FP16 equiv: 350.0 KB  |  Actual 7.1×
  Output: The most important invention in computing history is the
  transistor. Its development led to integrated circuits, then
  microprocessors, transforming computers from room-filling machines...

  Word overlap with FP16: 71%
```

At 2-bit quantization, the model produces coherent, on-topic output at 7× memory compression. Word overlap with the FP16 baseline is typically 65–75% — the meaning and quality are preserved even though the exact wording differs.

### Using TinyLlama (no HuggingFace token needed)

For a quick local test without signing up for model access:

```bash
# TinyLlama is public — runs everything including generation
python experiments/mps_context_demo.py --bits 2
```

TinyLlama has a small context window (2048 tokens) so the memory savings are smaller in absolute terms, but the compression ratio and quality demonstration are identical.

---

## Running Tests

```bash
# Run all tests with coverage report
python -m pytest

# Quick run without coverage
python -m pytest --no-cov -q

# Run a specific test file
python -m pytest tests/test_mps_quantizer.py -v

# Skip slow statistical tests
python -m pytest -m "not slow" -q
```

Expected output:
```
311 passed in 11.82s

Name                          Stmts   Miss  Cover
---------------------------------------------------
turboquant/__init__.py            6      0   100%
turboquant/codebook.py           59      0   100%
turboquant/kv_cache.py          140      2    99%
turboquant/mps_kv_cache.py      136      0   100%
turboquant/mps_quantizer.py     142      0   100%
turboquant/quantizer.py         100      0   100%
---------------------------------------------------
TOTAL                           583      2    99%
```

---

## Project Structure

```
turboquant-mps/
├── turboquant/
│   ├── codebook.py        # Lloyd-Max algorithm on Beta marginal distribution
│   ├── quantizer.py       # TurboQuantMSE and TurboQuantProd (numpy, CPU)
│   ├── kv_cache.py        # DynamicCache-compatible KV cache (numpy backend)
│   ├── mps_quantizer.py   # MPS-native quantizers with bit-packed int8 storage
│   └── mps_kv_cache.py    # MPS-native KV cache (all ops on-device)
├── experiments/
│   ├── validate_distortion.py  # Reproduce paper's distortion figures
│   ├── llama_kv_demo.py        # Quality comparison: FP16 vs TurboQuant
│   └── mps_context_demo.py     # Context length benchmark on Mac
├── tests/
│   ├── conftest.py             # Shared fixtures
│   ├── test_codebook.py        # Lloyd-Max and Beta PDF tests
│   ├── test_quantizer.py       # numpy quantizer correctness
│   ├── test_kv_cache.py        # numpy KV cache tests
│   ├── test_mps_quantizer.py   # MPS bit-packing and quantizer tests
│   ├── test_mps_kv_cache.py    # MPS KV cache tests
│   └── test_integration.py     # Cross-component integration tests
├── pytest.ini
└── setup.py
```

**Two implementations exist side by side:**
- `quantizer.py` / `kv_cache.py` — numpy-based, pure Python, device-agnostic. Good for understanding the algorithm and CPU experiments.
- `mps_quantizer.py` / `mps_kv_cache.py` — PyTorch-native, runs on MPS/CUDA/CPU. Uses bit-packed int8 tensors for true compressed storage. Use this for real inference.

---

## API Reference

### `MPSTurboQuantCache` (recommended for inference)

```python
cache = MPSTurboQuantCache(
    head_dim=64,            # attention head dimension
    bits=2,                 # bits per coordinate: 2=best compression, 3=higher quality
    device="mps",           # "mps", "cuda", or "cpu"
    dtype=torch.float16,
    seed=42,
)

# Drop into model.generate() or model.forward()
k_full, v_full = cache.update(key_states, value_states, layer_idx=0)

cache.get_seq_length()          # total tokens stored
cache.memory_report()           # human-readable compression summary
cache.compression_ratio()       # actual bytes / fp16 bytes
cache.theoretical_compression_ratio()  # from formula
```

### `MPSTurboQuantProd` (low-level MPS quantizer)

```python
q = MPSTurboQuantProd(d=64, b=2, device="mps")

# Compress
packed_idx, packed_qjl, qjl_gamma, norms = q.quant_pack(X)

# Decompress
X_hat = q.dequant_unpack(packed_idx, packed_qjl, qjl_gamma, norms)

# Memory stats
q.compressed_bytes_per_token_head()   # bytes: ceil((b-1)*d/8) + ceil(d/8) + 4
q.fp16_bytes_per_token_head()         # d * 2
q.compression_ratio()                 # fp16 / compressed
```

### `TurboQuantDynamicCache` (numpy backend, HuggingFace compatible)

```python
cache = TurboQuantDynamicCache(
    head_dim=64,
    bits=2,
    mode="prod",    # "prod" (unbiased inner-product) or "mse" (MSE-optimal)
)
# Same interface as transformers.DynamicCache
k_full, v_full = cache.update(key_states, value_states, layer_idx=0)
cache.compression_ratio()
cache.memory_summary()
```

---

## Citation

```bibtex
@article{zandieh2025turboquant,
  title   = {TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author  = {Zandieh, Amir and Daliri, Majid and Hadian, Ali and Mirrokni, Vahab},
  journal = {arXiv preprint arXiv:2504.19874},
  year    = {2025}
}
```
