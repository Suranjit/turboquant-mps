"""
Experiment 1: Empirical Validation of Distortion Bounds.

Reproduces Figure 3 and Table-like comparisons from Section 4.1 of the paper.

For a set of random unit-norm vectors:
  - Measures MSE distortion of TurboQuantMSE across bit-widths b=1..5
  - Measures inner-product distortion of TurboQuantProd across b=1..5
  - Plots against the theoretical upper and lower bounds from the paper

Run:
    python experiments/validate_distortion.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from turboquant import TurboQuantMSE, TurboQuantProd

# ---------------------------------------------------------------------------
# Constants from the paper (Theorem 1 and Theorem 2)
# ---------------------------------------------------------------------------

def mse_upper_bound(b: int) -> float:
    """
    Asymptotic upper bound √(3π)/2 · 4^{-b} (Theorem 1).
    Derived using the Gaussian approximation to the Beta marginal, valid as d→∞.
    May be exceeded for finite d at low bit-widths (b ≤ 2), but empirical values
    converge toward this bound as d grows.
    """
    return math.sqrt(3 * math.pi) / 2 * (4 ** (-b))

def mse_lower_bound(b: int) -> float:
    """Information-theoretic lower bound 4^{-b} (Theorem 3)."""
    return 4 ** (-b)

# Paper's refined MSE values for b=1,2,3,4 (from Theorem 1 discussion)
MSE_REFINED = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}

def iprod_upper_bound(b: int, d: int, norm_y: float = 1.0) -> float:
    """
    Asymptotic upper bound √(3π²)·||y||²/d · 4^{-b} (Theorem 2).
    Uses the Gaussian approximation; valid as d→∞. For finite d (especially
    d < 256) empirical D_prod may exceed this value at low bit-widths.
    """
    return math.sqrt(3 * math.pi**2) * norm_y**2 / d * (4 ** (-b))

def iprod_lower_bound(b: int, d: int, norm_y: float = 1.0) -> float:
    """Information-theoretic lower bound ||y||²/d · 4^{-b} (Theorem 3)."""
    return norm_y**2 / d * (4 ** (-b))

# Paper's refined inner-product distortion values for b=1,2,3,4
IPROD_REFINED_SCALE = {1: 1.57, 2: 0.56, 3: 0.18, 4: 0.047}


def run_mse_validation(d: int = 512, n: int = 5000, bits: range = range(1, 6)):
    """Measure empirical D_mse across bit-widths for dimension d."""
    rng = np.random.default_rng(0)
    # Random unit-norm vectors
    X = rng.standard_normal((n, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    results = {}
    for b in bits:
        print(f"  MSE b={b} ...", end=" ", flush=True)
        q = TurboQuantMSE(d=d, b=b, seed=b)
        idx = q.quant(X)
        X_hat = q.dequant(idx)
        # D_mse = E[||x - x̃||²₂]  (sum over coordinates, not mean)
        d_mse = float(np.mean(np.sum((X - X_hat) ** 2, axis=-1)))
        results[b] = d_mse
        ref = MSE_REFINED.get(b, None)
        ref_str = f"  paper_ref≈{ref}" if ref else ""
        print(f"D_mse = {d_mse:.4f}  (asymp_ub={mse_upper_bound(b):.4f}, it_lb={mse_lower_bound(b):.4f}){ref_str}")
    return results


def run_iprod_validation(d: int = 512, n: int = 5000, bits: range = range(1, 6)):
    """Measure empirical D_prod across bit-widths for dimension d."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Y = rng.standard_normal((n, d)).astype(np.float32)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)

    results = {}
    for b in bits:
        print(f"  Inner-prod b={b} ...", end=" ", flush=True)
        q = TurboQuantProd(d=d, b=b, seed=b)
        idx, qjl, gamma = q.quant(X)
        # For batched query comparison, pass single query per vector
        ip_true = (X * Y).sum(-1)
        ip_est = q.inner_product(Y, idx, qjl, gamma)
        d_prod = float(np.mean((ip_true - ip_est) ** 2))
        results[b] = d_prod
        print(
            f"D_prod = {d_prod:.6f}  "
            f"(asymp_ub={iprod_upper_bound(b, d):.6f}, it_lb={iprod_lower_bound(b, d):.6f})"
        )
    return results


def run_bias_test(d: int = 512, n: int = 10000, b: int = 2):
    """
    Show that TurboQuantMSE is BIASED for inner products while
    TurboQuantProd is UNBIASED (Section 4.1 / Figure 1).
    """
    rng = np.random.default_rng(2)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Y = rng.standard_normal((n, d)).astype(np.float32)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)

    ip_true = (X * Y).sum(-1)

    # TurboQuantMSE (biased)
    q_mse = TurboQuantMSE(d=d, b=b, seed=10)
    idx = q_mse.quant(X)
    X_hat = q_mse.dequant(idx)
    ip_mse = (X_hat * Y).sum(-1)
    bias_mse = float(np.mean(ip_mse - ip_true))

    # TurboQuantProd (unbiased)
    q_prod = TurboQuantProd(d=d, b=b, seed=11)
    idx2, qjl, gamma = q_prod.quant(X)
    ip_prod = q_prod.inner_product(Y, idx2, qjl, gamma)
    bias_prod = float(np.mean(ip_prod - ip_true))

    print(f"\n  Bias check at b={b}, d={d}:")
    print(f"    TurboQuantMSE  bias = {bias_mse:+.6f}  (expected ≠ 0 for small b)")
    print(f"    TurboQuantProd bias = {bias_prod:+.6f}  (expected ≈ 0)")
    return ip_true, ip_mse, ip_prod


def plot_results(mse_results: dict, iprod_results: dict, d: int, save_path: str = None):
    """Plot distortion vs. bit-width alongside theoretical bounds (like Figure 3)."""
    bits = sorted(mse_results.keys())
    b_arr = np.array(bits)

    mse_empirical = [mse_results[b] for b in bits]
    mse_upper = [mse_upper_bound(b) for b in bits]
    mse_lower = [mse_lower_bound(b) for b in bits]
    mse_refined = [MSE_REFINED.get(b, None) for b in bits]

    ip_empirical = [iprod_results[b] for b in bits]
    ip_upper = [iprod_upper_bound(b, d) for b in bits]
    ip_lower = [iprod_lower_bound(b, d) for b in bits]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"TurboQuant Distortion vs. Bit-width  (d={d})", fontsize=14)

    # --- MSE plot ---
    ax = axes[0]
    ax.semilogy(b_arr, mse_empirical, "b-o", label="TurboQuant$_{mse}$ (empirical)", linewidth=2)
    ax.semilogy(b_arr, mse_upper, "r--", label=f"Asymptotic UB (d→∞): $\\sqrt{{3\\pi}}/2 \\cdot 4^{{-b}}$", linewidth=1.5)
    ax.semilogy(b_arr, mse_lower, "g--", label="IT lower bound: $4^{-b}$", linewidth=1.5)
    refined_vals = [(b, v) for b, v in zip(bits, mse_refined) if v is not None]
    if refined_vals:
        ax.scatter([bv[0] for bv in refined_vals], [bv[1] for bv in refined_vals],
                   marker="x", color="purple", s=80, zorder=5, label="Paper refined values")
    ax.set_xlabel("Bit-width (b)")
    ax.set_ylabel("Mean Squared Error ($D_{mse}$)")
    ax.set_title("MSE Distortion")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Inner product plot ---
    ax = axes[1]
    ax.semilogy(b_arr, ip_empirical, "purple", marker="o", label="TurboQuant$_{prod}$ (empirical)", linewidth=2)
    ax.semilogy(b_arr, ip_upper, "r--", label=f"Asymptotic UB (d→∞): $\\sqrt{{3\\pi^2}}\\|y\\|^2/d \\cdot 4^{{-b}}$", linewidth=1.5)
    ax.semilogy(b_arr, ip_lower, "g--", label="IT lower bound: $\\|y\\|^2/d \\cdot 4^{{-b}}$", linewidth=1.5)
    ax.set_xlabel("Bit-width (b)")
    ax.set_ylabel("Inner-Product Distortion ($D_{prod}$)")
    ax.set_title("Inner-Product Distortion")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {save_path}")
    plt.show()


def plot_error_histograms(d: int = 512, n: int = 50000, save_path: str = None):
    """Reproduce Figure 1 from the paper: error distribution histograms per bit-width."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Y = rng.standard_normal((n, d)).astype(np.float32)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    ip_true = (X * Y).sum(-1)

    bits = [1, 2, 3, 4]
    colors_prod = ["#4DB6D0", "#4472C4", "#E07B54", "#3A7A3A"]
    colors_mse  = ["#7EC8D0", "#6699CC", "#F4A460", "#55A055"]

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    fig.suptitle("Inner-Product Error Distribution (Figure 1 from paper)", fontsize=13)

    for col, b in enumerate(bits):
        q_prod = TurboQuantProd(d=d, b=b, seed=b)
        idx, qjl, gamma = q_prod.quant(X)
        ip_prod = q_prod.inner_product(Y, idx, qjl, gamma)
        err_prod = ip_prod - ip_true

        q_mse = TurboQuantMSE(d=d, b=b, seed=b + 10)
        idx2 = q_mse.quant(X)
        X_hat = q_mse.dequant(idx2)
        ip_mse_est = (X_hat * Y).sum(-1)
        err_mse = ip_mse_est - ip_true

        ax_prod = axes[0, col]
        ax_prod.hist(err_prod, bins=60, color=colors_prod[col], alpha=0.8, density=True)
        ax_prod.axvline(0, color="black", linewidth=0.8)
        ax_prod.set_title(f"Bitwidth = {b}", fontsize=10)
        ax_prod.set_xlabel("Inner Product Distortion")
        if col == 0:
            ax_prod.set_ylabel("TurboQuant$_{prod}$\nDensity")

        ax_mse = axes[1, col]
        ax_mse.hist(err_mse, bins=60, color=colors_mse[col], alpha=0.8, density=True)
        ax_mse.axvline(0, color="black", linewidth=0.8)
        ax_mse.set_xlabel("Inner Product Distortion")
        if col == 0:
            ax_mse.set_ylabel("TurboQuant$_{mse}$\nDensity")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Histogram plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate TurboQuant distortion bounds")
    parser.add_argument("--d", type=int, default=256, help="Vector dimension")
    parser.add_argument("--n", type=int, default=3000, help="Number of test vectors")
    parser.add_argument("--max-bits", type=int, default=5, help="Maximum bit-width to test")
    parser.add_argument("--save-dir", type=str, default="results", help="Directory to save plots")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"TurboQuant Distortion Validation")
    print(f"  d={args.d}, n={args.n}, bits=1..{args.max_bits}")
    print(f"{'='*60}\n")

    print("MSE Distortion:")
    mse_results = run_mse_validation(d=args.d, n=args.n, bits=range(1, args.max_bits + 1))

    print("\nInner-Product Distortion:")
    iprod_results = run_iprod_validation(d=args.d, n=args.n, bits=range(1, args.max_bits + 1))

    print("\nBias Test:")
    ip_true, ip_mse, ip_prod = run_bias_test(d=args.d, n=min(args.n * 2, 10000), b=2)

    print("\nGenerating plots...")
    plot_results(
        mse_results, iprod_results, d=args.d,
        save_path=os.path.join(args.save_dir, "distortion_bounds.png")
    )
    plot_error_histograms(
        d=args.d, n=min(args.n * 10, 30000),
        save_path=os.path.join(args.save_dir, "error_histograms.png")
    )
