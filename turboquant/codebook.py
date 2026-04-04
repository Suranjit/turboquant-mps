"""
Codebook generation for TurboQuant using the Lloyd-Max algorithm.

After a random rotation, each coordinate of a unit-norm vector follows the
Beta-like distribution from Lemma 1 of the paper:

    f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2),  x ∈ [-1, 1]

In high dimensions this converges to N(0, 1/d).  The Lloyd-Max algorithm finds
the MSE-optimal scalar quantizer for this distribution.
"""

import numpy as np
from scipy import integrate
from scipy.special import gamma
import warnings

# Cache of precomputed codebooks keyed by (d, b)
_CODEBOOK_CACHE: dict[tuple[int, int], np.ndarray] = {}


def beta_pdf(x: np.ndarray, d: int) -> np.ndarray:
    """
    Marginal PDF of a single coordinate of a uniformly distributed point on
    the unit (d-1)-sphere S^{d-1}.

    Args:
        x: Points in [-1, 1].
        d: Ambient dimension.

    Returns:
        PDF values at x.
    """
    x = np.asarray(x, dtype=float)
    if d <= 2:
        # d=2: arcsine distribution; d=1: degenerate (±1)
        # Clip to avoid divide-by-zero; practically unused for LLMs
        return 1.0 / (np.pi * np.sqrt(np.maximum(1 - x**2, 1e-12)))
    if d == 3:
        # Uniform on [-1, 1]
        return np.full_like(x, 0.5)
    normalizer = gamma(d / 2) / (np.sqrt(np.pi) * gamma((d - 1) / 2))
    exponent = (d - 3) / 2
    density = normalizer * np.maximum(1.0 - x**2, 0.0) ** exponent
    return density


def _integrate_beta(func, a: float, b: float, d: int) -> float:
    """Numerical integration of func * beta_pdf over [a, b]."""
    if a >= b:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result, _ = integrate.quad(
            lambda x: func(x) * beta_pdf(x, d),
            a,
            b,
            limit=200,
            epsabs=1e-12,
            epsrel=1e-10,
        )
    return result


def lloyd_max(d: int, b: int, n_iter: int = 300, tol: float = 1e-10) -> np.ndarray:
    """
    Lloyd-Max algorithm: find the 2^b centroids that minimise MSE for the
    Beta coordinate distribution in dimension d.

    Args:
        d: Dimension of the ambient space (determines the Beta distribution).
        b: Bit-width (number of bits per coordinate).
        n_iter: Maximum number of Lloyd-Max iterations.
        tol: Convergence tolerance on centroid movement.

    Returns:
        Sorted array of 2^b centroids in [-1, 1].
    """
    n_levels = 2**b

    # Initialise centroids as quantiles of the Beta distribution.
    # A uniform spacing is fine for convergence; quantiles would be faster.
    centroids = np.linspace(-0.9, 0.9, n_levels)

    for iteration in range(n_iter):
        # Voronoi boundaries: midpoints between consecutive centroids, plus endpoints
        boundaries = np.empty(n_levels + 1)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        boundaries[1:-1] = (centroids[:-1] + centroids[1:]) / 2.0

        new_centroids = np.empty(n_levels)
        for i in range(n_levels):
            a, b_bound = boundaries[i], boundaries[i + 1]
            # Conditional mean: E[x | x ∈ [a, b]] = ∫ x f(x) dx / ∫ f(x) dx
            numerator = _integrate_beta(lambda x: x, a, b_bound, d)
            denominator = _integrate_beta(lambda x: 1.0, a, b_bound, d)
            new_centroids[i] = numerator / denominator if denominator > 1e-15 else 0.0

        shift = np.max(np.abs(new_centroids - centroids))
        centroids = new_centroids
        if shift < tol:
            break

    return np.sort(centroids)


def get_codebook(d: int, b: int) -> np.ndarray:
    """
    Return the Lloyd-Max codebook for dimension d and bit-width b.
    Results are cached in memory so each (d, b) pair is computed only once.

    Args:
        d: Ambient dimension.
        b: Bits per coordinate.

    Returns:
        Sorted array of 2^b centroids.
    """
    key = (d, b)
    if key not in _CODEBOOK_CACHE:
        _CODEBOOK_CACHE[key] = lloyd_max(d, b)
    return _CODEBOOK_CACHE[key]


def mse_cost(centroids: np.ndarray, d: int) -> float:
    """
    Compute the MSE distortion C(f_X, b) for a given codebook and dimension.
    Useful for validating that the codebook matches the paper's bounds.

    Args:
        centroids: Sorted array of centroids.
        d: Ambient dimension.

    Returns:
        MSE per coordinate (d · this = total D_mse).
    """
    n_levels = len(centroids)
    boundaries = np.empty(n_levels + 1)
    boundaries[0] = -1.0
    boundaries[-1] = 1.0
    boundaries[1:-1] = (centroids[:-1] + centroids[1:]) / 2.0

    total = 0.0
    for i in range(n_levels):
        a, b_bound = boundaries[i], boundaries[i + 1]
        c = centroids[i]
        cost = _integrate_beta(lambda x, c=c: (x - c) ** 2, a, b_bound, d)
        total += cost
    return total
