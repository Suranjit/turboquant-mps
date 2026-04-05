"""
Tests for turboquant/codebook.py

Covers:
  - beta_pdf: edge cases d=2, d=3, d=4+; normalisation; values
  - lloyd_max: output shape, sorted order, convergence, codebook symmetry
  - get_codebook: caching, determinism
  - mse_cost: non-negative, decreases with more bits
"""
import numpy as np
import pytest
from scipy import integrate

from turboquant.codebook import beta_pdf, get_codebook, lloyd_max, mse_cost


# ---------------------------------------------------------------------------
# beta_pdf
# ---------------------------------------------------------------------------

class TestBetaPdf:
    def test_d2_arcsine_shape(self):
        x = np.linspace(-0.9, 0.9, 50)
        vals = beta_pdf(x, d=2)
        assert vals.shape == x.shape
        assert np.all(vals > 0)

    def test_d3_uniform(self):
        x = np.linspace(-0.9, 0.9, 50)
        vals = beta_pdf(x, d=3)
        # d=3 → uniform on [-1,1], so f(x) = 0.5 everywhere
        np.testing.assert_allclose(vals, 0.5, atol=1e-10)

    def test_d4_general_shape(self):
        x = np.array([0.0, 0.5, -0.5])
        vals = beta_pdf(x, d=4)
        assert vals.shape == (3,)
        assert np.all(vals >= 0)
        # Symmetric: f(x) = f(-x)
        np.testing.assert_allclose(vals[1], vals[2], rtol=1e-6)

    def test_normalises_to_one(self):
        """Integral of beta_pdf over [-1,1] should equal 1."""
        for d in [3, 4, 8, 32, 128]:
            result, _ = integrate.quad(lambda x: beta_pdf(x, d), -1, 1)
            assert abs(result - 1.0) < 1e-6, f"d={d}: integral={result}"

    def test_symmetry(self):
        """beta_pdf is symmetric: f(x) == f(-x)."""
        for d in [3, 4, 16, 64]:
            x = np.linspace(0.1, 0.9, 20)
            np.testing.assert_allclose(beta_pdf(x, d), beta_pdf(-x, d), rtol=1e-6)

    def test_scalar_input(self):
        val = beta_pdf(0.0, d=8)
        assert np.isscalar(val) or val.ndim == 0 or val.shape == ()

    def test_d2_avoids_divide_by_zero(self):
        # At endpoints, 1-x²=0; the clip(1e-12) should prevent inf/nan
        x = np.array([-1.0, 0.0, 1.0])
        vals = beta_pdf(x, d=2)
        assert np.all(np.isfinite(vals))


# ---------------------------------------------------------------------------
# lloyd_max
# ---------------------------------------------------------------------------

class TestLloydMax:
    @pytest.mark.parametrize("b", [1, 2, 3, 4])
    def test_output_length(self, b):
        centroids = lloyd_max(d=64, b=b)
        assert len(centroids) == 2**b

    @pytest.mark.parametrize("b", [1, 2, 3])
    def test_sorted(self, b):
        centroids = lloyd_max(d=64, b=b)
        assert np.all(np.diff(centroids) > 0), "Centroids must be strictly sorted"

    @pytest.mark.parametrize("b", [1, 2, 3])
    def test_within_unit_interval(self, b):
        centroids = lloyd_max(d=64, b=b)
        assert np.all(centroids >= -1.0)
        assert np.all(centroids <= 1.0)

    @pytest.mark.parametrize("d", [8, 64, 256])
    def test_symmetric_around_zero(self, d):
        """Centroids should be symmetric: c_i = -c_{N-1-i}."""
        centroids = lloyd_max(d=d, b=2)
        n = len(centroids)
        np.testing.assert_allclose(
            centroids, -centroids[::-1], atol=1e-8,
            err_msg=f"d={d}: centroids not symmetric"
        )

    def test_b1_two_centroids_symmetric(self):
        centroids = lloyd_max(d=64, b=1)
        assert len(centroids) == 2
        np.testing.assert_allclose(centroids[0], -centroids[1], atol=1e-8)
        assert centroids[1] > 0

    def test_convergence_tolerance(self):
        # Should converge well within 300 iterations; result stable across runs
        c1 = lloyd_max(d=64, b=2, n_iter=300)
        c2 = lloyd_max(d=64, b=2, n_iter=300)
        np.testing.assert_array_equal(c1, c2)

    def test_more_bits_finer_resolution(self):
        c1 = lloyd_max(d=64, b=1)
        c2 = lloyd_max(d=64, b=2)
        c3 = lloyd_max(d=64, b=3)
        # Higher b means smaller gap between adjacent centroids (on average)
        assert np.mean(np.diff(c1)) > np.mean(np.diff(c2)) > np.mean(np.diff(c3))


# ---------------------------------------------------------------------------
# get_codebook (caching)
# ---------------------------------------------------------------------------

class TestGetCodebook:
    def test_returns_sorted_array(self):
        cb = get_codebook(64, 2)
        assert cb.ndim == 1
        assert len(cb) == 4
        assert np.all(np.diff(cb) > 0)

    def test_cached(self):
        cb1 = get_codebook(64, 2)
        cb2 = get_codebook(64, 2)
        assert cb1 is cb2, "get_codebook should return the cached object"

    def test_different_params_differ(self):
        cb_b2 = get_codebook(64, 2)
        cb_b3 = get_codebook(64, 3)
        assert len(cb_b2) != len(cb_b3)

    def test_consistent_with_lloyd_max(self):
        cb = get_codebook(32, 2)
        expected = lloyd_max(d=32, b=2)
        np.testing.assert_array_equal(cb, expected)


# ---------------------------------------------------------------------------
# mse_cost
# ---------------------------------------------------------------------------

class TestMseCost:
    def test_non_negative(self):
        cb = get_codebook(64, 2)
        cost = mse_cost(cb, d=64)
        assert cost >= 0.0

    def test_decreases_with_more_bits(self):
        d = 64
        costs = [mse_cost(get_codebook(d, b), d) for b in [1, 2, 3, 4]]
        for i in range(len(costs) - 1):
            assert costs[i] > costs[i + 1], (
                f"b={i+1} cost {costs[i]:.6f} not > b={i+2} cost {costs[i+1]:.6f}"
            )

    def test_matches_empirical(self):
        """mse_cost should agree with empirical D_mse / d at large n."""
        from turboquant.quantizer import TurboQuantMSE
        d, b = 128, 2
        cb = get_codebook(d, b)
        theoretical = mse_cost(cb, d)

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20_000, d)).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        q = TurboQuantMSE(d, b, seed=0)
        empirical = q.mse(X) / d  # per-coordinate MSE

        # Should agree within 5% (law of large numbers)
        assert abs(theoretical - empirical) / theoretical < 0.05
