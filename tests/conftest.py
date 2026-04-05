"""
Shared fixtures for TurboQuant tests.
"""
import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Device fixture — use MPS on Apple Silicon, CUDA if available, else CPU
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@pytest.fixture(scope="session")
def torch_dtype() -> torch.dtype:
    return torch.float16


# ---------------------------------------------------------------------------
# Reproducible random unit-norm vectors
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


def make_unit_vectors(n: int, d: int, seed: int = 0) -> np.ndarray:
    """Random unit-norm float32 vectors, shape (n, d)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def make_unit_tensors(n: int, d: int, device: str, dtype=torch.float32, seed: int = 0):
    """Random unit-norm torch tensors, shape (n, d)."""
    X = make_unit_vectors(n, d, seed=seed)
    return torch.from_numpy(X).to(device=device, dtype=dtype)
