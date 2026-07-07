"""
Shared fixtures for engramma-memory real-world tests.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engramma_memory import EngrammaMemory


@pytest.fixture
def mem():
    """Fresh 128-dim local memory instance."""
    return EngrammaMemory(dim=128, backend="local")


@pytest.fixture
def mem_256():
    """Fresh 256-dim local memory instance."""
    return EngrammaMemory(dim=256, backend="local")


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(seed=42)


def make_embedding(dim: int, seed: int) -> np.ndarray:
    """Create a deterministic normalized embedding."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-8)


def text_to_embedding(text: str, dim: int = 128) -> np.ndarray:
    """Deterministic text-to-embedding for testing."""
    rng = np.random.default_rng(hash(text) % (2**32))
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-8)
