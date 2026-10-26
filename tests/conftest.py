"""Shared fixtures, and the path shim for the vendored reference code."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REFERENCE = Path(__file__).parent / "reference"
if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20261026)


@pytest.fixture
def unit_vectors(rng: np.random.Generator):
    """A factory for (n, d) unit-norm float32 vectors."""

    def make(n: int, d: int = 16) -> np.ndarray:
        v = rng.normal(size=(n, d)).astype(np.float32)
        return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)

    return make
