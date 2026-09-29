"""Shared pytest fixtures."""

import numpy as np
import pytest


class _StubEncoder:
    """Encoder stub: returns identical unit vectors for any input list."""

    def encode(self, texts, normalize_embeddings=True):
        return np.ones((len(texts), 8), dtype=float) / np.sqrt(8)


@pytest.fixture
def stub_encoder():
    return _StubEncoder()
