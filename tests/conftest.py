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


class _FakeLLM:
    """LLM stub: counts calls and returns a fixed response string."""

    def __init__(self, response):
        self._response = response
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        return self._response


@pytest.fixture
def fake_llm():
    return _FakeLLM("Other")


@pytest.fixture
def fake_llm_returning():
    return _FakeLLM
