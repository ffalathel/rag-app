"""Shared pytest fixtures."""

import numpy as np
import pytest
from llama_index.core.schema import TextNode


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


@pytest.fixture
def text_nodes():
    """A handful of TextNodes with distinct text and full metadata, reusable
    by Tasks 7-9. "prepayment" appears in exactly one node's text so BM25
    ranking tests can assert on an unambiguous top hit."""
    texts = [
        "The borrower may prepay the loan in full at any time, subject to a prepayment penalty "
        "equal to six months of interest if paid within the first three years.",
        "The property shall be maintained in good condition and the borrower shall pay all "
        "property taxes and insurance premiums when due.",
        "Late payments incur a fee of five percent of the overdue amount, assessed ten days "
        "after the due date has passed.",
        "This promissory note is secured by a deed of trust covering the real property "
        "described in exhibit A attached hereto.",
    ]
    return [
        TextNode(
            text=text,
            metadata={
                "filename": "a.pdf",
                "page_number": i + 1,
                "doc_type": "Promissory Note",
                "source_type": "docling",
                "section_title": "Full Page",
                "table_present": False,
                "chunk_id": i,
            },
        )
        for i, text in enumerate(texts)
    ]
