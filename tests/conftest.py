"""Shared pytest fixtures."""

import numpy as np
import pytest
from llama_index.core.schema import TextNode

from docuchat.index import NODE_METADATA_KEYS


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
def make_node():
    """Factory for a TextNode with every NODE_METADATA_KEYS field defaulted,
    so a test can pass only the metadata it cares about."""
    defaults = {
        "filename": "a.pdf",
        "page_number": 1,
        "doc_type": "Promissory Note",
        "source_type": "docling",
        "section_title": "Full Page",
        "table_present": False,
        "chunk_id": 0,
    }

    def _make(text="some text", **kw):
        metadata = dict(defaults)
        metadata.update({k: v for k, v in kw.items() if k in NODE_METADATA_KEYS})
        extra = {k: v for k, v in kw.items() if k not in NODE_METADATA_KEYS}
        return TextNode(text=text, metadata=metadata, **extra)

    return _make


class _RecordingBM25:
    """BM25 stub: records every search() call and returns a fixed result list."""

    def __init__(self, results):
        self._results = results
        self.calls = []

    def search(self, query, top_k):
        self.calls.append((query, top_k))
        return self._results


@pytest.fixture
def fake_store(make_node):
    """A Store-like object with a recording BM25 stub. vector_index is unused
    directly -- tests dispatch through retrieval.vector_search, which they
    monkeypatch."""
    from docuchat.index import Store

    results = [(make_node(text=f"doc {i}", chunk_id=i), 1.0 - i * 0.1) for i in range(3)]
    return Store(vector_index=None, bm25=_RecordingBM25(results), nodes=[])


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
