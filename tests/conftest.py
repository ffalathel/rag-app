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


class _IdenticalVectorEncoder:
    """Encoder stub: same as _StubEncoder, returns identical unit vectors for
    any input list so every pair looks like a near-duplicate."""

    def encode(self, texts, normalize_embeddings=True):
        return np.ones((len(texts), 8), dtype=float) / np.sqrt(8)


@pytest.fixture
def identical_vector_encoder():
    return _IdenticalVectorEncoder()


class _DistinctVectorEncoder:
    """Encoder stub: returns orthogonal unit vectors, one axis per input, so
    no two texts ever look like near-duplicates."""

    def encode(self, texts, normalize_embeddings=True):
        return np.eye(len(texts), dtype=float)


@pytest.fixture
def distinct_vector_encoder():
    return _DistinctVectorEncoder()


class _RecordingCrossEncoder:
    """Cross-encoder stub: records the (query, text) pairs it was given and
    returns descending scores, one per pair."""

    def __init__(self):
        self.pairs = []

    def predict(self, pairs):
        self.pairs.extend(pairs)
        n = len(pairs)
        return [float(n - i) for i in range(n)]


@pytest.fixture
def recording_cross_encoder():
    return _RecordingCrossEncoder()


class _FakeLLM:
    """LLM stub: counts calls and returns a fixed response string."""

    def __init__(self, response):
        self._response = response
        self.calls = 0
        self.prompts = []

    def complete(self, prompt):
        self.calls += 1
        self.prompts.append(prompt)
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
    """A Store with a real 3-node VectorStoreIndex (MockEmbedding, so no
    network/model load) and a recording BM25 stub. Tests that only exercise
    BM25 dispatch monkeypatch retrieval.vector_search instead of relying on
    the vector index's actual similarity ordering."""
    from llama_index.core import VectorStoreIndex
    from llama_index.core.embeddings import MockEmbedding

    from docuchat.index import Store

    nodes = [make_node(text=f"doc {i}", chunk_id=i) for i in range(3)]
    results = [(nodes[i], 1.0 - i * 0.1) for i in range(3)]
    vector_index = VectorStoreIndex(nodes=nodes, embed_model=MockEmbedding(embed_dim=8))
    return Store(vector_index=vector_index, bm25=_RecordingBM25(results), nodes=nodes)


@pytest.fixture
def empty_store():
    """A Store with zero nodes in both indexes, for zero-retrieval-result tests."""
    from llama_index.core import VectorStoreIndex
    from llama_index.core.embeddings import MockEmbedding

    from docuchat.index import BM25Index, Store

    vector_index = VectorStoreIndex(nodes=[], embed_model=MockEmbedding(embed_dim=8))
    return Store(vector_index=vector_index, bm25=BM25Index([]), nodes=[])


class _CallCountingEncoder:
    """Encoder stub: counts calls and returns distinct unit vectors."""

    def __init__(self):
        self.calls = 0

    def encode(self, texts, normalize_embeddings=True):
        self.calls += 1
        return np.eye(len(texts), 8, dtype=float)


class _CallCountingCrossEncoder:
    """Cross-encoder stub: counts calls and returns one descending score per pair."""

    def __init__(self):
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        n = len(pairs)
        return [float(n - i) for i in range(n)]


@pytest.fixture
def fake_models():
    """{"encoder": ..., "cross_encoder": ...} call-counting stubs, splatted
    into ask()."""
    return {"encoder": _CallCountingEncoder(), "cross_encoder": _CallCountingCrossEncoder()}


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


@pytest.hookimpl(trylast=True)  # after -m deselection
def pytest_collection_modifyitems(config, items):
    # Integration tests call real APIs; load the API key from the repo's
    # gitignored dotenv file for them only, so unit tests never see it.
    if any(item.get_closest_marker("integration") for item in items):
        from dotenv import load_dotenv

        load_dotenv()


class _FakeService:
    """Stands in for docuchat.service.Service in API/UI tests."""

    def __init__(self):
        from docuchat.config import Settings

        self.cfg = Settings()
        self.ensured = False
        self.uploads = []
        self.resets = []
        self.error = None  # set to a ServiceError to make answer/ingest raise it

    def ensure_sample(self):
        self.ensured = True

    def new_session(self):
        return "sid"

    def answer(self, session_id, query):
        if self.error:
            raise self.error
        return {"answer": "a", "sources": [], "timings": {"total": 1.0}, "expired": False}

    def ingest_upload(self, session_id, files):
        if self.error:
            raise self.error
        self.uploads.append((session_id, files))
        return 3

    def reset(self, session_id):
        self.resets.append(session_id)


@pytest.fixture
def fake_service():
    return _FakeService()
