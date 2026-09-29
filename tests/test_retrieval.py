"""Tests for retrieve() mode dispatch and the LLM query-processing helpers."""

import pytest

import docuchat.retrieval as retrieval
from docuchat.config import Settings
from docuchat.retrieval import decompose_query, rewrite_query


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
def test_retrieve_dispatches_on_mode(mode, fake_store, make_node, monkeypatch):
    vector_calls = []
    vector_results = [(make_node(text=f"v{i}", chunk_id=100 + i), 1.0 - i * 0.1) for i in range(3)]

    def fake_vector_search(query, store, cfg):
        vector_calls.append((query, store, cfg))
        return vector_results

    monkeypatch.setattr(retrieval, "vector_search", fake_vector_search)

    cfg = Settings(retrieval_mode=mode, top_k=2)
    result = retrieval.retrieve("what is the rate?", fake_store, cfg)

    if mode == "vector":
        assert len(vector_calls) == 1
        assert fake_store.bm25.calls == []
    elif mode == "bm25":
        assert vector_calls == []
        assert len(fake_store.bm25.calls) == 1
    else:
        assert len(vector_calls) == 1
        assert len(fake_store.bm25.calls) == 1

    assert len(result) <= cfg.top_k


def test_rewrite_falls_back_to_original_on_empty_llm_output(fake_llm_returning):
    q = "what is the rate?"
    assert rewrite_query(q, fake_llm_returning("")) == q


def test_rewrite_falls_back_when_output_absurdly_long(fake_llm_returning):
    q = "what is the rate?"
    assert rewrite_query(q, fake_llm_returning("x" * 600)) == q


def test_decompose_returns_original_when_llm_returns_nothing(fake_llm_returning):
    assert decompose_query("q?", fake_llm_returning(""), Settings()) == ["q?"]


def test_decompose_caps_at_max_sub_queries(fake_llm_returning):
    llm = fake_llm_returning("\n".join(f"sub question number {i}?" for i in range(10)))
    assert len(decompose_query("q?", llm, Settings(max_sub_queries=4))) == 4


def test_decompose_strips_leading_markers_only(fake_llm_returning):
    # Decision 1: leading-marker regex, not the notebook's two-sided digit strip,
    # which would turn "30-year fixed rate?" into "year fixed rate?".
    llm = fake_llm_returning("30-year fixed rate terms?\n1. escrow account details?")
    result = decompose_query("q?", llm, Settings())
    assert "30-year fixed rate terms?" in result
    assert "escrow account details?" in result
