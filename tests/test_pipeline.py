"""Tests for docuchat.pipeline: ask() and build_prompt()."""

import pytest

import docuchat.pipeline as pipeline
from docuchat.config import Settings
from docuchat.pipeline import ask, build_prompt


def test_zero_retrieval_results_returns_documented_shape(fake_llm, empty_store, fake_models):
    result = ask("anything", empty_store, Settings(), llm=fake_llm, **fake_models)
    assert result["num_chunks_used"] == 0
    assert result["sources"] == []
    assert result["confidence"] == 0.0
    assert isinstance(result["answer"], str)


def test_empty_store_query_never_constructs_a_model_client(empty_store, monkeypatch):
    # With rewrite/decomposition/rerank all disabled, nothing before the
    # zero-results early return needs a model, so none of get_llm/get_encoder/
    # get_cross_encoder should even be called.
    def boom(cfg):
        raise AssertionError("should not construct a model client for an empty-result query")

    monkeypatch.setattr(pipeline, "get_llm", boom)
    monkeypatch.setattr(pipeline, "get_encoder", boom)
    monkeypatch.setattr(pipeline, "get_cross_encoder", boom)

    cfg = Settings(use_rewrite=False, use_decomposition=False, use_rerank=False)
    result = ask("anything", empty_store, cfg)
    assert result["num_chunks_used"] == 0


def test_sources_carry_citation_metadata(fake_llm, fake_store, fake_models):
    result = ask("q", fake_store, Settings(), llm=fake_llm, **fake_models)
    assert set(result["sources"][0]) == {
        "filename", "page_number", "doc_type", "section_title", "score", "preview"}


def test_debug_records_rewritten_query_and_sub_queries(fake_llm, fake_store, fake_models):
    result = ask("q", fake_store, Settings(), llm=fake_llm, **fake_models)
    assert set(result["debug"]) == {"original_query", "rewritten_query", "sub_queries"}


@pytest.mark.parametrize("profile", ["mortgage", "generic"])
def test_prompt_contains_grounding_and_refusal_instruction(profile, make_node):
    prompt = build_prompt("q", [(make_node(text="ctx"), 1.0)],
                           Settings(domain_profile=profile))
    assert "ONLY from the provided context" in prompt
    assert "cannot find this information" in prompt


def test_prompt_preamble_comes_from_the_profile(make_node):
    node = [(make_node(text="ctx"), 1.0)]
    assert "mortgage" in build_prompt("q", node, Settings()).lower()
    assert "mortgage" not in build_prompt(
        "q", node, Settings(domain_profile="generic")).lower()
