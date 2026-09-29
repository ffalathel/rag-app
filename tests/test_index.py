"""Tests for node building, the manual BM25 index, and Store."""

import sys

from docuchat.config import Settings
from docuchat.index import NODE_METADATA_KEYS, BM25Index, build_nodes, tokenize_for_bm25


def test_nodes_carry_required_metadata(stub_encoder):
    pages = [{"filename": "a.pdf", "page_number": 1, "text": "1. PAYMENTS\n" + "word " * 50,
              "source_type": "docling", "doc_type": "Promissory Note"}]
    nodes = build_nodes(pages, stub_encoder, Settings())
    assert nodes and set(NODE_METADATA_KEYS) <= set(nodes[0].metadata)


def test_chunk_ids_are_unique_across_pages(stub_encoder):
    pages = [{"filename": "a.pdf", "page_number": n, "text": "word " * 50,
              "source_type": "docling", "doc_type": "Unknown"} for n in (1, 2, 3)]
    ids = [n.metadata["chunk_id"] for n in build_nodes(pages, stub_encoder, Settings())]
    assert len(ids) == len(set(ids))


def test_bm25_tokenizer_keeps_currency_and_percent():
    tokens = tokenize_for_bm25("The rate is 6.5% and the payment is $1,234")
    assert any("%" in t for t in tokens) and any("$" in t for t in tokens)


def test_bm25_tokenizer_drops_stopwords():
    assert "the" not in tokenize_for_bm25("the interest rate")


def test_bm25_tokenizer_needs_no_downloaded_corpus():
    # guards the CI no-network constraint: no nltk, no punkt, no stopwords corpus
    import docuchat.index as m
    assert "nltk" not in sys.modules
    assert not hasattr(m, "nltk")


def test_bm25_search_ranks_exact_term_match_first(text_nodes):
    idx = BM25Index(text_nodes)
    top = idx.search("prepayment penalty", top_k=3)
    assert "prepayment" in top[0][0].get_content().lower()


def test_bm25_search_on_empty_corpus_returns_empty():
    assert BM25Index([]).search("anything", top_k=5) == []


def test_bm25_tokenizer_keeps_accented_word_whole():
    assert "café" in tokenize_for_bm25("café")


def test_bm25_search_on_non_latin_corpus_does_not_raise(text_nodes, make_node):
    # A single-node corpus gives every BM25Okapi term a non-positive idf
    # (pre-existing rank_bm25 behavior, independent of tokenization), so this
    # uses a multi-node corpus to keep the assertion meaningful.
    node = make_node(text="مرحبا بالعالم", chunk_id=99)
    idx = BM25Index(text_nodes + [node])
    results = idx.search("مرحبا", 3)
    assert results and results[0][0] is node


def test_bm25_index_on_all_stopword_corpus_returns_empty(make_node):
    node = make_node(text="the a an")
    idx = BM25Index([node])
    assert idx.search("the", top_k=5) == []
