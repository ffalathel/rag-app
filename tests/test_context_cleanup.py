"""Tests for docuchat.rerank.clean_context.

Ported from complete_mortgage_rag_pipeline.py lines 663-690 (clean_context).
Three stages run in order: score filter, near-duplicate removal, token budget.
"""

from docuchat.config import Settings
from docuchat.rerank import clean_context


def test_score_filter_applies_when_min_rerank_score_is_set(make_node, stub_encoder):
    kept = clean_context(
        [(make_node(chunk_id=1), 5.0), (make_node(chunk_id=2), -9.0)],
        stub_encoder,
        Settings(min_rerank_score=0.0),
    )
    assert [n.metadata["chunk_id"] for n, _ in kept] == [1]


def test_no_score_filter_by_default(make_node, distinct_vector_encoder):
    # distinct_vector_encoder, not stub_encoder: identical vectors would
    # collapse both nodes in the dedup stage, failing len == 2 for a reason
    # unrelated to the score filter this test targets.
    pairs = [(make_node(chunk_id=1), -50.0), (make_node(chunk_id=2), -60.0)]
    assert len(clean_context(pairs, distinct_vector_encoder, Settings())) == 2


def test_keeps_highest_scoring_chunk_when_all_below_threshold(make_node, stub_encoder):
    pairs = [(make_node(chunk_id=1), -1.0), (make_node(chunk_id=2), -2.0)]
    kept = clean_context(pairs, stub_encoder, Settings(min_rerank_score=10.0))
    assert [n.metadata["chunk_id"] for n, _ in kept] == [1]


def test_near_duplicates_are_removed(make_node, identical_vector_encoder):
    pairs = [
        (make_node(chunk_id=1, text="a " * 20), 1.0),
        (make_node(chunk_id=2, text="a " * 20), 0.9),
    ]
    assert len(clean_context(pairs, identical_vector_encoder, Settings())) == 1


def test_token_budget_truncates_the_tail(make_node, distinct_vector_encoder):
    pairs = [(make_node(chunk_id=i, text="word " * 500), 1.0) for i in range(4)]
    kept = clean_context(pairs, distinct_vector_encoder, Settings(max_context_tokens=2000))
    assert len(kept) < 4


# Review Focus 1
def test_single_oversized_chunk_is_still_returned(make_node, distinct_vector_encoder):
    pairs = [(make_node(chunk_id=1, text="word " * 5000), 1.0)]
    kept = clean_context(pairs, distinct_vector_encoder, Settings(max_context_tokens=2000))
    assert len(kept) == 1
