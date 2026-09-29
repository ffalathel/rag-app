"""Tests for docuchat.rerank.rerank.

Ported from complete_mortgage_rag_pipeline.py lines 653-660 (rerank_results).
The first test is the bug-2 regression: the notebook sliced doc.text to 512
characters before scoring, discarding most of a chunk even though the
cross-encoder's max_length (tokens, not characters) already truncates safely.
"""

from docuchat.config import Settings
from docuchat.rerank import rerank


def test_cross_encoder_receives_full_chunk_text_not_512_chars(make_node, recording_cross_encoder):
    long_text = "word " * 400  # ~2000 characters
    node = make_node(text=long_text)
    rerank("query", [(node, 0.0)], recording_cross_encoder, Settings())
    _, passed_text = recording_cross_encoder.pairs[0]
    assert passed_text == long_text  # not long_text[:512]


class _ShuffledScoreCrossEncoder:
    """Cross-encoder stub whose scores are deliberately NOT in input pair
    order, so a passing test proves rerank actually sorts rather than
    happening to preserve input order."""

    def predict(self, pairs):
        # chunk 4 (5th pair) scores highest; everything else descends from
        # the front and back of the input, none of it already sorted.
        order = {4: 100, 0: 10, 1: 9, 2: 8, 3: 7, 5: 6, 6: 5}
        return [order[i] for i in range(len(pairs))]


def test_rerank_sorts_descending_and_truncates_to_top_n(make_node):
    results = [(make_node(text=f"doc {i}", chunk_id=i), 0.0) for i in range(7)]
    reranked = rerank("query", results, _ShuffledScoreCrossEncoder(), Settings(rerank_top_n=3))

    assert len(reranked) == 3
    scores = [s for _, s in reranked]
    assert scores == sorted(scores, reverse=True)
    assert reranked[0][0].metadata["chunk_id"] == 4


def test_rerank_on_empty_results_returns_empty(recording_cross_encoder):
    assert rerank("q", [], recording_cross_encoder, Settings()) == []
