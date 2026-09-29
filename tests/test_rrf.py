"""Tests for reciprocal_rank_fusion. Bug 1 regression: the notebook keyed its
accumulator on doc.text[:200], which collapsed distinct chunks sharing a long
boilerplate prefix (common in mortgage instruments) into one entry."""

import pytest

from docuchat.retrieval import reciprocal_rank_fusion


def test_chunks_sharing_a_200_char_prefix_are_not_collapsed(make_node):
    shared = "WHEREAS the Borrower and the Lender agree as follows " * 5  # >200 chars
    a = make_node(text=shared + " FIRST DISTINCT TAIL", chunk_id=1, page_number=1)
    b = make_node(text=shared + " SECOND DISTINCT TAIL", chunk_id=2, page_number=1)
    fused = reciprocal_rank_fusion([[(a, 0.9), (b, 0.8)]], k=60)
    assert len(fused) == 2


def test_same_node_in_two_lists_is_fused_once_with_summed_score(make_node):
    a = make_node(text="alpha", chunk_id=1)
    b = make_node(text="beta", chunk_id=2)
    fused = reciprocal_rank_fusion([[(a, 0.9), (b, 0.1)], [(a, 0.5)]], k=60)
    assert len(fused) == 2
    assert fused[0][0].metadata["chunk_id"] == 1
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 61)


def test_rank_order_is_by_fused_score_descending(make_node):
    a = make_node(text="alpha", chunk_id=1)
    b = make_node(text="beta", chunk_id=2)
    c = make_node(text="gamma", chunk_id=3)
    # a: rank 0 in both lists (highest fused score).
    # b: rank 1 in list one only.
    # c: rank 1 in list two only, same rank as b but appears once like b.
    fused = reciprocal_rank_fusion(
        [[(a, 0.9), (b, 0.5)], [(a, 0.8), (c, 0.4)]], k=60
    )
    scores = [score for _, score in fused]
    assert scores == sorted(scores, reverse=True)
    # a appears in both lists, so it must outrank b and c, which each appear once.
    assert fused[0][0].metadata["chunk_id"] == 1
    fused_ids = [node.metadata["chunk_id"] for node, _ in fused]
    assert fused_ids.index(1) < fused_ids.index(2)
    assert fused_ids.index(1) < fused_ids.index(3)


def test_empty_input_returns_empty():
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []
