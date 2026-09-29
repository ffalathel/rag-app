"""Reranking and context cleanup.

Ported from complete_mortgage_rag_pipeline.py lines 653-690 (rerank_results,
clean_context). Fixes bug 2: the notebook sliced doc.text to 512 characters
before scoring, even though the cross-encoder's max_length (tokens, not
characters) already truncates safely -- `rerank` now passes full node text.

The cross-encoder and encoder are injected parameters, not module globals,
and cosine similarity is a plain numpy dot product of already-normalised
vectors, matching the house pattern in chunking.semantic_merge.
"""

import numpy as np

from docuchat.config import Settings


def rerank(
    query: str,
    results: list[tuple],
    cross_encoder,
    cfg: Settings,
) -> list[tuple]:
    """Cross-encoder reranking: score full chunk text, sort descending,
    truncate to cfg.rerank_top_n."""
    if not results:
        return []

    pairs = [(query, node.get_content()) for node, _ in results]
    scores = cross_encoder.predict(pairs)
    scored = list(zip([node for node, _ in results], scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[: cfg.rerank_top_n]


def clean_context(
    reranked: list[tuple],
    encoder,
    cfg: Settings,
) -> list[tuple]:
    """Filter by score, remove near-duplicates, enforce token budget."""
    if not reranked:
        return []

    # Score filter (skipped entirely when cfg.min_rerank_score is None; when
    # everything falls below it, keep the single highest-scoring item).
    if cfg.min_rerank_score is None:
        filtered = list(reranked)
    else:
        filtered = [(n, s) for n, s in reranked if s >= cfg.min_rerank_score]
        if not filtered:
            filtered = [max(reranked, key=lambda x: x[1])]

    # Near-duplicate removal.
    if len(filtered) > 1:
        texts = [n.get_content() for n, _ in filtered]
        embeddings = encoder.encode(texts, normalize_embeddings=True)
        keep = [0]
        for i in range(1, len(filtered)):
            if all(
                float(np.dot(embeddings[i], embeddings[j])) < cfg.dedup_threshold
                for j in keep
            ):
                keep.append(i)
        filtered = [filtered[i] for i in keep]

    # Token budget: the first chunk is always admitted regardless of size;
    # the loop then breaks on the first later chunk that would exceed it.
    final: list[tuple] = []
    total = 0.0
    for i, (node, score) in enumerate(filtered):
        tokens = len(node.get_content().split()) * 1.3
        if i == 0 or total + tokens <= cfg.max_context_tokens:
            final.append((node, score))
            total += tokens
        else:
            break
    return final
