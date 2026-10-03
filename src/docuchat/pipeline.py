"""The one public query entry point: ask().

Ported from `ask_v3` in complete_mortgage_rag_pipeline.py (lines 692-790).
Drops the notebook's `verbose` prints and its `text[:200]` dedup key --
dedup across sub-queries now uses `retrieval.node_key`, for the same reason
`reciprocal_rank_fusion` does (see retrieval.py). The notebook's separate
`ask()` (line 265) does not port; this is the only entry point.
"""

import time

import numpy as np

from docuchat.config import Settings
from docuchat.index import Store
from docuchat.models import get_cross_encoder, get_encoder, get_llm
from docuchat.rerank import clean_context, rerank
from docuchat.retrieval import decompose_query, node_key, retrieve, rewrite_query


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def build_prompt(query: str, cleaned: list[tuple], cfg: Settings) -> str:
    """Build the grounded-answer prompt: profile preamble, numbered source
    headers, context delimiters, and the trailing question -- verbatim from
    the notebook's ask_v3."""
    context_parts = []
    for i, (node, _score) in enumerate(cleaned):
        m = node.metadata
        context_parts.append(
            f"[Source {i + 1}: {m.get('filename', '?')}, Page {m.get('page_number', '?')}, "
            f"Type: {m.get('doc_type', '?')}, Section: {m.get('section_title', '?')}]\n"
            f"{node.get_content()}"
        )
    context_block = "\n\n---\n\n".join(context_parts)

    return f"""{cfg.profile.answer_system_prompt}

=== CONTEXT ===
{context_block}
=== END CONTEXT ===

Question: {query}

Answer (with citations):"""


def ask(
    query: str,
    store: Store,
    cfg: Settings,
    llm=None,
    encoder=None,
    cross_encoder=None,
) -> dict:
    """Run the full pipeline: optional rewrite -> optional decompose ->
    retrieve per sub-query -> dedup on node_key -> optional rerank ->
    clean_context -> build_prompt -> llm.complete."""
    start = time.perf_counter()
    timings: dict[str, float] = {}
    debug = {"original_query": query, "timings": timings}

    if cfg.use_rewrite:
        if llm is None:
            llm = get_llm(cfg)
        t = time.perf_counter()
        rewritten = rewrite_query(query, llm, cfg)
        timings["rewrite"] = _ms(t)
    else:
        rewritten = query
    debug["rewritten_query"] = rewritten

    if cfg.use_decomposition:
        if llm is None:
            llm = get_llm(cfg)
        t = time.perf_counter()
        sub_queries = decompose_query(rewritten, llm, cfg)
        timings["decompose"] = _ms(t)
    else:
        sub_queries = [rewritten]
    debug["sub_queries"] = sub_queries

    t = time.perf_counter()
    all_retrieved = []
    for sq in sub_queries:
        all_retrieved.extend(retrieve(sq, store, cfg))

    seen = set()
    unique = []
    for node, score in all_retrieved:
        key = node_key(node)
        if key not in seen:
            seen.add(key)
            unique.append((node, score))
    timings["retrieve"] = _ms(t)

    debug["candidates"] = [
        (node.metadata.get("filename", "?"), node.metadata.get("page_number", "?"))
        for node, _score in unique
    ]

    if not unique:
        debug["contexts"] = []
        timings["total"] = _ms(start)
        return {
            "answer": "No relevant documents found.",
            "sources": [],
            "num_chunks_used": 0,
            "confidence": 0.0,
            "debug": debug,
        }

    if cfg.use_rerank:
        if cross_encoder is None:
            cross_encoder = get_cross_encoder(cfg)
        t = time.perf_counter()
        reranked = rerank(query, unique, cross_encoder, cfg)
        timings["rerank"] = _ms(t)
    else:
        reranked = sorted(unique, key=lambda pair: pair[1], reverse=True)[: cfg.rerank_top_n]

    if encoder is None:
        encoder = get_encoder(cfg)
    t = time.perf_counter()
    cleaned = clean_context(reranked, encoder, cfg)
    timings["clean_context"] = _ms(t)

    if llm is None:
        llm = get_llm(cfg)

    prompt = build_prompt(query, cleaned, cfg)
    t = time.perf_counter()
    answer = str(llm.complete(prompt)).strip()
    timings["llm"] = _ms(t)

    sources, scores, contexts = [], [], []
    for node, score in cleaned:
        m = node.metadata
        contexts.append(node.get_content())
        sources.append(
            {
                "filename": m.get("filename", "?"),
                "page_number": m.get("page_number", "?"),
                "doc_type": m.get("doc_type", "unknown"),
                "section_title": m.get("section_title", ""),
                "score": round(float(score), 4),
                "preview": " ".join(node.get_content().split()[:25]) + "...",
            }
        )
        scores.append(float(score))

    debug["contexts"] = contexts
    timings["total"] = _ms(start)

    return {
        "answer": answer,
        "sources": sources,
        "num_chunks_used": len(sources),
        "confidence": round(float(np.mean(scores)), 4) if scores else 0.0,
        "debug": debug,
    }
