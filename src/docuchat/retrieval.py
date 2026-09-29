"""Vector/BM25/hybrid retrieval, RRF fusion, and LLM query processing.

Ported from complete_mortgage_rag_pipeline.py (`vector_search`,
`reciprocal_rank_fusion`, `hybrid_retrieve`, `rewrite_query`,
`decompose_query`). The notebook's `USE_LLAMAINDEX_BM25` branch does not
port -- the only BM25 here is `store.bm25.search`.

Fixes bug 1: `reciprocal_rank_fusion` used to key its accumulator on
`doc.text[:200]`, which silently collapsed distinct chunks sharing a long
boilerplate prefix (common in mortgage instruments) into a single result.
It now keys on node identity (`node_key`) instead. Everything here returns
`TextNode`, never the notebook's re-wrapped `Document`.
"""

import re

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import TextNode

from docuchat.config import Settings
from docuchat.index import Store


def node_key(node: TextNode) -> tuple[str, int, int]:
    return (node.metadata["filename"], node.metadata["page_number"], node.metadata["chunk_id"])


def vector_search(query: str, store: Store, cfg: Settings) -> list[tuple[TextNode, float]]:
    """Vector search over the Store's index."""
    nodes = VectorIndexRetriever(
        index=store.vector_index, similarity_top_k=cfg.top_k
    ).retrieve(query)
    return [(n.node, n.score or 0.0) for n in nodes]


def reciprocal_rank_fusion(
    result_lists: list[list[tuple[TextNode, float]]], k: int
) -> list[tuple[TextNode, float]]:
    """Merge ranked lists with RRF, keyed on node identity."""
    node_scores: dict[tuple[str, int, int], float] = {}
    node_objects: dict[tuple[str, int, int], TextNode] = {}
    for result_list in result_lists:
        for rank, (node, _) in enumerate(result_list):
            key = node_key(node)
            if key not in node_scores:
                node_scores[key] = 0.0
                node_objects[key] = node
            node_scores[key] += 1.0 / (k + rank + 1)
    sorted_keys = sorted(node_scores, key=lambda key: node_scores[key], reverse=True)
    return [(node_objects[key], node_scores[key]) for key in sorted_keys]


def retrieve(query: str, store: Store, cfg: Settings) -> list[tuple[TextNode, float]]:
    """Dispatch on cfg.retrieval_mode. All modes truncate to cfg.top_k."""
    if cfg.retrieval_mode == "vector":
        results = vector_search(query, store, cfg)
    elif cfg.retrieval_mode == "bm25":
        results = store.bm25.search(query, cfg.top_k)
    else:
        vector_results = vector_search(query, store, cfg)
        bm25_results = store.bm25.search(query, cfg.top_k)
        results = reciprocal_rank_fusion([vector_results, bm25_results], k=cfg.rrf_k)
    return results[: cfg.top_k]


def rewrite_query(user_query: str, llm, cfg: Settings) -> str:
    """LLM rewrites vague question into a precise retrieval query."""
    domain = cfg.profile.search_domain
    prompt = f"""You are a query rewriting assistant for a {domain} search system.
Rewrite the user's question to be more specific and comprehensive for searching {domain}s.
Add relevant {domain} terminology. Keep it as a single search query.

User question: "{user_query}"

Rewritten query:"""
    try:
        response = llm.complete(prompt)
        rewritten = str(response).strip().split("\n")[0].strip().strip("\"'")
        return rewritten if 5 <= len(rewritten) <= 500 else user_query
    except Exception:
        return user_query


def decompose_query(user_query: str, llm, cfg: Settings) -> list[str]:
    """Break complex questions into up to cfg.max_sub_queries focused sub-queries."""
    domain = cfg.profile.search_domain
    prompt = f"""You are a query decomposition assistant for a {domain} search system.
If the question is simple, return just the original question.
If complex, break into 2-4 focused sub-questions.
Return ONLY the sub-questions, one per line. No numbering, no bullets.

Question: "{user_query}"

Sub-questions:"""
    try:
        response = llm.complete(prompt)
        sub_queries = []
        for line in str(response).strip().split("\n"):
            line = re.sub(r"^(?:[-•*]+|\d+[.)])\s*", "", line.strip())
            line = line.strip().strip("\"'")
            if len(line) > 10:
                sub_queries.append(line)
        return sub_queries[: cfg.max_sub_queries] if sub_queries else [user_query]
    except Exception:
        return [user_query]
