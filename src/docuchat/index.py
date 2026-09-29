"""Nodes, manual BM25 index, and the Store bundle replacing pipeline globals.

Ported from complete_mortgage_rag_pipeline.py (`create_documents_from_pages`,
`build_phase2_documents`, `BM25Index`, `tokenize_for_bm25`). The notebook's
`USE_LLAMAINDEX_BM25` dual path and `llamaindex_bm25` do not port -- only the
manual `BM25Okapi` implementation is kept.

`tokenize_for_bm25` uses a regex tokenizer plus a module-level stopword set,
not `nltk`: `nltk.word_tokenize`/`nltk.corpus.stopwords` both require
downloaded corpora, which would make the default test run hit the network.
"""

import re
from dataclasses import dataclass

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from rank_bm25 import BM25Okapi

from docuchat.chunking import chunk_page, detect_table_content
from docuchat.config import Settings
from docuchat.models import get_embed_model, get_encoder

NODE_METADATA_KEYS: tuple[str, ...] = (
    "filename",
    "page_number",
    "doc_type",
    "source_type",
    "section_title",
    "table_present",
    "chunk_id",
)

STOP_WORDS: frozenset[str] = frozenset(
    """
    a an the and or but if while is are was were be been being
    of at by for with about against between into through during
    before after above below to from up down in out on off over under
    again further then once here there when where why how all any both
    each few more most other some such no nor not only own same so than
    too very s t can will just don should now this that these those i
    you he she it we they what which who whom as
    """.split()
)


def tokenize_for_bm25(text: str) -> list[str]:
    """Lowercase, keep alnum/currency/percent runs, strip edge punctuation,
    drop stopwords and pure-punctuation tokens."""
    raw_tokens = re.findall(r"[\w$%.,]+", text.lower())
    tokens = []
    for token in raw_tokens:
        token = token.strip(".,")
        if not token:
            continue
        if not any(c.isalnum() for c in token) and "$" not in token and "%" not in token:
            continue
        if token in STOP_WORDS:
            continue
        tokens.append(token)
    return tokens


class BM25Index:
    """Manual BM25Okapi index over a corpus of TextNodes."""

    def __init__(self, nodes: list[TextNode]):
        self.nodes = nodes
        self.corpus_tokens = [tokenize_for_bm25(node.get_content()) for node in nodes]
        has_tokens = any(self.corpus_tokens)
        self.bm25 = BM25Okapi(self.corpus_tokens) if has_tokens else None

    def search(self, query: str, top_k: int) -> list[tuple[TextNode, float]]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize_for_bm25(query))
        ranked = sorted(zip(self.nodes, scores), key=lambda pair: pair[1], reverse=True)
        return [(node, score) for node, score in ranked[:top_k] if score > 0]


def build_nodes(pages: list[dict], encoder, cfg: Settings) -> list[TextNode]:
    """Chunk every page and wrap each chunk in a TextNode with corpus-wide
    metadata, including a unique incrementing chunk_id."""
    nodes = []
    chunk_id = 0
    for page in pages:
        if not page["text"].strip():
            continue
        for chunk_text, section_title in chunk_page(page["text"], encoder, cfg):
            metadata = {
                "filename": page["filename"],
                "page_number": page["page_number"],
                "doc_type": page["doc_type"],
                "source_type": page["source_type"],
                "section_title": section_title,
                "table_present": detect_table_content(chunk_text),
                "chunk_id": chunk_id,
            }
            nodes.append(TextNode(text=chunk_text, metadata=metadata))
            chunk_id += 1
    return nodes


@dataclass
class Store:
    vector_index: VectorStoreIndex
    bm25: BM25Index
    nodes: list[TextNode]


def store_from_nodes(nodes: list[TextNode], cfg: Settings) -> Store:
    """Build a Store directly from already-chunked nodes."""
    vector_index = VectorStoreIndex(nodes=nodes, embed_model=get_embed_model(cfg))
    return Store(vector_index=vector_index, bm25=BM25Index(nodes), nodes=nodes)


def build_store(pages: list[dict], cfg: Settings) -> Store:
    encoder = get_encoder(cfg)
    return store_from_nodes(build_nodes(pages, encoder, cfg), cfg)
