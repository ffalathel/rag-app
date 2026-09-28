"""Lazy, cached getters for every heavy model docuchat uses.

Every model-loading library (torch, sentence_transformers, docling, and the
llama_index provider/embedding integrations that transitively import them)
is imported inside a function body, never at module scope. Importing this
module must never download a GGUF, load a HuggingFace model, or touch the
network -- see tests/test_import_purity.py, which enforces this in a
subprocess for every module in the package.

Each getter is @lru_cache(maxsize=1) over a Settings instance, so it loads
its model once per distinct config and reuses it afterwards.
"""

import os
from functools import lru_cache

from docuchat.config import Settings


@lru_cache(maxsize=1)
def get_llm(cfg: Settings):
    """Return the configured LLM (LlamaIndex `LLM`): Anthropic or LlamaCPP."""
    if cfg.llm_provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set; required for llm_provider='anthropic'"
            )
        from llama_index.llms.anthropic import Anthropic

        return Anthropic(
            model=cfg.llm_model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_new_tokens,
        )

    # cfg.llm_provider == "llamacpp" (Settings validates there is no other value)
    if not cfg.gguf_path or not os.path.exists(cfg.gguf_path):
        raise RuntimeError(
            f"gguf_path is not set to an existing file (got {cfg.gguf_path!r}); "
            "required for llm_provider='llamacpp'"
        )
    from llama_index.llms.llama_cpp import LlamaCPP

    return LlamaCPP(
        model_path=cfg.gguf_path,
        temperature=cfg.temperature,
        max_new_tokens=cfg.max_new_tokens,
        context_window=cfg.context_window,
        model_kwargs={"n_gpu_layers": -1},
    )


@lru_cache(maxsize=1)
def get_embed_model(cfg: Settings):
    """Return the configured HuggingFaceEmbedding."""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(model_name=cfg.embed_model_name)


def _sentence_transformer_of(embed_model):
    """Return the SentenceTransformer that a HuggingFaceEmbedding already
    loaded internally, instead of constructing a second copy of it.

    HuggingFaceEmbedding (llama-index-embeddings-huggingface==0.3.1) stores
    it on the private attribute `_model`. If a future version renames or
    drops that attribute, fail loudly rather than silently falling back to
    loading a second SentenceTransformer -- that duplicate load is exactly
    the bug this helper exists to remove.
    """
    if not hasattr(embed_model, "_model"):
        raise RuntimeError(
            "HuggingFaceEmbedding has no '_model' attribute in the installed "
            "llama-index-embeddings-huggingface version; the SentenceTransformer "
            "it holds internally could not be located. Refusing to construct a "
            "second SentenceTransformer instead."
        )
    return embed_model._model


@lru_cache(maxsize=1)
def get_encoder(cfg: Settings):
    """Return the SentenceTransformer underlying get_embed_model(cfg).

    Not a second load: bge-small was being loaded twice in the notebook
    (once via HuggingFaceEmbedding, once via a standalone SentenceTransformer),
    wasting a load and ~130MB. This getter reuses the one HuggingFaceEmbedding
    already holds.
    """
    return _sentence_transformer_of(get_embed_model(cfg))


@lru_cache(maxsize=1)
def get_cross_encoder(cfg: Settings):
    """Return the configured CrossEncoder reranker."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(cfg.cross_encoder_name, max_length=cfg.cross_encoder_max_length)
