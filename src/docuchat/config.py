"""Settings: every tuning constant for docuchat, in one frozen dataclass.

Frozen (and built only from tuple/scalar fields) so instances are hashable
and can key an lru_cache. Domain-specific values live in profiles.py and are
reached via the `profile` property, never as fields here.
"""

import dataclasses
import os
from typing import Optional, Union, get_args, get_origin

from docuchat.profiles import PROFILES, DomainProfile

_RETRIEVAL_MODES = {"vector", "bm25", "hybrid"}
_LLM_PROVIDERS = {"anthropic", "llamacpp"}

_TRUE_VALUES = {"1", "true", "yes"}
_FALSE_VALUES = {"0", "false", "no"}


def _parse_bool(raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise ValueError(f"cannot parse {raw!r} as a boolean")


def _cast(field_type, raw: str):
    if field_type is bool:
        return _parse_bool(raw)

    if get_origin(field_type) is Union:
        args = [a for a in get_args(field_type) if a is not type(None)]
        if raw == "":
            return None
        return _cast(args[0], raw)

    if field_type is int:
        return int(raw)
    if field_type is float:
        return float(raw)
    return raw


@dataclasses.dataclass(frozen=True)
class Settings:
    # Ingestion
    min_text_length: int = 50
    do_ocr: bool = True
    do_table_structure: bool = True

    # Chunking
    chunk_max_tokens: int = 512
    chunk_min_tokens: int = 50
    merge_threshold: float = 0.75
    merge_max_tokens: int = 600
    min_chunk_words: int = 10

    # Domain
    domain_profile: str = "mortgage"

    # Classification
    use_llm_classification: bool = True
    classify_snippet_chars: int = 500

    # Retrieval
    retrieval_mode: str = "hybrid"
    top_k: int = 10
    rrf_k: int = 60
    use_rewrite: bool = True
    use_decomposition: bool = True
    max_sub_queries: int = 4

    # Rerank
    use_rerank: bool = True
    rerank_top_n: int = 5
    min_rerank_score: Optional[float] = None
    dedup_threshold: float = 0.92
    max_context_tokens: int = 2000

    # Models
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-5"
    temperature: float = 0.1
    max_new_tokens: int = 1024
    context_window: int = 4096
    embed_model_name: str = "BAAI/bge-small-en-v1.5"
    cross_encoder_name: str = "BAAI/bge-reranker-v2-m3"
    cross_encoder_max_length: int = 1024
    gguf_path: str = ""

    def __post_init__(self) -> None:
        if self.retrieval_mode not in _RETRIEVAL_MODES:
            raise ValueError(
                f"retrieval_mode must be one of {sorted(_RETRIEVAL_MODES)}, "
                f"got {self.retrieval_mode!r}"
            )
        if self.llm_provider not in _LLM_PROVIDERS:
            raise ValueError(
                f"llm_provider must be one of {sorted(_LLM_PROVIDERS)}, "
                f"got {self.llm_provider!r}"
            )
        if self.domain_profile not in PROFILES:
            raise ValueError(
                f"domain_profile must be one of {sorted(PROFILES)}, "
                f"got {self.domain_profile!r}"
            )

    @property
    def profile(self) -> DomainProfile:
        return PROFILES[self.domain_profile]

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        kwargs = {}
        for f in dataclasses.fields(cls):
            env_name = f"DOCUCHAT_{f.name.upper()}"
            raw = os.environ.get(env_name)
            if raw is not None:
                try:
                    kwargs[f.name] = _cast(f.type, raw)
                except ValueError as exc:
                    raise ValueError(f"{env_name}={raw!r}: expected {f.type}") from exc
        kwargs.update(overrides)
        return cls(**kwargs)
