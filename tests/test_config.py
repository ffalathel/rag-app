import pytest

from docuchat.config import Settings


def test_defaults_match_notebook_constants():
    cfg = Settings()
    assert cfg.chunk_max_tokens == 512
    assert cfg.merge_threshold == 0.75
    assert cfg.dedup_threshold == 0.92
    assert cfg.max_context_tokens == 2000
    assert cfg.rrf_k == 60
    assert cfg.top_k == 10
    assert cfg.rerank_top_n == 5
    assert cfg.min_rerank_score is None
    assert cfg.llm_model == "claude-sonnet-5"
    assert cfg.cross_encoder_name == "BAAI/bge-reranker-v2-m3"
    assert cfg.cross_encoder_max_length == 1024
    assert cfg.max_new_tokens == 1024


def test_from_env_casts_by_field_type(monkeypatch):
    monkeypatch.setenv("DOCUCHAT_TOP_K", "25")
    monkeypatch.setenv("DOCUCHAT_MERGE_THRESHOLD", "0.9")
    cfg = Settings.from_env()
    assert cfg.top_k == 25 and isinstance(cfg.top_k, int)
    assert cfg.merge_threshold == 0.9


@pytest.mark.parametrize("raw,expected", [
    ("false", False), ("False", False), ("0", False), ("no", False),
    ("true", True), ("True", True), ("1", True), ("yes", True),
])
def test_from_env_parses_bools_explicitly(monkeypatch, raw, expected):
    monkeypatch.setenv("DOCUCHAT_USE_RERANK", raw)
    assert Settings.from_env().use_rerank is expected


def test_from_env_overrides_take_precedence(monkeypatch):
    monkeypatch.setenv("DOCUCHAT_TOP_K", "25")
    assert Settings.from_env(top_k=3).top_k == 3


def test_from_env_cast_error_names_the_variable(monkeypatch):
    monkeypatch.setenv("DOCUCHAT_TOP_K", "ten")
    with pytest.raises(ValueError, match="DOCUCHAT_TOP_K"):
        Settings.from_env()


def test_settings_is_hashable_for_lru_cache():
    assert hash(Settings()) == hash(Settings())


# Review Focus 3
def test_invalid_retrieval_mode_raises():
    with pytest.raises(ValueError, match="retrieval_mode"):
        Settings(retrieval_mode="vektor")


def test_invalid_llm_provider_raises():
    with pytest.raises(ValueError, match="llm_provider"):
        Settings(llm_provider="openai")


def test_invalid_domain_profile_raises():
    with pytest.raises(ValueError, match="domain_profile"):
        Settings(domain_profile="banking")


def test_profile_property_resolves():
    assert Settings().profile.name == "mortgage"
    assert Settings(domain_profile="generic").profile.name == "generic"
