import pytest

from docuchat.config import Settings
from docuchat.models import (
    _sentence_transformer_of,
    get_embed_model,
    get_encoder,
    get_llm,
)


def test_get_llm_is_cached(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    get_llm.cache_clear()
    cfg = Settings()
    assert get_llm(cfg) is get_llm(cfg)


def test_get_llm_without_api_key_raises_naming_the_variable(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_llm.cache_clear()
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        get_llm(Settings())


def test_get_llm_llamacpp_without_gguf_path_raises():
    get_llm.cache_clear()
    with pytest.raises(RuntimeError, match="gguf_path"):
        get_llm(Settings(llm_provider="llamacpp", gguf_path=""))


@pytest.mark.integration
def test_encoder_and_embed_model_share_one_loaded_model():
    cfg = Settings()
    assert get_encoder(cfg) is _sentence_transformer_of(get_embed_model(cfg))
