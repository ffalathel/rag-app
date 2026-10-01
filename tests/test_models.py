import sys
import types

import pytest

from docuchat.config import Settings
from docuchat.models import (
    _load_cross_encoder,
    _load_embed_model,
    _load_llm,
    _sentence_transformer_of,
    get_cross_encoder,
    get_embed_model,
    get_encoder,
    get_judge_llm,
    get_llm,
)


def test_get_llm_is_cached(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _load_llm.cache_clear()
    cfg = Settings()
    assert get_llm(cfg) is get_llm(cfg)


def test_get_llm_without_api_key_raises_naming_the_variable(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _load_llm.cache_clear()
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        get_llm(Settings())


def test_get_llm_llamacpp_without_gguf_path_raises():
    _load_llm.cache_clear()
    with pytest.raises(RuntimeError, match="gguf_path"):
        get_llm(Settings(llm_provider="llamacpp", gguf_path=""))


def _fake_llm_module(monkeypatch, module_name, class_name):
    """Install a fake llama_index LLM module whose class records its kwargs,
    so constructing it never imports the real SDK or touches the network."""
    class _Recorder:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    module = types.ModuleType(module_name)
    setattr(module, class_name, _Recorder)
    monkeypatch.setitem(sys.modules, module_name, module)


def test_empty_llm_model_resolves_to_the_provider_default(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _fake_llm_module(monkeypatch, "llama_index.llms.anthropic", "Anthropic")
    _load_llm.cache_clear()
    assert get_llm(Settings()).kwargs["model"] == "claude-sonnet-5"


@pytest.mark.parametrize("key_var", ["GOOGLE_API_KEY", "GEMINI_API_KEY"])
def test_get_llm_gemini_accepts_either_key_variable(monkeypatch, key_var):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv(key_var, "test-key")
    _fake_llm_module(monkeypatch, "llama_index.llms.google_genai", "GoogleGenAI")
    _load_llm.cache_clear()
    llm = get_llm(Settings(llm_provider="gemini"))
    assert llm.kwargs["model"] == "gemini-3.7-flash"
    # both set, or GoogleGenAI makes a network call from its constructor
    assert llm.kwargs["max_tokens"] is not None
    assert llm.kwargs["context_window"] is not None
    # Gemini 3 models may reject an explicit temperature
    assert llm.kwargs["temperature"] is None


def test_get_llm_gemini_keeps_temperature_for_older_models(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    _fake_llm_module(monkeypatch, "llama_index.llms.google_genai", "GoogleGenAI")
    _load_llm.cache_clear()
    llm = get_llm(Settings(llm_provider="gemini", llm_model="gemini-2.5-flash"))
    assert llm.kwargs["temperature"] == Settings().temperature


def test_get_llm_gemini_without_api_key_raises_naming_the_variable(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    _load_llm.cache_clear()
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        get_llm(Settings(llm_provider="gemini"))


@pytest.mark.integration
def test_encoder_and_embed_model_share_one_loaded_model():
    cfg = Settings()
    assert get_encoder(cfg) is _sentence_transformer_of(get_embed_model(cfg))


def test_get_judge_llm_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        get_judge_llm(Settings())


def test_cross_encoder_and_encoder_ignore_unrelated_settings_changes(monkeypatch):
    """Two Settings differing only in use_rerank must not evict/reload the
    encoder or cross-encoder -- they're cached on a narrower key than the
    whole Settings instance. Heavy constructors are swapped for fakes via
    sys.modules so this never imports torch/sentence_transformers for real."""

    class FakeHuggingFaceEmbedding:
        def __init__(self, model_name):
            self._model = types.SimpleNamespace(encode=lambda *a, **k: None)

    fake_hf_module = types.ModuleType("llama_index.embeddings.huggingface")
    fake_hf_module.HuggingFaceEmbedding = FakeHuggingFaceEmbedding
    monkeypatch.setitem(sys.modules, "llama_index.embeddings.huggingface", fake_hf_module)

    class FakeCrossEncoder:
        def __init__(self, name, max_length):
            self.name = name

        def predict(self, pairs):
            return [0.0] * len(pairs)

    fake_st_module = types.ModuleType("sentence_transformers")
    fake_st_module.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st_module)

    _load_embed_model.cache_clear()
    _load_cross_encoder.cache_clear()

    cfg_rerank = Settings(use_rerank=True)
    cfg_no_rerank = Settings(use_rerank=False)

    assert get_encoder(cfg_rerank) is get_encoder(cfg_no_rerank)
    assert get_cross_encoder(cfg_rerank) is get_cross_encoder(cfg_no_rerank)
