"""Lazy, cached getters for every heavy model docuchat uses.

Every model-loading library (torch, sentence_transformers, docling, and the
llama_index provider/embedding integrations that transitively import them)
is imported inside a function body, never at module scope. Importing this
module must never download a GGUF, load a HuggingFace model, or touch the
network -- see tests/test_import_purity.py, which enforces this in a
subprocess for every module in the package.

Each getter delegates to an @lru_cache(maxsize=1) loader keyed only on the
Settings fields that model depends on, so changing an unrelated field (e.g.
use_rerank between ablation arms) reuses the loaded model.
"""

import functools
import os
import threading
from functools import lru_cache

from docuchat.config import Settings

# used when Settings.llm_model is left empty; llamacpp takes gguf_path instead
_DEFAULT_MODELS = {"anthropic": "claude-sonnet-5", "gemini": "gemini-3.7-flash"}

# Local models on Apple's MPS backend segfault under concurrent inference
# (the eval runner and Streamlit both call from several threads), so every
# load and forward pass of a local model goes through this one lock.
_LOCAL_MODEL_LOCK = threading.RLock()


def _serialized(method):
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        with _LOCAL_MODEL_LOCK:
            return method(*args, **kwargs)

    return wrapper


@lru_cache(maxsize=1)
def _load_llm(llm_provider, llm_model, temperature, max_new_tokens, context_window, gguf_path):
    """Construct the LLM, cached only on the fields it actually depends on so
    an ablation arm that varies an unrelated Settings field (e.g. use_rerank)
    does not evict and reload it."""
    if llm_provider == "anthropic":
        from llama_index.llms.anthropic import Anthropic

        return Anthropic(model=llm_model, temperature=temperature, max_tokens=max_new_tokens)

    if llm_provider == "gemini":
        from llama_index.llms.google_genai import GoogleGenAI

        # max_tokens and context_window are both passed because GoogleGenAI
        # fetches model metadata over the network when either is missing.
        # Gemini 3 models may reject an explicit temperature.
        return GoogleGenAI(
            model=llm_model,
            temperature=None if "gemini-3" in llm_model else temperature,
            max_tokens=max_new_tokens,
            context_window=context_window,
        )

    # llm_provider == "llamacpp" (Settings validates there is no other value)
    return _llamacpp(gguf_path, temperature, max_new_tokens, context_window)


def _qwen3_prompt(completion: str) -> str:
    # Qwen3's ChatML format; the empty think block switches off thinking mode
    return f"<|im_start|>user\n{completion}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _llamacpp(gguf_path, temperature, max_new_tokens, context_window):
    """A Qwen3 GGUF via llama.cpp. Not thread-safe (LlamaCPP mutates its
    generate_kwargs per call): run the eval with --workers 1."""
    from llama_index.llms.llama_cpp import LlamaCPP

    return LlamaCPP(
        model_path=gguf_path,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        context_window=context_window,
        completion_to_prompt=_qwen3_prompt,
        generate_kwargs={"stop": ["<|im_end|>"]},
        model_kwargs={"n_gpu_layers": -1},
    )


def get_llm(cfg: Settings):
    """Return the configured LLM (LlamaIndex `LLM`): Anthropic, Gemini, or
    LlamaCPP. An empty `llm_model` means the provider's default model.

    The ANTHROPIC_API_KEY / gguf_path checks run on every call, not cached,
    so a config that later has the env var set (or the file created) is not
    stuck with a cached failure.
    """
    if cfg.llm_provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set; required for llm_provider='anthropic'"
            )
    elif cfg.llm_provider == "gemini":
        # the google-genai SDK reads either variable itself
        if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
            raise RuntimeError(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) is not set; "
                "required for llm_provider='gemini'"
            )
    else:
        if not cfg.gguf_path or not os.path.exists(cfg.gguf_path):
            raise RuntimeError(
                f"gguf_path is not set to an existing file (got {cfg.gguf_path!r}); "
                "required for llm_provider='llamacpp'"
            )
    return _load_llm(
        cfg.llm_provider, cfg.llm_model or _DEFAULT_MODELS.get(cfg.llm_provider, ""),
        cfg.temperature, cfg.max_new_tokens,
        cfg.context_window, cfg.gguf_path,
    )


@lru_cache(maxsize=1)
def _load_judge_llm(judge_model, max_new_tokens, context_window):
    """Separate cache from _load_llm so judging never evicts the answer LLM
    from _load_llm's maxsize=1 cache."""
    if judge_model.startswith("gemini"):
        from llama_index.llms.google_genai import GoogleGenAI

        # see _load_llm: explicit limits avoid a metadata fetch; Gemini 3
        # may reject an explicit temperature
        return GoogleGenAI(
            model=judge_model,
            temperature=None if "gemini-3" in judge_model else 0.0,
            max_tokens=max_new_tokens,
            context_window=context_window,
        )

    if judge_model.endswith(".gguf"):
        return _llamacpp(judge_model, 0.0, max_new_tokens, context_window)

    from llama_index.llms.anthropic import Anthropic

    return Anthropic(model=judge_model, temperature=0.0, max_tokens=max_new_tokens)


def get_judge_llm(cfg: Settings):
    """Return the judge LLM used to score answers during evaluation: Gemini
    when `judge_model` starts with "gemini", a local llama.cpp model when it
    is a path ending ".gguf", otherwise Anthropic. Temperature 0.0 where the
    model accepts one."""
    if cfg.judge_model.endswith(".gguf"):
        if not os.path.exists(cfg.judge_model):
            raise RuntimeError(f"judge_model GGUF not found: {cfg.judge_model!r}")
    elif cfg.judge_model.startswith("gemini"):
        if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
            raise RuntimeError(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) is not set; required for a gemini judge_model"
            )
    elif not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set; required for get_judge_llm")
    return _load_judge_llm(cfg.judge_model, cfg.max_new_tokens, cfg.context_window)


@lru_cache(maxsize=1)
def _load_embed_model(embed_model_name):
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    with _LOCAL_MODEL_LOCK:
        embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
    st = _sentence_transformer_of(embed_model)
    st.encode = _serialized(st.encode)  # covers get_encoder() too: same object
    return embed_model


def get_embed_model(cfg: Settings):
    """Return the configured HuggingFaceEmbedding, cached only on
    embed_model_name."""
    with _LOCAL_MODEL_LOCK:  # lru_cache alone lets racing threads each load a copy
        return _load_embed_model(cfg.embed_model_name)


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


def get_encoder(cfg: Settings):
    """Return the SentenceTransformer underlying get_embed_model(cfg).

    Not a second load: bge-small was being loaded twice in the notebook
    (once via HuggingFaceEmbedding, once via a standalone SentenceTransformer),
    wasting a load and ~130MB. This getter reuses the one HuggingFaceEmbedding
    already holds -- cached only on embed_model_name via get_embed_model, so
    still one loaded model, not two.
    """
    return _sentence_transformer_of(get_embed_model(cfg))


@lru_cache(maxsize=1)
def _load_cross_encoder(cross_encoder_name, cross_encoder_max_length):
    from sentence_transformers import CrossEncoder

    with _LOCAL_MODEL_LOCK:
        cross_encoder = CrossEncoder(cross_encoder_name, max_length=cross_encoder_max_length)
    cross_encoder.predict = _serialized(cross_encoder.predict)
    return cross_encoder


def get_cross_encoder(cfg: Settings):
    """Return the configured CrossEncoder reranker, cached only on
    cross_encoder_name and cross_encoder_max_length."""
    with _LOCAL_MODEL_LOCK:  # lru_cache alone lets racing threads each load a copy
        return _load_cross_encoder(cfg.cross_encoder_name, cfg.cross_encoder_max_length)
