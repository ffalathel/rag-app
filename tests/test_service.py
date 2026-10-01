"""Tests for docuchat.service: sessions, TTL, quota, and LLM error mapping."""

import datetime
from types import SimpleNamespace

import pytest

from docuchat.config import Settings
from docuchat.service import (
    LLMError, LLMNotConfigured, QuotaExceeded, Service, ServiceError,
)

NOON = datetime.datetime(2026, 10, 1, 12, tzinfo=datetime.timezone.utc).timestamp()
SAMPLE = SimpleNamespace(name="SAMPLE", nodes=[])


class Clock:
    def __init__(self, t=NOON):
        self.t = t

    def __call__(self):
        return self.t


def fake_ask(query, store, cfg, llm):
    return {"answer": store.name, "sources": [], "debug": {"timings": {"total": 1.0}}}


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def make_service(clock):
    def _make(ask_fn=fake_ask, llm_factory=None, **overrides):
        kwargs = {"llm_factory": llm_factory} if llm_factory else {}
        return Service(Settings(**overrides), sample_store=SAMPLE, clock=clock,
                       ask_fn=ask_fn, **kwargs)
    return _make


def test_unknown_session_answers_from_the_sample(make_service):
    result = make_service().answer("anyone", "q")
    assert result == {"answer": "SAMPLE", "sources": [], "timings": {"total": 1.0},
                      "expired": False}


def test_query_cap_raises_quota_exceeded(make_service):
    svc = make_service(daily_query_cap=2)
    svc.answer("s", "q")
    svc.answer("s", "q")
    with pytest.raises(QuotaExceeded) as exc:
        svc.answer("s", "q")
    assert exc.value.status == 429


def test_quota_resets_at_utc_midnight(make_service, clock):
    svc = make_service(daily_query_cap=1)
    svc.answer("s", "q")
    clock.t = NOON + 12 * 3600 - 1  # 23:59:59 the same day
    with pytest.raises(QuotaExceeded):
        svc.answer("s", "q")
    clock.t = NOON + 12 * 3600 + 1  # 00:00:01 the next day
    svc.answer("s", "q")


# Review Focus 5
def test_blank_question_is_422_and_spends_nothing(make_service):
    def must_not_run(*a, **k):
        raise AssertionError("ask() must not run for a blank question")

    svc = make_service(ask_fn=must_not_run, daily_query_cap=1)
    with pytest.raises(ServiceError) as exc:
        svc.answer("s", "   ")
    assert exc.value.status == 422
    assert str(exc.value) == "Ask a question first."
    svc._ask = fake_ask
    svc.answer("s", "q")  # the one allowed query is still available


def _ask_calling_llm(query, store, cfg, llm):
    llm.complete("prompt")
    return fake_ask(query, store, cfg, llm)


def test_missing_key_maps_to_llm_not_configured(make_service):
    def no_key(cfg):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    with pytest.raises(LLMNotConfigured) as exc:
        make_service(ask_fn=_ask_calling_llm, llm_factory=no_key).answer("s", "q")
    assert exc.value.status == 503
    assert str(exc.value) == "LLM not configured."


def test_llm_call_failure_maps_to_llm_error(make_service):
    class Broken:
        def complete(self, prompt):
            raise ConnectionError("upstream reset")

    with pytest.raises(LLMError) as exc:
        make_service(ask_fn=_ask_calling_llm, llm_factory=lambda cfg: Broken()).answer("s", "q")
    assert exc.value.status == 502


def test_llm_is_resolved_lazily(make_service):
    def no_key(cfg):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    # fake_ask never calls llm.complete, so a missing key must not matter
    assert make_service(llm_factory=no_key).answer("s", "q")["answer"] == "SAMPLE"


def test_ensure_sample_is_a_no_op_when_a_store_is_given(make_service):
    svc = make_service()
    svc.ensure_sample()
    assert svc.sample_store is SAMPLE


def test_served_config_is_the_rerank_arm():
    cfg = Service.from_env().cfg
    assert (cfg.retrieval_mode, cfg.use_rewrite, cfg.use_decomposition, cfg.use_rerank) == (
        "hybrid", False, False, True)


def test_new_session_ids_are_unique_and_unguessable(make_service):
    svc = make_service()
    ids = {svc.new_session() for _ in range(100)}
    assert len(ids) == 100 and all(len(i) == 32 for i in ids)
