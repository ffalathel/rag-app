"""Tests for docuchat.service: sessions, TTL, quota, and LLM error mapping."""

import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from docuchat.config import Settings
from docuchat.service import (
    LLMError, LLMNotConfigured, QuotaExceeded, Service, ServiceError,
    UploadRejected,
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


def test_llm_failure_falls_back_to_the_local_gguf_when_one_is_set(make_service):
    class Broken:
        def complete(self, prompt):
            raise ConnectionError("503 high demand")

    class Local:
        def complete(self, prompt):
            return "local answer"

    def factory(cfg):
        return Local() if cfg.llm_provider == "llamacpp" else Broken()

    def ask_returning_completion(query, store, cfg, llm):
        return {"answer": llm.complete("p"), "sources": [], "debug": {"timings": {}}}

    svc = make_service(ask_fn=ask_returning_completion, llm_factory=factory,
                       llm_provider="gemini", gguf_path="/models/qwen.gguf")
    assert svc.answer("s", "q")["answer"] == "local answer"
    # without a gguf_path the failure still surfaces as a 502
    with pytest.raises(LLMError):
        make_service(ask_fn=ask_returning_completion, llm_factory=factory,
                     llm_provider="gemini").answer("s", "q")


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


PRIVATE = SimpleNamespace(name="PRIVATE", nodes=[1, 2, 3])
FILES = [("a.pdf", b"%PDF")]


@pytest.fixture
def upload_service(clock, monkeypatch):
    # check_upload is covered in test_upload_caps.py; skip real PDF parsing here
    monkeypatch.setattr("docuchat.service.check_upload", lambda files, cfg: None)

    def _make(build=lambda files, llm, cfg: PRIVATE, **overrides):
        return Service(Settings(**overrides), sample_store=SAMPLE, clock=clock,
                       ask_fn=fake_ask, build_upload_store_fn=build)
    return _make


def test_upload_switches_only_that_session_to_its_private_store(upload_service):
    svc = upload_service()
    assert svc.ingest_upload("s1", FILES) == 3
    assert svc.answer("s1", "q")["answer"] == "PRIVATE"
    assert svc.answer("s2", "q")["answer"] == "SAMPLE"


def test_reset_returns_the_session_to_the_sample(upload_service):
    svc = upload_service()
    svc.ingest_upload("s1", FILES)
    svc.reset("s1")
    assert svc.answer("s1", "q")["answer"] == "SAMPLE"


def test_idle_session_expires_and_reports_it_once(upload_service, clock):
    svc = upload_service(session_ttl_minutes=30)
    svc.ingest_upload("s1", FILES)
    clock.t += 31 * 60
    first = svc.answer("s1", "q")
    assert (first["answer"], first["expired"]) == ("SAMPLE", True)
    assert svc.answer("s1", "q")["expired"] is False


def test_activity_keeps_a_session_alive(upload_service, clock):
    svc = upload_service(session_ttl_minutes=30)
    svc.ingest_upload("s1", FILES)
    for _ in range(3):
        clock.t += 29 * 60
        assert svc.answer("s1", "q")["answer"] == "PRIVATE"


def test_failed_ingestion_keeps_the_previous_store(upload_service):
    calls = []

    def build(files, llm, cfg):
        calls.append(files)
        if len(calls) == 2:
            raise RuntimeError("docling blew up")
        return PRIVATE

    svc = upload_service(build=build)
    svc.ingest_upload("s1", FILES)
    with pytest.raises(RuntimeError):
        svc.ingest_upload("s1", FILES)
    assert svc.answer("s1", "q")["answer"] == "PRIVATE"


def test_upload_cap(upload_service):
    svc = upload_service(daily_upload_cap=1)
    svc.ingest_upload("s1", FILES)
    with pytest.raises(QuotaExceeded):
        svc.ingest_upload("s1", FILES)


def test_rejected_upload_never_reaches_ingestion_or_quota(clock):
    def must_not_run(files, llm, cfg):
        raise AssertionError("ingestion must not run for a rejected upload")

    svc = Service(Settings(daily_upload_cap=1), sample_store=SAMPLE, clock=clock,
                  ask_fn=fake_ask, build_upload_store_fn=must_not_run)
    with pytest.raises(UploadRejected):
        svc.ingest_upload("s1", [("notes.txt", b"hello")])
    svc._build_upload_store = lambda files, llm, cfg: PRIVATE
    pdf_bytes = (Path(__file__).parent / "fixtures" / "cfpb_closing_disclosure.pdf").read_bytes()
    svc.ingest_upload("s1", [("cd.pdf", pdf_bytes)])  # the one allowed upload is unspent


def test_overlong_question_is_422_and_spends_nothing(make_service):
    def must_not_run(*a, **k):
        raise AssertionError("ask() must not run for an over-long question")

    svc = make_service(ask_fn=must_not_run, daily_query_cap=1)
    with pytest.raises(ServiceError) as exc:
        svc.answer("s", "x" * 1001)
    assert exc.value.status == 422
    assert str(exc.value) == "Question is too long."
    svc._ask = fake_ask
    svc.answer("s", "x" * 1000)  # the limit itself is allowed, quota untouched


def test_touch_prunes_idle_storeless_sessions_but_keeps_pending_notices(make_service, clock):
    svc = make_service(session_ttl_minutes=1)
    for i in range(5):
        svc.reset(f"r{i}")
        svc.answer(f"a{i}", "q")
    with svc._lock:
        svc._touch("up").store = object()
    clock.t += 61
    with svc._lock:
        svc._touch("other")  # sweep: "up" expires, storeless ones are pruned
    assert set(svc._sessions) == {"up", "other"}
    assert svc._sessions["up"].expired
    clock.t += 61
    with svc._lock:
        svc._touch("other")
    assert "up" in svc._sessions  # still waiting to report its expiry
