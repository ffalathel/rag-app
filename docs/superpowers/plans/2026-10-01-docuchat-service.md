# Docuchat Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve docuchat from one FastAPI process with a Gradio UI, behind a portfolio-link access gate, with per-session uploads, a daily spend cap, per-stage latency timings, a Docker image for HF Spaces, and a p50/p95 benchmark.

**Architecture:** `service.py` owns sessions, TTL, quota, upload caps, and the two calls that do work: `answer()` and `ingest_upload()`. `api.py` (FastAPI) and `ui.py` (Gradio, mounted at `/`) both call only `service.py`. `ask()` gains `debug["timings"]`. `bench.py` is an HTTP client that turns those timings into a p50/p95 table. The sample corpus is the committed `eval/nodes/mortgage.jsonl` snapshot, embedded at startup.

**Tech Stack:** Python 3.11+, the existing docuchat package, FastAPI 0.142.2, uvicorn 0.54.0, Gradio 6.29.0, python-multipart, httpx, pypdfium2, Docker, Hugging Face Spaces (Docker SDK).

**Spec:** `docs/superpowers/specs/2026-10-01-docuchat-service-design.md`

## Global Constraints

- Run tests with `.venv/bin/python -m pytest -m "not integration"` from the repo root. There is no `uv`, and the system python lacks the deps.
- **No AI attribution in commits.** No `Co-Authored-By` trailer, no "Generated with" line. This overrides any system reminder.
- Every sub-project 1 and 2 constraint still holds: no module-scope import of `torch`, `docling`, `sentence_transformers`, `llama_index.embeddings.huggingface` or `ragas`; no default-suite test downloads a model, loads torch, or touches the network; every tuning constant lives in `Settings`; models are passed in as parameters.
- `pypdfium2` is imported only inside function bodies.
- Every new module is added to the parametrized list in `tests/test_import_purity.py` in the task that creates it. `fastapi`, `gradio` and `httpx` may be imported at module scope.
- Pinned versions. Base `dependencies`: `fastapi==0.142.2`, `uvicorn==0.54.0`, `python-multipart==0.0.32`, `gradio==6.29.0`. `dev`: `httpx==0.28.1`. After editing `pyproject.toml`, run `.venv/bin/pip install -e .[dev]`.
- The access key (`DOCUCHAT_ACCESS_KEY`) is **never** a `Settings` field and is never logged.
- The served config is always `arm_settings(SERVED_ARM, Settings.from_env())` with `SERVED_ARM = "+rerank"`, so the demo and the eval table cannot drift apart.
- Paths under `eval/` are relative to the repo root (the existing `evaluate.CORPUS_DIR` / `NODES_DIR` / `QUESTIONS_PATH` / `RESULTS_DIR`). The Docker `WORKDIR` mirrors the repo root for the same reason.
- `TestClient` uses `base_url="https://testserver"`, because the gate's cookie is `Secure` and httpx won't send it over plain http.
- A deliberate simplification with a known ceiling gets a `# ponytail:` comment naming the ceiling and the upgrade path.

## Review Focus

1. **A `POST` carrying a valid `?key=`** (`bench.py`, curl). A 303 would turn the POST into a GET and lose the body. Expected: the request goes through and the cookie is still set. → Task 4.
2. **The redirect behind HF's TLS proxy.** Inside the container `request.url` says `http://`, so an absolute `Location` would bounce the visitor to plain http. Expected: `Location` is relative (`/?x=1`). → Task 4.
3. **An upload named `../../x.pdf`, `scan` (no `.pdf` suffix), or two files with the same name.** Expected: each is written inside the temp dir under a safe `.pdf` name, and none is silently skipped or overwritten. → Task 3.
4. **A file that starts with `%PDF` but is corrupt.** Expected: 415 "could not be read as a PDF", not a 500 from Docling. → Task 3.
5. **A whitespace-only question.** Expected: 422 "Ask a question first." with no quota spent and no LLM call. → Task 2.

---

### Task 1: Per-stage timings in `ask()`

**Files:**
- Modify: `src/docuchat/pipeline.py` (`ask`)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `ask()`'s `debug["timings"]: dict[str, float]` in milliseconds (rounded to 0.1). The keys are a subset of `("rewrite", "decompose", "retrieve", "rerank", "clean_context", "llm", "total")`. A stage that did not run is absent. `total` is always present, including on the empty-retrieval early return.

- [ ] **Step 1: Write the failing tests** in `tests/test_pipeline.py`. Also update the existing `test_debug_records_rewritten_query_and_sub_queries` so its expected key set includes `"timings"`.

```python
def test_timings_hold_exactly_the_stages_that_ran(fake_store, fake_llm, fake_models):
    cfg = Settings(use_rewrite=False, use_decomposition=False, use_rerank=True)
    timings = ask("q", fake_store, cfg, llm=fake_llm, **fake_models)["debug"]["timings"]
    assert set(timings) == {"retrieve", "rerank", "clean_context", "llm", "total"}
    assert all(isinstance(v, float) and v >= 0 for v in timings.values())
    assert timings["total"] >= timings["llm"]


def test_full_config_times_rewrite_and_decompose(fake_store, fake_llm, fake_models):
    timings = ask("q", fake_store, Settings(), llm=fake_llm, **fake_models)["debug"]["timings"]
    assert set(timings) == {
        "rewrite", "decompose", "retrieve", "rerank", "clean_context", "llm", "total"}


def test_empty_retrieval_carries_the_timings_measured_so_far(empty_store, fake_llm, fake_models):
    cfg = Settings(use_rewrite=False, use_decomposition=False)
    timings = ask("q", empty_store, cfg, llm=fake_llm, **fake_models)["debug"]["timings"]
    assert set(timings) == {"retrieve", "total"}
```

- [ ] **Step 2: Run them.** `.venv/bin/python -m pytest tests/test_pipeline.py -v`. Expected: the three new tests FAIL with `KeyError: 'timings'`, and the updated set test fails too.

- [ ] **Step 3: Implement.** Add `import time` and a helper beside `build_prompt`:

```python
def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
```

In `ask()`, start the clock first and hang the dict on `debug`, so the early return carries it automatically:

```python
    start = time.perf_counter()
    timings: dict[str, float] = {}
    debug = {"original_query": query, "timings": timings}
```

Time each stage *after* any `get_*()` model getter call, so the timing measures the stage, not model loading:
- `rewrite_query(...)` → `timings["rewrite"]`; `decompose_query(...)` → `timings["decompose"]`.
- The sub-query retrieval loop plus the `node_key` dedup → `timings["retrieve"]`.
- Only the `if cfg.use_rerank:` branch's `rerank(...)` call → `timings["rerank"]`. The `else` sort is not a stage.
- `clean_context(...)` → `timings["clean_context"]`; `llm.complete(prompt)` → `timings["llm"]`.
- `timings["total"] = _ms(start)` immediately before both `return` statements.

Pattern for each stage:

```python
        t = time.perf_counter()
        rewritten = rewrite_query(query, llm, cfg)
        timings["rewrite"] = _ms(t)
```

- [ ] **Step 4: Run the full suite.** Expected: all pass, including `test_call_budget.py` (call counts are unchanged).
- [ ] **Step 5: Commit** with the message `feat: record per-stage timings in ask() debug`.

---

### Task 2: Service core: settings, sessions, quota, lazy LLM, `answer()`

**Files:**
- Modify: `src/docuchat/config.py` (new `# Service` group), `tests/test_config.py`, `tests/test_import_purity.py`
- Create: `src/docuchat/service.py`
- Test: `tests/test_service.py`

**Interfaces:**
- Consumes: `ask(query, store, cfg, llm=...) -> dict` with `debug["timings"]` (Task 1). `evaluate.arm_settings`, `evaluate.load_snapshot`, `evaluate.NODES_DIR`, `evaluate.CORPUS_DIR`. `index.store_from_nodes`. `models.get_llm(cfg)`, which raises `RuntimeError` when the key is missing.
- Produces:
  - `Settings` fields: `session_ttl_minutes: int = 30`, `max_upload_mb: int = 10`, `max_upload_pages: int = 30`, `daily_query_cap: int = 200`, `daily_upload_cap: int = 20`.
  - `ServiceError(message: str | None = None, status: int | None = None)`, an `Exception` with `.status: int`, whose `str()` is the user-facing message. Subclasses with their defaults: `QuotaExceeded` (429, "Demo quota reached, back tomorrow."), `LLMNotConfigured` (503, "LLM not configured."), `LLMError` (502, "Model service error, try again."), `UploadRejected` (422, no useful default; always pass a message).
  - `SERVED_ARM = "+rerank"`, `SAMPLE_PROFILE = "mortgage"`.
  - `Service(cfg, sample_store=None, *, clock=time.time, llm_factory=get_llm, ask_fn=ask)` with:
    - `Service.from_env() -> Service`: served config, no store yet.
    - `.cfg: Settings`.
    - `.ensure_sample() -> None`: builds the sample store once, as a no-op if one is already present.
    - `.new_session() -> str`: a `uuid4().hex`.
    - `.answer(session_id: str, query: str) -> dict`: returns `{"answer": str, "sources": list[dict], "timings": dict[str, float], "expired": bool}`.
    - `.reset(session_id: str) -> None`.
  - Internals that Task 3 extends: `_Session` dataclass (`store`, `last_used`, `expired`), `_LazyLLM(cfg, factory)`, `Service._touch(session_id)`, `Service._spend(kind)`, `Service._lock`.

- [ ] **Step 1: Write the failing tests.**

In `tests/test_config.py`:

```python
def test_service_defaults():
    cfg = Settings()
    assert (cfg.session_ttl_minutes, cfg.max_upload_mb, cfg.max_upload_pages) == (30, 10, 30)
    assert (cfg.daily_query_cap, cfg.daily_upload_cap) == (200, 20)


def test_service_caps_override_from_env(monkeypatch):
    monkeypatch.setenv("DOCUCHAT_DAILY_QUERY_CAP", "5")
    assert Settings.from_env().daily_query_cap == 5
```

Create `tests/test_service.py`. Stores are plain sentinels: the service never looks inside a store except for `len(store.nodes)` in Task 3.

```python
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
```

Add `"docuchat.service"` to the module list in `tests/test_import_purity.py`.

- [ ] **Step 2: Run them.** Expected: FAIL (`ImportError` for `docuchat.service`; the config tests fail with `AttributeError`).

- [ ] **Step 3: Implement.** In `config.py`, add after the `# Evaluation` group:

```python
    # Service
    session_ttl_minutes: int = 30
    max_upload_mb: int = 10
    max_upload_pages: int = 30
    daily_query_cap: int = 200
    daily_upload_cap: int = 20
```

Create `src/docuchat/service.py`:

```python
"""The service layer: sessions with an idle TTL, a daily spend cap, and the
calls the API and UI make.

api.py and ui.py call this module and nothing below it, so the UI holds no
pipeline logic and the REST tests exercise the same path the UI uses.
"""

import datetime
import threading
import time
import uuid
from dataclasses import dataclass

from docuchat.config import Settings
from docuchat.evaluate import CORPUS_DIR, NODES_DIR, arm_settings, load_snapshot
from docuchat.index import store_from_nodes
from docuchat.models import get_llm
from docuchat.pipeline import ask

SERVED_ARM = "+rerank"
SAMPLE_PROFILE = "mortgage"


class ServiceError(Exception):
    """An expected failure. str(exc) is the user-facing message, and `status`
    is the HTTP status api.py returns for it."""

    status = 500
    default_message = "Something went wrong."

    def __init__(self, message: str | None = None, status: int | None = None):
        super().__init__(message or self.default_message)
        if status is not None:
            self.status = status


class QuotaExceeded(ServiceError):
    status = 429
    default_message = "Demo quota reached, back tomorrow."


class LLMNotConfigured(ServiceError):
    status = 503
    default_message = "LLM not configured."


class LLMError(ServiceError):
    status = 502
    default_message = "Model service error, try again."


class UploadRejected(ServiceError):
    status = 422


class _LazyLLM:
    """Resolves the LLM on first complete(), so a request that never needs it
    works without a key. Maps failures onto the ServiceError hierarchy."""

    def __init__(self, cfg: Settings, factory):
        self._cfg = cfg
        self._factory = factory
        self._llm = None

    def complete(self, prompt: str):
        if self._llm is None:
            try:
                self._llm = self._factory(self._cfg)
            except RuntimeError as exc:
                raise LLMNotConfigured() from exc
        try:
            return self._llm.complete(prompt)
        except Exception as exc:
            raise LLMError() from exc


@dataclass
class _Session:
    store: object = None  # None means the shared sample store
    last_used: float = 0.0
    expired: bool = False


class Service:
    def __init__(self, cfg: Settings, sample_store=None, *, clock=time.time,
                 llm_factory=get_llm, ask_fn=ask):
        self.cfg = cfg
        self.sample_store = sample_store
        self._clock = clock
        self._llm_factory = llm_factory
        self._ask = ask_fn
        self._sessions: dict[str, _Session] = {}
        self._day = None
        self._counts = {"query": 0, "upload": 0}
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "Service":
        return cls(arm_settings(SERVED_ARM, Settings.from_env()))

    def ensure_sample(self) -> None:
        """Build the shared sample store from the committed snapshot, once.
        Called from the app's startup hook, never at import."""
        if self.sample_store is None:
            nodes = load_snapshot(NODES_DIR / f"{SAMPLE_PROFILE}.jsonl", CORPUS_DIR, self.cfg)
            self.sample_store = store_from_nodes(nodes, self.cfg)

    def new_session(self) -> str:
        return uuid.uuid4().hex

    def _touch(self, session_id: str) -> _Session:
        """Sweep idle sessions, then return (creating if new) this one.
        Caller holds self._lock."""
        now = self._clock()
        ttl = self.cfg.session_ttl_minutes * 60
        for session in self._sessions.values():
            if session.store is not None and now - session.last_used > ttl:
                session.store = None
                session.expired = True
        # ponytail: session records (minus their stores) are never removed;
        # a few bytes per visitor is fine at portfolio scale.
        session = self._sessions.setdefault(session_id, _Session())
        session.last_used = now
        return session

    def _spend(self, kind: str) -> None:
        """Count one query or upload against today's cap. Caller holds self._lock."""
        today = datetime.datetime.fromtimestamp(self._clock(), datetime.timezone.utc).date()
        if today != self._day:
            self._day = today
            self._counts = {"query": 0, "upload": 0}
        # ponytail: counters live in memory and reset on restart; persist them
        # if a restart ever becomes an abuse vector.
        cap = self.cfg.daily_query_cap if kind == "query" else self.cfg.daily_upload_cap
        if self._counts[kind] >= cap:
            raise QuotaExceeded()
        self._counts[kind] += 1

    def answer(self, session_id: str, query: str) -> dict:
        if not query.strip():
            raise ServiceError("Ask a question first.", status=422)
        with self._lock:
            session = self._touch(session_id)
            self._spend("query")
            expired, session.expired = session.expired, False
            store = session.store if session.store is not None else self.sample_store
        result = self._ask(query, store, self.cfg, llm=_LazyLLM(self.cfg, self._llm_factory))
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "timings": result["debug"]["timings"],
            "expired": expired,
        }

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._touch(session_id).store = None
```

- [ ] **Step 4: Run the full suite.** Expected: all pass, including import purity for `docuchat.service`.
- [ ] **Step 5: Commit** with the message `feat: add service layer with sessions, daily quota, and lazy LLM`.

---

### Task 3: Uploads: caps, safe filenames, `ingest_upload()`, TTL

**Files:**
- Modify: `src/docuchat/service.py`
- Test: `tests/test_upload_caps.py` (create), `tests/test_service.py`

**Interfaces:**
- Consumes: Task 2's `Service`, `_LazyLLM`, `_touch`, `_spend`, `UploadRejected`. `ingest.load_directory(folder, cfg) -> list[dict]`. `classify.classify_pages(pages, llm, cfg) -> list[dict]`, which calls the LLM only when the heuristic returns "Unknown". `index.build_store(pages, cfg) -> Store`.
- Produces:
  - `check_upload(files: list[tuple[str, bytes]], cfg: Settings) -> None`, which raises `UploadRejected` with status 422, 413 or 415.
  - `safe_name(name: str, index: int, taken: set[str]) -> str`.
  - `build_upload_store(files: list[tuple[str, bytes]], llm, cfg: Settings) -> Store`.
  - `Service.__init__` gains `build_upload_store_fn=build_upload_store`.
  - `Service.ingest_upload(session_id: str, files: list[tuple[str, bytes]]) -> int`, returning the number of chunks indexed.

- [ ] **Step 1: Write the failing tests.** Create `tests/test_upload_caps.py`:

```python
"""Upload validation runs before Docling ever sees a file."""

from pathlib import Path

import pytest

from docuchat.config import Settings
from docuchat.service import UploadRejected, check_upload, safe_name

FIXTURE = Path(__file__).parent / "fixtures" / "cfpb_closing_disclosure.pdf"  # 6 pages


def pdf(name="cd.pdf"):
    return (name, FIXTURE.read_bytes())


def status_of(files, **overrides):
    with pytest.raises(UploadRejected) as exc:
        check_upload(files, Settings(**overrides))
    return exc.value.status


def test_accepts_a_pdf_within_caps():
    check_upload([pdf()], Settings())


def test_rejects_an_empty_upload():
    assert status_of([]) == 422


def test_rejects_a_non_pdf():
    assert status_of([("notes.txt", b"hello")]) == 415


# Review Focus 4
def test_rejects_a_corrupt_pdf():
    assert status_of([("bad.pdf", b"%PDF-1.7 this is not really a pdf")]) == 415


def test_size_cap_is_checked_before_parsing():
    # not parseable: a 413 proves the size check ran first
    huge = ("big.pdf", b"%PDF" + b"0" * (1024 * 1024))
    assert status_of([huge], max_upload_mb=1) == 413


def test_page_cap_counts_pages_across_files():
    check_upload([pdf()], Settings(max_upload_pages=6))
    assert status_of([pdf()], max_upload_pages=5) == 413
    assert status_of([pdf("a.pdf"), pdf("b.pdf")], max_upload_pages=11) == 413


# Review Focus 3
def test_safe_name_strips_paths_adds_suffix_and_dedupes():
    assert safe_name("../../etc/x.pdf", 0, set()) == "x.pdf"
    assert safe_name("scan", 1, set()) == "scan.pdf"
    assert safe_name("", 2, set()) == "upload_2.pdf"
    assert safe_name("x.pdf", 3, {"x.pdf"}) == "3_x.pdf"
```

Append to `tests/test_service.py`:

```python
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
```

Add `from pathlib import Path` and `UploadRejected` to `test_service.py`'s imports.

- [ ] **Step 2: Run them.** Expected: FAIL (`ImportError` for `check_upload`; `build_upload_store_fn` is an unexpected keyword).

- [ ] **Step 3: Implement** in `service.py`. Add the imports `import tempfile`, `from pathlib import Path`, `from docuchat.classify import classify_pages`, `from docuchat.index import Store, build_store, store_from_nodes` (merging with the existing index import) and `from docuchat.ingest import load_directory`.

```python
def check_upload(files: list[tuple[str, bytes]], cfg: Settings) -> None:
    """Reject an upload before Docling sees it: cheapest checks first."""
    import pypdfium2

    if not files:
        raise UploadRejected("Upload at least one PDF.", status=422)
    if sum(len(data) for _, data in files) > cfg.max_upload_mb * 1024 * 1024:
        raise UploadRejected(f"Uploads are limited to {cfg.max_upload_mb} MB in total.", status=413)
    pages = 0
    for name, data in files:
        if not data.startswith(b"%PDF"):
            raise UploadRejected(f"{name} is not a PDF. PDF files only.", status=415)
        try:
            pdf = pypdfium2.PdfDocument(data)
        except pypdfium2.PdfiumError as exc:
            raise UploadRejected(f"{name} could not be read as a PDF.", status=415) from exc
        pages += len(pdf)
        pdf.close()
    if pages > cfg.max_upload_pages:
        raise UploadRejected(
            f"Uploads are limited to {cfg.max_upload_pages} pages in total.", status=413)


def safe_name(name: str, index: int, taken: set[str]) -> str:
    """A filename safe to write into the temp dir: no path components, a .pdf
    suffix (load_directory only globs *.pdf), and unique within the upload."""
    base = Path(name).name or f"upload_{index}.pdf"
    if not base.lower().endswith(".pdf"):
        base += ".pdf"
    return base if base not in taken else f"{index}_{base}"


def build_upload_store(files: list[tuple[str, bytes]], llm, cfg: Settings) -> Store:
    """Docling -> classify -> chunk -> index, for one session's upload."""
    with tempfile.TemporaryDirectory() as tmp:
        taken: set[str] = set()
        for i, (name, data) in enumerate(files):
            filename = safe_name(name, i, taken)
            taken.add(filename)
            (Path(tmp) / filename).write_bytes(data)
        pages = load_directory(tmp, cfg)
    if not pages:
        raise UploadRejected("No readable text found in the upload.", status=422)
    return build_store(classify_pages(pages, llm, cfg), cfg)
```

In `Service.__init__`, add the keyword `build_upload_store_fn=build_upload_store`, stored as `self._build_upload_store`, and add `self._ingest_slot = threading.Semaphore(1)`. Then add the method:

```python
    def ingest_upload(self, session_id: str, files: list[tuple[str, bytes]]) -> int:
        """Replace the session's documents with `files`. Returns the chunk count.
        On any failure the session keeps the store it had."""
        check_upload(files, self.cfg)
        with self._lock:
            self._touch(session_id)
            self._spend("upload")
        # ponytail: one ingestion at a time on 2 vCPU; a queue with progress if visitors wait too long
        with self._ingest_slot:
            store = self._build_upload_store(files, _LazyLLM(self.cfg, self._llm_factory), self.cfg)
        with self._lock:
            session = self._touch(session_id)
            session.store = store
            session.expired = False
        return len(store.nodes)
```

- [ ] **Step 4: Run the full suite.** Expected: all pass.
- [ ] **Step 5: Commit** with the message `feat: add upload caps and per-session upload ingestion`.

---

### Task 4: FastAPI app, access gate, and error mapping

**Files:**
- Modify: `pyproject.toml` (add `fastapi==0.142.2`, `uvicorn==0.54.0`, `python-multipart==0.0.32` to `dependencies`; add `httpx==0.28.1` to `dev`), `tests/conftest.py`, `tests/test_import_purity.py`
- Create: `src/docuchat/api.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: Task 2/3's `Service` methods and the `ServiceError` hierarchy (`.status`, `str()`).
- Produces:
  - `create_app(service, access_key: str | None) -> FastAPI`.
  - Module-level `app = create_app(Service.from_env(), os.environ.get("DOCUCHAT_ACCESS_KEY") or None)`.
  - `COOKIE = "docuchat_key"`.
  - Routes: `GET /api/health` returns `{"status": "ok", "gated": bool}`; `POST /api/sessions` returns `{"session_id": str}`; `POST /api/ask` takes a JSON body `{session_id, query}` and returns `Service.answer`'s dict; `POST /api/sessions/{session_id}/documents` takes multipart `files` and returns `{"chunks": int}`; `POST /api/sessions/{session_id}/reset` returns `{"ok": true}`.
  - conftest fixture `fake_service`, which Task 5 reuses.

- [ ] **Step 1: Edit `pyproject.toml` and install:** `.venv/bin/pip install -e .[dev]`.

- [ ] **Step 2: Write the failing tests.** Add to `tests/conftest.py`:

```python
class _FakeService:
    """Stands in for docuchat.service.Service in API/UI tests."""

    def __init__(self):
        from docuchat.config import Settings

        self.cfg = Settings()
        self.ensured = False
        self.uploads = []
        self.resets = []
        self.error = None  # set to a ServiceError to make answer/ingest raise it

    def ensure_sample(self):
        self.ensured = True

    def new_session(self):
        return "sid"

    def answer(self, session_id, query):
        if self.error:
            raise self.error
        return {"answer": "a", "sources": [], "timings": {"total": 1.0}, "expired": False}

    def ingest_upload(self, session_id, files):
        if self.error:
            raise self.error
        self.uploads.append((session_id, files))
        return 3

    def reset(self, session_id):
        self.resets.append(session_id)


@pytest.fixture
def fake_service():
    return _FakeService()
```

Create `tests/test_api.py`:

```python
"""Tests for docuchat.api: the access gate, routes, and error mapping."""

import pytest
from fastapi.testclient import TestClient

from docuchat.api import COOKIE, create_app
from docuchat.service import (
    LLMError, LLMNotConfigured, QuotaExceeded, ServiceError, UploadRejected,
)


@pytest.fixture
def client(fake_service):
    with TestClient(create_app(fake_service, "secret"), base_url="https://testserver") as c:
        yield c


@pytest.fixture
def authed(client):
    client.cookies.set(COOKIE, "secret")
    return client


def test_startup_builds_the_sample_store(client, fake_service):
    assert fake_service.ensured


def test_no_key_is_forbidden(client):
    response = client.post("/api/sessions")
    assert response.status_code == 403
    assert "invitation" in response.text


def test_wrong_key_or_cookie_is_forbidden(client):
    assert client.get("/?key=nope", follow_redirects=False).status_code == 403
    client.cookies.set(COOKIE, "nope")
    assert client.post("/api/sessions").status_code == 403


# Review Focus 2
def test_key_link_sets_cookie_and_redirects_relative_without_the_key(client):
    response = client.get("/?key=secret&x=1", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/?x=1"
    set_cookie = response.headers["set-cookie"]
    assert f"{COOKIE}=secret" in set_cookie
    assert "HttpOnly" in set_cookie and "Secure" in set_cookie
    assert "samesite=lax" in set_cookie.lower()


# Review Focus 1
def test_post_with_key_is_served_not_redirected(client):
    response = client.post("/api/sessions?key=secret", follow_redirects=False)
    assert response.status_code == 200
    assert COOKIE in response.headers["set-cookie"]


def test_cookie_grants_access(authed):
    assert authed.post("/api/sessions").json() == {"session_id": "sid"}


def test_health_is_exempt_and_reports_the_gate(client):
    assert client.get("/api/health").json() == {"status": "ok", "gated": True}


def test_unset_key_leaves_the_gate_open(fake_service):
    with TestClient(create_app(fake_service, None), base_url="https://testserver") as c:
        assert c.post("/api/sessions").status_code == 200
        assert c.get("/api/health").json()["gated"] is False


def test_ask_returns_the_service_result(authed):
    response = authed.post("/api/ask", json={"session_id": "sid", "query": "q"})
    assert response.json() == {
        "answer": "a", "sources": [], "timings": {"total": 1.0}, "expired": False}


@pytest.mark.parametrize("error, status", [
    (QuotaExceeded(), 429),
    (LLMNotConfigured(), 503),
    (LLMError(), 502),
    (UploadRejected("Uploads are limited to 10 MB in total.", status=413), 413),
    (UploadRejected("x.txt is not a PDF. PDF files only.", status=415), 415),
    (ServiceError("Ask a question first.", status=422), 422),
])
def test_service_errors_map_to_status_and_message(authed, fake_service, error, status):
    fake_service.error = error
    response = authed.post("/api/ask", json={"session_id": "sid", "query": "q"})
    assert response.status_code == status
    assert response.json()["detail"] == str(error)


def test_upload_passes_names_and_bytes_to_the_service(authed, fake_service):
    response = authed.post(
        "/api/sessions/s1/documents",
        files=[("files", ("a.pdf", b"%PDF-1", "application/pdf"))],
    )
    assert response.json() == {"chunks": 3}
    assert fake_service.uploads == [("s1", [("a.pdf", b"%PDF-1")])]


def test_upload_rejection_maps_to_its_status(authed, fake_service):
    fake_service.error = UploadRejected("x.txt is not a PDF. PDF files only.", status=415)
    response = authed.post("/api/sessions/s1/documents",
                           files=[("files", ("x.txt", b"hi", "text/plain"))])
    assert response.status_code == 415


def test_reset(authed, fake_service):
    assert authed.post("/api/sessions/s1/reset").json() == {"ok": True}
    assert fake_service.resets == ["s1"]
```

Add `"docuchat.api"` to `tests/test_import_purity.py`.

- [ ] **Step 3: Run them.** Expected: FAIL (`ImportError` for `docuchat.api`).

- [ ] **Step 4: Implement `src/docuchat/api.py`:**

```python
"""HTTP surface: the access-token gate and the /api/* routes.

create_app(service, access_key) is the test seam. uvicorn imports the
module-level `app`, built from the environment. Nothing heavy happens at
import: the sample store is built in the lifespan hook.
"""

import hmac
import json
import logging
import os
from contextlib import asynccontextmanager, contextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

from docuchat.service import Service, ServiceError

COOKIE = "docuchat_key"
COOKIE_MAX_AGE = 30 * 24 * 3600
FORBIDDEN_PAGE = (
    "<!doctype html><title>docuchat</title>"
    "<p>This demo is by invitation. Please use the link from the portfolio page.</p>"
)

log = logging.getLogger("docuchat")


class AskRequest(BaseModel):
    session_id: str
    query: str


@contextmanager
def _http_errors():
    try:
        yield
    except ServiceError as exc:
        raise HTTPException(exc.status, str(exc)) from exc


def create_app(service: Service, access_key: str | None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        if not access_key:
            log.warning("DOCUCHAT_ACCESS_KEY is unset: the access gate is OFF")
        service.ensure_sample()
        yield

    app = FastAPI(title="docuchat", lifespan=lifespan)

    def valid(candidate: str) -> bool:
        return bool(candidate) and hmac.compare_digest(candidate.encode(), access_key.encode())

    @app.middleware("http")
    async def gate(request: Request, call_next):
        if not access_key or request.url.path == "/api/health":
            return await call_next(request)
        if valid(request.cookies.get(COOKIE, "")):
            return await call_next(request)
        if valid(request.query_params.get("key", "")):
            if request.method == "GET":
                # Relative Location: behind HF's TLS proxy request.url says http://.
                url = request.url.remove_query_params("key")
                response = RedirectResponse(
                    url.path + (f"?{url.query}" if url.query else ""), status_code=303)
            else:
                # A 303 would turn a POST into a GET and drop its body.
                response = await call_next(request)
            response.set_cookie(COOKIE, access_key, max_age=COOKIE_MAX_AGE,
                                httponly=True, secure=True, samesite="lax")
            return response
        return HTMLResponse(FORBIDDEN_PAGE, status_code=403)

    # Sync handlers: FastAPI runs them in its threadpool, so a slow ask() or
    # ingestion doesn't block the event loop.
    @app.get("/api/health")
    def health():
        return {"status": "ok", "gated": bool(access_key)}

    @app.post("/api/sessions")
    def new_session():
        return {"session_id": service.new_session()}

    @app.post("/api/ask")
    def api_ask(body: AskRequest):
        with _http_errors():
            result = service.answer(body.session_id, body.query)
        log.info(json.dumps({"event": "ask", "chunks": len(result["sources"]),
                             "timings": result["timings"]}))
        return result

    @app.post("/api/sessions/{session_id}/documents")
    def upload(session_id: str, files: list[UploadFile] = File(...)):
        # Read at most cap+1 bytes per file: check_upload rejects anything bigger.
        limit = service.cfg.max_upload_mb * 1024 * 1024 + 1
        payload = [(f.filename or "", f.file.read(limit)) for f in files]
        with _http_errors():
            chunks = service.ingest_upload(session_id, payload)
        return {"chunks": chunks}

    @app.post("/api/sessions/{session_id}/reset")
    def reset(session_id: str):
        service.reset(session_id)
        return {"ok": True}

    return app


logging.basicConfig(level=logging.INFO)
app = create_app(Service.from_env(), os.environ.get("DOCUCHAT_ACCESS_KEY") or None)
```

- [ ] **Step 5: Run the full suite.** Expected: all pass, including import purity for `docuchat.api`.
- [ ] **Step 6: Commit** with the message `feat: add FastAPI app with access-token gate and error mapping`.

---

### Task 5: Gradio UI mounted at `/`

**Files:**
- Modify: `pyproject.toml` (add `gradio==6.29.0`), `src/docuchat/api.py` (mount the UI), `tests/conftest.py` (disable Gradio analytics), `tests/test_import_purity.py`, `tests/test_api.py`
- Create: `src/docuchat/ui.py`
- Test: `tests/test_ui.py`

**Interfaces:**
- Consumes: `Service.answer`, `Service.ingest_upload`, `Service.reset`, `Service.cfg.max_upload_mb`, and `ServiceError`. Gradio's `gr.Request.session_hash` serves as the session ID.
- Produces: `ui.build(service) -> gr.Blocks` and `ui.format_answer(result: dict) -> str`. `create_app` mounts the UI with `gr.mount_gradio_app(app, ui.build(service), path="/", max_file_size=f"{service.cfg.max_upload_mb}mb")` after the `/api` routes.

- [ ] **Step 1: Edit `pyproject.toml` and install:** `.venv/bin/pip install -e .[dev]`. At the very top of `tests/conftest.py`, before any import that could load gradio, add:

```python
import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no network from the default suite
```

- [ ] **Step 2: Write the failing tests.** Create `tests/test_ui.py`:

```python
"""Tests for docuchat.ui: construction and answer formatting only."""

import gradio as gr

from docuchat.ui import build, format_answer

SOURCE = {"filename": "cd.pdf", "page_number": 2, "doc_type": "Closing Disclosure",
          "section_title": "", "score": 0.9, "preview": "Loan Amount $162,000..."}


def test_build_returns_blocks(fake_service):
    assert isinstance(build(fake_service), gr.Blocks)


def test_answer_lists_sources():
    text = format_answer({"answer": "It is $162,000.", "sources": [SOURCE], "expired": False})
    assert text.startswith("It is $162,000.")
    assert "cd.pdf, page 2: Loan Amount $162,000..." in text


def test_expired_session_is_announced():
    text = format_answer({"answer": "x", "sources": [], "expired": True})
    assert "session expired" in text.lower()
```

Append to `tests/test_api.py`:

```python
def test_ui_is_served_behind_the_gate(client, authed):
    assert authed.get("/").status_code == 200
    client.cookies.clear()
    assert client.get("/").status_code == 403
```

Add `"docuchat.ui"` to `tests/test_import_purity.py`.

- [ ] **Step 3: Run them.** Expected: FAIL (`ImportError` for `docuchat.ui`; `/` returns 404 with the cookie).

- [ ] **Step 4: Implement `src/docuchat/ui.py`:**

```python
"""Gradio UI. Every handler calls one Service method; there is no pipeline
logic here. Layout follows the notebook's Gradio app."""

from pathlib import Path

import gradio as gr

from docuchat.service import Service, ServiceError

EXAMPLES = [
    "What is the loan amount and interest rate on the fixed-rate sample loan?",
    "What are the total closing costs?",
    "Is there a prepayment penalty?",
]


def format_answer(result: dict) -> str:
    parts = []
    if result["expired"]:
        parts.append("_Your session expired, so this answer uses the sample documents._\n")
    parts.append(result["answer"])
    if result["sources"]:
        parts.append("\n**Sources**")
        parts += [f"- {s['filename']}, page {s['page_number']}: {s['preview']}"
                  for s in result["sources"]]
    return "\n".join(parts)


def build(service: Service) -> gr.Blocks:
    def chat(message, history, request: gr.Request):
        try:
            return format_answer(service.answer(request.session_hash, message))
        except ServiceError as exc:
            return f"⚠️ {exc}"

    def process(paths, request: gr.Request):
        if not paths:
            return "Choose one or more PDFs first."
        files = [(Path(p).name, Path(p).read_bytes()) for p in paths]
        try:
            chunks = service.ingest_upload(request.session_hash, files)
        except ServiceError as exc:
            return f"⚠️ {exc}"
        return f"Indexed {chunks} chunks from {len(files)} file(s). Questions now use your documents."

    def back_to_sample(request: gr.Request):
        service.reset(request.session_hash)
        return "Using the sample documents."

    with gr.Blocks(title="docuchat") as demo:
        gr.Markdown("# docuchat\nAsk questions about sample CFPB mortgage documents, "
                    "or upload your own PDFs.")
        with gr.Row():
            with gr.Column(scale=1):
                upload = gr.File(label="Upload PDF(s)", file_types=[".pdf"],
                                 file_count="multiple")
                process_btn = gr.Button("Process & Index", variant="primary")
                sample_btn = gr.Button("Back to sample documents")
                status = gr.Textbox(label="Status", interactive=False, lines=3,
                                    value="Using the sample documents.")
            with gr.Column(scale=3):
                gr.ChatInterface(chat, examples=EXAMPLES)
        process_btn.click(process, inputs=upload, outputs=status)
        sample_btn.click(back_to_sample, outputs=status)
    return demo
```

In `api.py`, add `import gradio as gr` and `from docuchat import ui`. As the last line of `create_app` before `return app`, add:

```python
    # Mounted last so the /api routes above take precedence over the UI at /.
    gr.mount_gradio_app(app, ui.build(service), path="/",
                        max_file_size=f"{service.cfg.max_upload_mb}mb")
```

The gate middleware already wraps the mount, so the UI and its upload/queue endpoints are gated too.

- [ ] **Step 5: Run the full suite.** Expected: all pass, including import purity for `docuchat.ui` and `docuchat.api`, which now imports gradio. If `test_ui_is_served_behind_the_gate` fails because Gradio 6 expects the page to be served from a different path under `mount_gradio_app`, fix the mount, not the test.

- [ ] **Step 6: Run the app and look at it.** Run `.venv/bin/uvicorn docuchat.api:app --port 7860` from the repo root without `DOCUCHAT_ACCESS_KEY`, then open `http://localhost:7860`. This needs the local model cache; it is the first time the real sample store builds. Expected:
  - the UI renders and the status reads "Using the sample documents."
  - asking a question shows "⚠️ LLM not configured." if no key is in the environment
  - uploading `tests/fixtures/cfpb_closing_disclosure.pdf` succeeds, because the heuristic classifies it
  - "Back to sample documents" works

  Stop the server.

- [ ] **Step 7: Commit** with the message `feat: add Gradio UI mounted on the FastAPI app`.

---

### Task 6: Latency benchmark

**Files:**
- Create: `src/docuchat/bench.py`
- Modify: `tests/test_import_purity.py`
- Test: `tests/test_bench.py`

**Interfaces:**
- Consumes: the `/api/sessions` and `/api/ask` routes and `timings` (Tasks 1 and 4). `evaluate.QUESTIONS_PATH`, `evaluate.RESULTS_DIR`, `evaluate.load_questions` and `evaluate._git_commit`. `service.SERVED_ARM`.
- Produces:
  - `percentile(values: list[float], p: float) -> float`, nearest-rank.
  - `summarize(samples: list[dict[str, float]]) -> dict[str, tuple[float, float]]`, mapping each stage to `(p50, p95)` with `samples[0]` excluded.
  - `run(url: str, key: str | None, questions: list[dict], transport=None) -> list[dict[str, float]]`, where each sample is the server `timings` plus `"client"` ms.
  - `render(url, samples, commit, date) -> str`.
  - `main(argv) -> int`, the CLI `python -m docuchat.bench --url URL [--key KEY] [--out PATH]`.

- [ ] **Step 1: Write the failing tests.** Create `tests/test_bench.py`:

```python
"""Tests for docuchat.bench: percentiles, cold-request exclusion, the HTTP loop."""

import json

import httpx

from docuchat.bench import percentile, render, run, summarize


def test_percentile_nearest_rank():
    assert percentile([5.0], 50) == 5.0
    assert percentile([5.0], 95) == 5.0
    values = [float(v) for v in range(1, 21)]
    assert percentile(values, 50) == 10.0
    assert percentile(values, 95) == 19.0


def test_summarize_excludes_the_cold_first_request_and_absent_stages():
    samples = [
        {"total": 1000.0, "client": 1000.0},
        {"total": 10.0, "client": 12.0},
        {"total": 20.0, "client": 22.0},
    ]
    summary = summarize(samples)
    assert summary["total"] == (10.0, 20.0)
    assert "rewrite" not in summary


def test_render_has_table_and_cold_line():
    samples = [{"total": 900.0, "client": 950.0}, {"total": 10.0, "client": 12.0}]
    text = render("https://x.hf.space", samples, "abc123", "2026-10-01")
    assert "| total | 10 | 10 |" in text
    assert "First request (cold): 950 ms" in text
    assert "+rerank" in text and "abc123" in text


def test_run_sends_the_key_on_every_request_and_collects_timings():
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.path == "/api/sessions":
            return httpx.Response(200, json={"session_id": "sid"})
        body = json.loads(request.content)
        assert body == {"session_id": "sid", "query": "Q1"}
        return httpx.Response(200, json={"timings": {"total": 5.0, "llm": 3.0}})

    samples = run("https://x", "secret", [{"question": "Q1"}],
                  transport=httpx.MockTransport(handler))
    assert all(r.url.params["key"] == "secret" for r in seen)
    assert samples[0]["total"] == 5.0 and samples[0]["client"] >= 0
```

Add `"docuchat.bench"` to `tests/test_import_purity.py`.

- [ ] **Step 2: Run them.** Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement `src/docuchat/bench.py`:**

```python
"""Latency benchmark: p50/p95 per pipeline stage against a running docuchat.

    python -m docuchat.bench --url https://<space>.hf.space --key KEY

An HTTP client, because free Spaces have no shell. Runs every question in
eval/questions.yaml sequentially in one session. The first request is
reported separately as cold and excluded from the percentiles. Each run
spends len(questions) of the server's daily query cap.
"""

import argparse
import datetime
import math
import time
from pathlib import Path

import httpx

from docuchat.evaluate import QUESTIONS_PATH, RESULTS_DIR, _git_commit, load_questions
from docuchat.service import SERVED_ARM

STAGES = ("rewrite", "decompose", "retrieve", "rerank", "clean_context", "llm", "total", "client")


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile; defined for any non-empty list."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def summarize(samples: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    warm = samples[1:]
    summary = {}
    for stage in STAGES:
        values = [s[stage] for s in warm if stage in s]
        if values:
            summary[stage] = (percentile(values, 50), percentile(values, 95))
    return summary


def run(url: str, key: str | None, questions: list[dict], transport=None) -> list[dict[str, float]]:
    # The key rides on every request (the gate accepts it on POSTs), so the
    # bench never depends on the Secure cookie, which plain-http localhost drops.
    params = {"key": key} if key else None
    samples = []
    with httpx.Client(base_url=url, params=params, timeout=300, transport=transport) as client:
        session_id = client.post("/api/sessions").raise_for_status().json()["session_id"]
        for q in questions:
            start = time.perf_counter()
            response = client.post("/api/ask", json={"session_id": session_id, "query": q["question"]})
            response.raise_for_status()
            sample = dict(response.json()["timings"])
            sample["client"] = (time.perf_counter() - start) * 1000
            samples.append(sample)
    return samples


def render(url: str, samples: list[dict[str, float]], commit: str, date: str) -> str:
    lines = [
        "# Latency",
        "",
        f"- URL: {url}",
        f"- Date: {date}",
        f"- Commit: {commit}",
        f"- Arm: {SERVED_ARM}",
        f"- Requests: 1 cold + {len(samples) - 1} warm, sequential",
        "- `client` is end-to-end from the benchmark machine, network included; "
        "every other row is server-side.",
        "",
        "| Stage | p50 (ms) | p95 (ms) |",
        "|---|---|---|",
    ]
    lines += [f"| {stage} | {p50:.0f} | {p95:.0f} |" for stage, (p50, p95) in summarize(samples).items()]
    cold = samples[0]
    lines += ["", f"First request (cold): {cold['client']:.0f} ms client-side, "
                  f"{cold.get('total', 0):.0f} ms server-side."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m docuchat.bench")
    parser.add_argument("--url", required=True)
    parser.add_argument("--key")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "latency.md")
    args = parser.parse_args(argv)
    samples = run(args.url, args.key, load_questions(QUESTIONS_PATH))
    report = render(args.url, samples, _git_commit(), datetime.date.today().isoformat())
    args.out.write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the full suite.** Expected: all pass.
- [ ] **Step 5: Commit** with the message `feat: add HTTP latency benchmark with per-stage p50/p95`.

---

### Task 7: Dockerfile

**Files:**
- Create: `Dockerfile`, `.dockerignore`

**Interfaces:**
- Consumes: `docuchat.api:app`, `eval/corpus`, `eval/nodes`, `eval/questions.yaml`, and `tests/fixtures/cfpb_closing_disclosure.pdf` (used only to warm the Docling models at build time).
- Produces: an image serving on port 7860 as uid 1000, with every model already on disk.

- [ ] **Step 1: Write `.dockerignore`.** `.env` must never reach the image or the Space.

```
.git
.github
.venv
.env
**/__pycache__
.pytest_cache
docs
notebooks
eval/results
complete_mortgage_rag_pipeline.py
```

- [ ] **Step 2: Write `Dockerfile`:**

```dockerfile
FROM python:3.11-slim

# HF Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    GRADIO_ANALYTICS_ENABLED=False \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# CPU-only torch first, so sentence-transformers doesn't pull the CUDA build (~2 GB)
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user pyproject.toml ./
COPY --chown=user src ./src
RUN pip install --no-cache-dir --user .

# Fetch every model at build time so a wake-up loads from disk. Converting the
# fixture pulls exactly the Docling models (layout, tables, OCR) runtime uses.
COPY --chown=user tests/fixtures/cfpb_closing_disclosure.pdf /tmp/warm.pdf
RUN python -c "from docuchat.config import Settings; \
from docuchat.models import get_embed_model, get_cross_encoder; \
from docuchat.ingest import load_pdf; \
cfg = Settings(); get_embed_model(cfg); get_cross_encoder(cfg); load_pdf('/tmp/warm.pdf', cfg)"

COPY --chown=user eval/corpus ./eval/corpus
COPY --chown=user eval/nodes ./eval/nodes
COPY --chown=user eval/questions.yaml ./eval/questions.yaml

EXPOSE 7860
CMD ["uvicorn", "docuchat.api:app", "--host", "0.0.0.0", "--port", "7860", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
```

- [ ] **Step 3: Build.** `docker build -t docuchat .` Expected: success. Record the size with `docker image ls docuchat`; the expected size is roughly 4–5 GB. If it is far larger, check that CUDA torch did not sneak in: `docker run --rm docuchat python -c "import torch; print(torch.version.cuda)"` must print `None`.

- [ ] **Step 4: Verify success criterion 1, offline, with no key.**

```bash
docker run -d --name docuchat -p 7860:7860 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 docuchat
until curl -sf localhost:7860/api/health; do sleep 5; done   # {"status":"ok","gated":false}
SID=$(curl -s -X POST localhost:7860/api/sessions | python -c "import sys,json;print(json.load(sys.stdin)['session_id'])")
curl -s -X POST localhost:7860/api/ask -H 'content-type: application/json' \
  -d "{\"session_id\":\"$SID\",\"query\":\"What is the loan amount?\"}"   # 503 {"detail":"LLM not configured."}
docker logs docuchat 2>&1 | grep -i "gate is OFF"
docker rm -f docuchat
```

Expected: health is ok, `ask` returns 503 (proving the sample store was built and retrieval ran offline), and the log has the gate warning. Also open `http://localhost:7860` in a browser before removing the container and confirm the UI renders.

- [ ] **Step 5: Verify the gate in the container.**

```bash
docker run -d --name docuchat -p 7860:7860 -e DOCUCHAT_ACCESS_KEY=testkey docuchat
until curl -sf localhost:7860/api/health; do sleep 5; done   # "gated":true
curl -s -o /dev/null -w '%{http_code}\n' localhost:7860/                # 403
curl -s -o /dev/null -w '%{http_code} %{redirect_url}\n' 'localhost:7860/?key=testkey'   # 303 .../
docker rm -f docuchat
```

- [ ] **Step 6: Commit** with the message `feat: add CPU-only Dockerfile with models baked in`.

---

### Task 8: Deploy workflow and Space README

**Files:**
- Create: `deploy/space-README.md`, `.github/workflows/deploy.yml`

**Interfaces:**
- Consumes: the Task 7 `Dockerfile` and the paths it copies. A GitHub secret `HF_TOKEN` and a repo variable `HF_SPACE` (`<user>/<space>`), both set in Task 9.
- Produces: a `workflow_dispatch` job that replaces the Space's contents with exactly what the Docker build needs.

- [ ] **Step 1: Write `deploy/space-README.md`:**

```markdown
---
title: docuchat
emoji: 📄
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

docuchat: chat with PDF documents. Source and evaluation: see the GitHub repository.
```

- [ ] **Step 2: Write `.github/workflows/deploy.yml`:**

```yaml
name: deploy

on:
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install huggingface_hub==1.33.0
      - name: Stage exactly what the Docker build needs
        run: |
          mkdir -p _space/tests/fixtures _space/eval
          cp Dockerfile .dockerignore pyproject.toml _space/
          cp -r src _space/
          cp -r eval/corpus eval/nodes eval/questions.yaml _space/eval/
          cp tests/fixtures/cfpb_closing_disclosure.pdf _space/tests/fixtures/
          cp deploy/space-README.md _space/README.md
      - name: Upload to the Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: >
          hf upload "${{ vars.HF_SPACE }}" _space . --repo-type space
          --delete "*" --commit-message "deploy ${{ github.sha }}"
```

- [ ] **Step 3: Verify the staging locally.** Run the "Stage" block's commands in the scratchpad against a checkout of the branch, then `docker build` from `_space/`. Expected: the build succeeds, which proves the staged tree is complete. Also check the YAML parses: `.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/deploy.yml'))"`.

- [ ] **Step 4: Commit** with the message `ci: add manual deploy workflow to Hugging Face Spaces`.

---

### Task 9: Go live — BLOCKED until API keys

Do not start this task until the owner has an LLM API key (Anthropic, or Gemini with `DOCUCHAT_LLM_PROVIDER=gemini`) and a Hugging Face account. Stop and ask the owner at each step marked **owner**.

- [ ] **Step 1 (owner): Create the Space.** On huggingface.co: New Space, SDK **Docker**, hardware **CPU basic** (free), visibility **public**. The access gate is the access control; a private Space would require visitors to have HF accounts. Add these Space **secrets**:
  - the LLM key (`ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY`)
  - `DOCUCHAT_ACCESS_KEY`, generated with `python -c "import secrets; print(secrets.token_urlsafe(24))"`
  - `DOCUCHAT_LLM_PROVIDER`, only if not `anthropic`

- [ ] **Step 2 (owner): Wire GitHub.** Add the repo secret `HF_TOKEN` (an HF token with write access to the Space) and the repo variable `HF_SPACE=<user>/<space>`. The workflow only exists on GitHub once this branch is pushed.

- [ ] **Step 3: Deploy.** Run `gh workflow run deploy.yml --ref feat/docuchat-service`, then watch the build in the Space's logs until it reports Running.

- [ ] **Step 4: Verify live.**
  - `curl -s https://<space>.hf.space/api/health` gives `"gated":true`.
  - `/` without the key gives 403.
  - In a browser, `https://<space>.hf.space/?key=<KEY>` lands on the UI with the key gone from the address bar.
  - A sample question gets a cited answer.
  - Uploading `tests/fixtures/cfpb_closing_disclosure.pdf` makes answers cite `cfpb_closing_disclosure.pdf`.
  - **Isolation:** open the key link in a private window; it still answers from the sample documents.

- [ ] **Step 5: Benchmark.** Run `.venv/bin/python -m docuchat.bench --url https://<space>.hf.space --key <KEY>`. It spends about 43 of the day's 200 queries. Commit `eval/results/latency.md`.

- [ ] **Step 6 (owner): Confirm the caps against real spend.** After a day of use, check the provider console. Classification costs at most one LLM call per uploaded file, and only when the heuristic can't label it, so queries dominate. Adjust `DOCUCHAT_DAILY_QUERY_CAP` / `DOCUCHAT_DAILY_UPLOAD_CAP` as Space variables if needed; no code change.

- [ ] **Step 7: Commit** the benchmark results with the message `docs: add live p50/p95 latency from the HF Space`.

---

## Not in this plan

- Index persistence (dropped in the spec).
- Per-IP rate limiting, and persisting quota counters or sessions across restarts.
- The `full` arm in the demo.
- A Docker build in CI.
- A custom "warming up" page: HF Spaces shows its own loading screen while a sleeping Space wakes, and the app cannot render anything before uvicorn is up.
- README, architecture diagram, demo recording (sub-project 4).
