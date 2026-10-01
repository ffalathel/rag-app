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
