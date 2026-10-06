"""The service layer: sessions with an idle TTL, a daily spend cap, and the
calls the API and UI make.

api.py and ui.py call this module and nothing below it, so the UI holds no
pipeline logic and the REST tests exercise the same path the UI uses.
"""

import datetime
import logging
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

from docuchat.classify import classify_pages
from docuchat.config import Settings
from docuchat.evaluate import CORPUS_DIR, INGEST_FIELDS, NODES_DIR, arm_settings, load_snapshot
from docuchat.index import Store, build_store, store_from_nodes
from docuchat.ingest import load_directory
from docuchat.models import get_llm
from docuchat.pipeline import ask, shrink_prompt

logger = logging.getLogger(__name__)

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
            # a free-tier API's 503s/429s fall back to a local GGUF when one is set
            if not self._cfg.gguf_path or self._cfg.llm_provider == "llamacpp":
                raise LLMError() from exc
            logger.warning("%s failed (%s); falling back to the local model",
                           self._cfg.llm_provider, exc)
            try:
                local = self._factory(replace(self._cfg, llm_provider="llamacpp", llm_model=""))
                return local.complete(shrink_prompt(prompt, self._cfg.local_max_context_tokens))
            except Exception as fallback_exc:
                raise LLMError() from fallback_exc


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
    base = Path(name.replace("\x00", "")).name or f"upload_{index}.pdf"
    if not base.lower().endswith(".pdf"):
        base += ".pdf"
    base = base[:-4][:100] + base[-4:]  # keep well under the filesystem name limit
    candidate = base
    while candidate in taken:
        candidate = f"{index}_{candidate}"
    return candidate


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


@dataclass
class _Session:
    store: object = None  # None means the shared sample store
    last_used: float = 0.0
    expired: bool = False


class Service:
    def __init__(self, cfg: Settings, sample_store=None, *, clock=time.time,
                 llm_factory=get_llm, ask_fn=ask,
                 build_upload_store_fn=build_upload_store):
        self.cfg = cfg
        self.sample_store = sample_store
        self._clock = clock
        self._llm_factory = llm_factory
        self._ask = ask_fn
        self._build_upload_store = build_upload_store_fn
        self._ingest_slot = threading.Semaphore(1)
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
            # The snapshot is prebuilt, so ingestion settings (do_ocr, ...) only
            # apply to uploads; check it against the defaults it was built with.
            defaults = Settings()
            built_with = replace(self.cfg, **{f: getattr(defaults, f) for f in INGEST_FIELDS})
            nodes = load_snapshot(NODES_DIR / f"{SAMPLE_PROFILE}.jsonl", CORPUS_DIR, built_with)
            self.sample_store = store_from_nodes(nodes, self.cfg)

    def new_session(self) -> str:
        return uuid.uuid4().hex

    def _touch(self, session_id: str) -> _Session:
        """Sweep idle sessions, then return (creating if new) this one.
        Caller holds self._lock."""
        now = self._clock()
        ttl = self.cfg.session_ttl_minutes * 60
        for sid, session in list(self._sessions.items()):
            if now - session.last_used <= ttl:
                continue
            if session.store is not None:
                session.store = None
                session.expired = True
            elif not session.expired:
                del self._sessions[sid]
        # ponytail: the sweep is O(sessions) under the global lock, and records
        # with a pending expiry notice live until that session's next answer.
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
        if len(query) > self.cfg.max_query_chars:
            raise ServiceError("Question is too long.", status=422)
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
