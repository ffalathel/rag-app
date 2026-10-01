"""HTTP surface: the access-token gate and the /api/* routes.

create_app(service, access_key) is the test seam. uvicorn imports the
module-level `app`, built from the environment. Nothing heavy happens at
import: the sample store is built in the lifespan hook.
"""

import hmac
import json
import logging
import os
import re
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


class _RedactKey(logging.Filter):
    """Keeps the access key out of uvicorn's access log (it logs the request line)."""

    def filter(self, record):
        if isinstance(record.args, tuple):
            record.args = tuple(
                re.sub(r'(key=)[^&\s"]+', r"\1[redacted]", a) if isinstance(a, str) else a
                for a in record.args)
        return True


logging.getLogger("uvicorn.access").addFilter(_RedactKey())


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
                    "/" + url.path.lstrip("/") + (f"?{url.query}" if url.query else ""), status_code=303)
            else:
                # A 303 would turn a POST into a GET and drop its body.
                response = await call_next(request)
            # ponytail: the cookie holds the shared access key itself; no per-visitor
            # revocation short of rotating the key. Upgrade: a signed random session token.
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
