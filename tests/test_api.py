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
    response = client.get("/?key=nope", follow_redirects=False)
    assert response.status_code == 403
    assert "nope" not in response.text
    client.cookies.set(COOKIE, "nope")
    assert client.post("/api/sessions").status_code == 403


# Review Focus 2
def test_key_link_sets_cookie_and_redirects_relative_without_the_key(client):
    response = client.get("/?key=secret&x=1", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/?x=1"
    assert "secret" not in response.headers["location"]
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


def test_redirect_is_never_protocol_relative(client):
    response = client.get("///evil.example/?key=secret", follow_redirects=False)
    assert response.headers["location"] == "/evil.example/"


def test_access_log_redacts_the_key():
    import logging

    record = logging.LogRecord(
        "uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:1", "GET", "/?key=secret&x=1", "1.1", 303), None)
    for f in logging.getLogger("uvicorn.access").filters:
        f.filter(record)
    assert "secret" not in record.getMessage()
    assert "x=1" in record.getMessage()


def test_ui_is_served_behind_the_gate(client, authed):
    assert authed.get("/").status_code == 200
    client.cookies.clear()
    assert client.get("/").status_code == 403


def test_gradio_upload_and_queue_are_gated(client):
    assert client.post("/gradio_api/upload", files={"files": ("a.pdf", b"x")}).status_code == 403
    assert client.get("/gradio_api/queue/data").status_code == 403
