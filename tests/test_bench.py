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

    samples, failed = run("https://x", "secret", [{"question": "Q1"}],
                          transport=httpx.MockTransport(handler))
    assert all(r.url.params["key"] == "secret" for r in seen)
    assert samples[0]["total"] == 5.0 and samples[0]["client"] >= 0
    assert failed == 0


def test_run_counts_and_skips_failed_asks():
    def handler(request):
        if request.url.path == "/api/sessions":
            return httpx.Response(200, json={"session_id": "sid"})
        if json.loads(request.content)["query"] == "bad":
            return httpx.Response(502, json={"detail": "Model service error, try again."})
        return httpx.Response(200, json={"timings": {"total": 5.0}})

    samples, failed = run("https://x", None, [{"question": "ok"}, {"question": "bad"}],
                          transport=httpx.MockTransport(handler))
    assert len(samples) == 1 and failed == 1
