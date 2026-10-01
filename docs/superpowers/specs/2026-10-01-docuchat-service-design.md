# Docuchat — Sub-project 3: Service, UI, Docker, and Deploy

**Date:** 2026-10-01
**Status:** Design approved, pending spec review
**Scope:** Sub-project 3 of 4

---

## Purpose

Put docuchat on the web behind a link only portfolio visitors have. A hiring
manager clicks the link, lands in a chat UI already loaded with sample mortgage
documents, asks questions, and can optionally upload their own PDFs. The same
image runs locally with `docker run`, with no key required.

This is still a portfolio project, not a product. The service exists to be
demonstrated and measured, not to carry real users. Every hardening measure
below is sized to "a handful of invited visitors and a bounded API bill."

### Success criteria

1. `docker run` of the built image serves the UI on port 7860 with the sample
   corpus loaded, with no API key and no network access to model hubs.
2. The deployed Space rejects visitors without the portfolio link's key.
3. No visitor sees another visitor's uploads.
4. Daily API spend has a hard ceiling set by environment variables.
5. A p50/p95 latency table per pipeline stage, measured against the deploy
   host, is committed to `eval/results/latency.md`.
6. The default test suite still runs with no GPU, no API key, no model
   downloads, and no network.

Criteria 2, 4 (in production) and 5 require API keys and are deferred to the
blocked final steps (see "Blocked until API keys").

---

## Decisions made during design

| Decision | Choice | Rationale |
|---|---|---|
| Scope | Service, UI, Docker, latency, plus session isolation, upload caps, spend cap, deploy | A live link means concurrent visitors and a real API bill. |
| Access control | Secret token in the portfolio link, exchanged for a cookie | No friction for visitors; rotate by changing one env var. |
| Abuse backstop | Daily query and upload caps, not per-IP rate limiting | A leaked link is the realistic threat; a spend cap bounds it with less code than rate limiting. |
| Host | Hugging Face Spaces, Docker SDK, free CPU tier | 2 vCPU / 16 GB for $0; the ~4–6 GB model footprint fits. Upgrading to never-sleep is a setting, not a code change. |
| Default corpus | `eval/nodes/mortgage.jsonl`, embedded at startup | Visitors chat immediately; no Docling wait. |
| Served pipeline | The `+rerank` arm | One LLM call per question; the tuned arm (71.5% recall). |
| Process layout | One FastAPI process, Gradio mounted at `/`, both calling `service.py` | HF Spaces exposes one container and one port. |
| Index persistence | **Dropped** | Embedding 147 snapshot nodes takes ~5 s inside a 60–120 s cold start, uploads are session-scoped, and Space disk is ephemeral. Nothing worth persisting. |

### Approaches considered and rejected

- **Gradio as an HTTP client of the API** (httpx to localhost). Proves the REST
  contract is the only interface, but adds a serialization hop, internal
  cookie handling, and a second set of error paths. The shared `service.py`
  already guarantees Gradio holds no pipeline logic.
- **Separate UI and API containers.** HF Spaces free tier hosts one container.
- **Password gate.** Same secrecy as the link token, more friction.
- **Referer check.** Browsers and extensions strip Referer (locks out real
  visitors), and curl forges it.
- **Private HF Space.** Requires visitors to hold an invited HF account.
- **Paid always-on host** (Fly.io/Railway/Render, ~$15–30/month). Revisit if
  cold starts hurt.
- **`full` arm or a "deep mode" toggle.** 3 LLM calls and up to 4× retrieval
  work, with judged quality still unmeasured.

---

## Architecture

```
src/docuchat/
  service.py   Service: sessions + TTL, daily quota, upload caps, ingest_upload(), answer()
  api.py       create_app(service): access-token middleware, /api/* routes, mounts the UI at /
  ui.py        build(service) -> gr.Blocks; handlers call the service only
  bench.py     python -m docuchat.bench: p50/p95 per stage over eval/questions.yaml
  pipeline.py  (changed) ask() adds debug["timings"]
  config.py    (changed) adds service fields
Dockerfile
deploy/space-README.md        HF Space README (YAML frontmatter)
.github/workflows/deploy.yml  workflow_dispatch upload to the Space
```

### Data flow

```
startup (FastAPI lifespan):
    load_snapshot(eval/nodes/mortgage.jsonl) -> store_from_nodes() -> shared sample Store

question:
    UI handler | POST /api/ask -> Service.answer(session_id, query)
        -> quota check -> session's Store (private, or the sample)
        -> ask(query, store, +rerank cfg) -> {answer, sources, timings}

upload:
    UI handler | POST /api/sessions/{id}/documents -> Service.ingest_upload(session_id, files)
        -> type/size/page caps (before Docling) -> quota check
        -> ingest -> classify -> build_store() -> session's private Store
```

`create_app(service)` is the test seam. The module-level `app` that uvicorn
imports is `create_app(Service(Settings.from_env()))`, and the sample store is
built in the lifespan hook, never at import, so `test_import_purity.py` keeps
passing.

### Concurrency

One uvicorn worker. Docling ingestion is the CPU hog, so a semaphore admits one
ingestion at a time; other sessions keep chatting meanwhile. The existing
`models._LOCAL_MODEL_LOCK` continues to serialize local-model inference.

---

## Components

### `config.py` additions

| Field | Default |
|---|---|
| `session_ttl_minutes` | `30` |
| `max_upload_mb` | `10` |
| `max_upload_pages` | `30` |
| `daily_query_cap` | `200` |
| `daily_upload_cap` | `20` |

The served arm is built with `evaluate.arm_settings("+rerank", base)`, so the
demo and the eval table cannot drift apart.

The access key is **not** a `Settings` field: `Settings` is printed into results
tables, and the key is a secret. `api.py` reads `DOCUCHAT_ACCESS_KEY` directly
from the environment, the way the Anthropic SDK reads its key.

### `service.py`

`Service(cfg, clock=time.time)` with an injectable clock for TTL and quota
tests.

- **Sessions.** A dict keyed by an opaque string ID. A new session uses the
  shared sample store. An upload gives it a private `Store` built from its
  uploads only, and `reset(session_id)` returns it to the sample. The UI uses
  Gradio's `session_hash` as the ID; REST clients get a UUID from
  `POST /api/sessions`.
- **Unknown IDs.** Any session ID not yet seen starts a new session on the
  sample. There is no 404 path. Gradio's `session_hash` and the server-minted
  UUIDs are both random, so one visitor cannot guess another's ID.
- **TTL.** When a session sits idle longer than `session_ttl_minutes`, its
  private store is dropped and the session is marked expired. Sweeping happens
  lazily on every service call, with no background thread. The next answer
  carries `expired: true` once, so the UI can say "session expired, back on the
  sample documents."
  `# ponytail: session records (minus their stores) are never removed; a few bytes per visitor is fine at portfolio scale.`
- **Quota.** In-memory query and upload counters, reset at UTC midnight. Over
  the cap, the service raises `QuotaExceeded`.
  `# ponytail: counters reset on restart; persist them if a restart ever becomes an abuse vector.`
- **Upload caps, checked before Docling runs.** All rejections raise one
  `UploadRejected(message, status=...)` (a `ServiceError` subclass). No files: 422. A file not starting with
  `%PDF`, or one pypdfium2 cannot open: 415. Total size over `max_upload_mb`,
  or total pages over `max_upload_pages`: 413. The page count uses pypdfium2,
  already a dependency.
- **Failure isolation.** If ingestion fails, the session keeps its previous
  store and the error message reaches the caller.

### `api.py`

**Routes:** `POST /api/sessions`, `POST /api/sessions/{id}/documents`,
`POST /api/sessions/{id}/reset`, `POST /api/ask`, `GET /api/health`.
`/api/ask` takes `{session_id, query}` and returns
`{answer, sources, timings, expired}`. `/api/health` returns
`{status, gated}`.

**Access gate (middleware on the whole app, UI included):**

1. A valid `docuchat_key` cookie: allow.
2. `?key=` matching the key: set the cookie (HttpOnly, Secure, SameSite=Lax),
   then 303-redirect to the same URL without `key`, so the key never sits in
   browser history.
3. Anything else: 403 with a short "access by invitation" page.

Comparison uses `hmac.compare_digest`. `/api/health` is exempt, because HF
probes it. **If `DOCUCHAT_ACCESS_KEY` is unset, the gate is off**, so local
`docker run` works without a key. Startup logs a warning, and `/api/health`
reports `gated: false`, so a Space missing its secret is visible. The quota
still bounds spend in that case.

**Error mapping:**

| Condition | Status | UI message |
|---|---|---|
| Malformed request | 422 | — |
| `QuotaExceeded` | 429 | "Demo quota reached, back tomorrow." |
| `UploadRejected` | its `status` (413 / 415 / 422) | its message, e.g. the cap that was exceeded |
| No LLM key configured (`get_llm()` raises) | 503 | "LLM not configured." |
| LLM API call fails | 502 | "Model service error, try again." |

Classification runs once per uploaded file, heuristic first, and calls the
LLM only when the heuristic returns "Unknown". The service hands the pipeline a
lazy LLM wrapper that resolves `get_llm()` on first use, so without a key an
upload succeeds when the heuristic classifies every file, and returns 503 only
when the LLM is actually needed.

Each answered query logs one JSON line to stdout (session-free: timings,
number of chunks, status), which lands in the HF Space logs.

### `ui.py`

`build(service) -> gr.Blocks`: an upload panel with "Process & Index" and
"Back to sample documents" buttons, a status line, and a chat panel that shows
sources (filename, page, preview) under each answer. Every handler calls one
`Service` method and maps its exceptions to the messages above. The UI contains
no pipeline logic. Layout follows the notebook's Gradio app, rebranded from
"Mortgage RAG Analyst" to docuchat.

### `pipeline.py`: per-stage timings

`ask()` wraps each stage in `time.perf_counter()` and writes `debug["timings"]`
in milliseconds. The keys are `rewrite`, `decompose`, `retrieve`, `rerank`,
`clean_context` and `llm`, plus `total`. A disabled stage's key is absent. The
empty-retrieval early return carries the timings measured so far.

### `bench.py`

`python -m docuchat.bench --url URL [--key KEY] [--out eval/results/latency.md]`.
An HTTP client, because free Spaces have no shell. It runs every question in
`eval/questions.yaml` sequentially in a single session, recording client-side
end-to-end time (network included) and the server's per-stage `timings`. The
first request is reported separately as "first request (cold)" and excluded
from the percentiles. The output is a markdown table of p50/p95 per stage plus
total, headed with the URL, date, git commit, and served arm. Sub-project 4
lifts it into the README.

### Dockerfile

- `python:3.11-slim`.
- **CPU-only torch** from `https://download.pytorch.org/whl/cpu`, installed
  before `pip install .`, which saves ~2 GB of CUDA libraries.
- **Models downloaded at build time:** bge-small, bge-reranker-v2-m3, and
  Docling's layout models. A wake-up loads from disk, not the network.
- Runs as uid 1000 (an HF requirement), on port 7860:
  `CMD ["uvicorn", "docuchat.api:app", "--host", "0.0.0.0", "--port", "7860"]`.
- Expected image size: ~4–5 GB.

### Deploy

- The Space is created by hand once (Docker SDK), with secrets for the LLM
  provider key, `DOCUCHAT_ACCESS_KEY`, and `DOCUCHAT_LLM_PROVIDER`.
- `.github/workflows/deploy.yml` (`workflow_dispatch` only) uploads the repo
  with `huggingface-cli upload`, substituting `deploy/space-README.md` as the
  Space's README so HF's YAML frontmatter stays out of the portfolio README. Its
  only secret is `HF_TOKEN`.
- Cold start: a free Space sleeps after ~48 h idle, and the first visit then
  waits while it wakes and loads models. Accepted for $0; the fix is HF's paid
  CPU upgrade with no code change.

---

## Testing

Hard constraint, unchanged: nothing in the default suite downloads a model,
loads torch, or touches the network. Tests reuse `tests/conftest.py`'s
`fake_store`, `fake_llm`, `fake_models` and `make_node`.

| File | Covers |
|---|---|
| `test_service.py` | An unknown ID starts on the sample; upload switches to a private store; `reset` switches back. TTL drops the private store and reports `expired` exactly once (injected clock). Query cap: request 201 raises `QuotaExceeded`; counters reset at UTC midnight. Failed ingestion keeps the previous store. |
| `test_upload_caps.py` | A non-PDF is rejected; over-MB and over-page uploads are rejected **before** ingestion, proven by an ingest stub that must never be called. The page count runs on the CFPB fixture. |
| `test_api.py` | `TestClient` on `create_app(fake_service)`. Gate: no key → 403; `?key=` → cookie + 303 without `key`; valid cookie → 200; wrong key → 403; health exempt; unset key → open, `gated: false`. Error mapping: 429, 413, 415, 503, 502. |
| `test_pipeline.py` (extended) | `debug["timings"]` holds exactly the stages that ran, for `+rerank` and `full`, including on the empty-retrieval return. |
| `test_bench.py` | The percentile function (including n=1) and exclusion of the cold first request. |
| `test_ui.py` | `build(fake_service)` returns a `gr.Blocks` without raising. |
| `test_import_purity.py` | Add `docuchat.service`, `docuchat.api`, `docuchat.ui` and `docuchat.bench` to its module list. |

Not tested in CI: the Gradio layout beyond construction, the Docker build (too
large for a runner; HF builds it on deploy), and the live benchmark.

---

## Dependencies

Base, pinned exactly: `fastapi`, `uvicorn`, `gradio`, `python-multipart`.
Dev: `httpx` (for `TestClient`). `bench.py` uses `httpx`, which Gradio already
depends on. `gradio` and `fastapi` may be imported at module scope: neither
pulls in torch, docling or sentence-transformers, and the import-purity test
enforces that.

---

## Blocked until API keys

Everything above is built and tested without keys. Without a key, the app runs
and serves the UI and gate, while `/api/ask` (and uploads that need the LLM) return 503 "LLM not
configured" whenever the LLM is needed. These steps wait for keys, the same way sub-project 2's Task 10 is
blocked:

1. Create the Space and set its secrets.
2. First deploy via `deploy.yml`, and verify the gate and quota live.
3. Run `bench.py` against the Space and commit `eval/results/latency.md`.
4. Confirm the daily caps against real spend. Classification costs at most
   one LLM call per uploaded file, and only for files the heuristic can't
   label, so queries dominate the bill.

---

## Out of scope

- Index persistence (dropped; see Decisions).
- Per-IP rate limiting.
- Persisting quota counters or sessions across restarts.
- The `full` arm in the demo.
- A Docker build in CI.
- README, architecture diagram, demo recording (sub-project 4).

## Risks

- **Cold start on the free tier.** Most portfolio clicks may land on a sleeping
  Space. Mitigation: a visible "warming up" state; the paid upgrade if it hurts.
- **Image size.** At ~4–5 GB, HF builds are slow. CPU-only torch is the main
  lever.
- **Docling on 2 vCPU.** Ingesting 30 pages may take a minute or more. The
  page cap and the one-at-a-time semaphore bound it; the UI shows progress.
- **A leaked link.** Anyone holding it can use the demo until the key is
  rotated. The daily caps bound the cost.
