# Mortgage RAG — Sub-project 1: Package, Config, and Provider Seam

**Date:** 2026-09-27
**Status:** Design approved, pending spec review
**Scope:** Sub-project 1 of 4

---

## Purpose

Turn `complete_mortgage_rag_pipeline.py` — a Colab notebook export in which every
cell depends on globals defined above it — into an installable Python package that
runs outside Colab, imports without downloading models, and exposes a single
configurable entry point.

This is a portfolio project. Success means a hiring manager can clone the repo,
install it, run the tests on a laptop with no GPU and no API key, and read code
that looks like it was written for other people to read. It is not a product with
users or uptime obligations.

The work is the first of four sub-projects. This one exists so the three that
follow are straightforward rather than awkward — in particular, the evaluation
ablation in sub-project 2 needs a configurable pipeline, and that seam is cheaper
to build now than to retrofit.

### Context: the four sub-projects

| # | Scope | Status |
|---|---|---|
| 1 | Package, config, provider seam, bug fixes, unit tests, test CI | **this spec** |
| 2 | Ground-truth dataset, retrieval/generation metrics, ablation, RAGAS | later |
| 3 | FastAPI service, index persistence, Gradio rewrite, Docker, deploy | later |
| 4 | README, architecture diagram, demo recording, results tables | later |

Each gets its own spec, plan, and implementation cycle.

---

## Decisions made during design

| Decision | Choice | Rationale |
|---|---|---|
| Primary LLM | Anthropic API, `claude-sonnet-5` | Deploys on cheap CPU hosting; no GPU host needed. ~$0.011/query. |
| Judge LLM (sub-project 2) | `claude-opus-5` | A judge should be at least as capable as what it grades. |
| Local LLM | Optional alternate backend, Qwen3 8B GGUF | Replaces Mistral-7B-Instruct-v0.2 (Dec 2023). Keeps an offline story. |
| Ingestion | Docling | Replaces PyMuPDF + PaddleOCR + Tesseract + OpenCV + pdf2image. Deletes ~60 lines and the fragile `paddlepaddle` install, and adds table-structure extraction. |
| Embeddings | Keep `BAAI/bge-small-en-v1.5` | Make the upgrade an ablation arm in sub-project 2 rather than guessing. |
| Reranker | `BAAI/bge-reranker-v2-m3` | 8192-token window vs. 512. Apache 2.0. |
| Framework | Keep LlamaIndex | User's call. `VectorStoreIndex`, `Document`, `HuggingFaceEmbedding` port unchanged. |
| Structure | Flat modules in one package | Directory-per-concern is ceremony at ~1000 lines. |
| Perf testing | Structural assertions, not wall-clock | CI runners vary 2–3×; a flaky gate gets disabled. |

### Approaches considered and rejected

- **Per-stage `Protocol` abstractions.** One implementation per stage today. An
  interface written before its second implementation costs indirection on every
  read and buys nothing. Promote later if the ablation surfaces real competitors.
- **An `LLMProvider` ABC.** LlamaIndex's `LLM` base class already is that
  interface; every call site does `.complete(prompt)`. A wrapper around it would
  be an interface wrapping an interface.
- **Minimal lift-and-shift** (two modules, delete Colab lines). Cheapest, but
  leaves no config seam, so sub-project 2's ablation gets hacked with globals.
- **A vector database** (LanceDB/Chroma). Solves a scale problem that does not
  exist at 3–5 documents.

---

## Architecture

```
pyproject.toml
.gitignore                    # *.gguf, indexes/, data/documents/, .env
src/mortgage_rag/
  config.py                   # Settings dataclass + from_env()
  models.py                   # lazy, cached model/LLM getters
  ingest.py                   # Docling → page dicts
  chunking.py                 # recursive split, semantic merge, table detection
  classify.py                 # heuristic + LLM page classification
  index.py                    # TextNode building, VectorStoreIndex, BM25Index, Store
  retrieval.py                # vector, BM25, RRF, rewrite, decompose
  rerank.py                   # cross-encoder rerank, context cleanup
  pipeline.py                 # ask() — the single entry point
tests/
  fixtures/cfpb_closing_disclosure.pdf
notebooks/demo.ipynb
.github/workflows/test.yml
```

`eval/` and `app/` are deliberately absent; they belong to sub-projects 2 and 3.

### Data flow

```
PDF → ingest.load_pdf() → [page dicts]
    → classify.classify_pages() → [page dicts + doc_type]
    → chunking.chunk_page() → [TextNode + metadata]
    → index.build_store() → Store{vector_index, bm25_index, nodes}

query + Store + Settings → pipeline.ask()
    → retrieval.rewrite_query()        (if cfg.use_rewrite)
    → retrieval.decompose_query()      (if cfg.use_decomposition)
    → retrieval.retrieve()             (vector | bm25 | hybrid, per sub-query)
    → dedup across sub-queries
    → rerank.rerank()                  (if cfg.use_rerank)
    → rerank.clean_context()
    → LLM answer
    → {answer, sources, num_chunks_used, confidence, debug}
```

### Key structural changes from the notebook

**`Store` replaces five mutable globals.** The notebook's Gradio handler reassigns
`CURRENT_INDEX`, `CURRENT_PAGES`, `bm25_index`, `llamaindex_bm25`, and
`USE_LLAMAINDEX_BM25`. These become one object holding the vector index, the BM25
index, and the node list together, passed explicitly. Sub-project 3 persists and
reloads this object.

**No `Settings` global.** The notebook sets `Settings.llm` and
`Settings.embed_model` process-wide and mutates `Settings.text_splitter` twice to
control re-splitting. Instead, `embed_model=` is passed explicitly to
`VectorStoreIndex`, and the index is built from pre-chunked `TextNode`s rather than
`Document`s, so LlamaIndex never re-splits and the `chunk_size=2048` workaround
disappears. This also unblocks the ablation: a process-wide embedder makes it
impossible to run two embedding arms in one process.

**bge-small is loaded once, not twice.** The notebook loads the same weights as
both `HuggingFaceEmbedding` and `SentenceTransformer`. One cached getter serves
indexing, `semantic_merge`, and dedup; the functions that need an encoder take it
as an argument.

**Single BM25 implementation.** The notebook builds a manual `BM25Okapi` index and
then tries LlamaIndex's `BM25Retriever`, keeping both behind a
`USE_LLAMAINDEX_BM25` flag. Two code paths mean the ablation measures whichever one
happened to load. Keep only the manual `BM25Okapi` implementation — it is already
written, has no import fragility, and its tokenizer (currency- and percent-aware)
is tuned for this corpus.

---

## Components

### `config.py`

One frozen dataclass, `Settings`. Frozen so it is hashable and can key `lru_cache`.
`from_env()` walks `dataclasses.fields()` and casts by declared type — about ten
lines of stdlib rather than one hand-written `os.environ.get` call per field, or
a new settings dependency. Prefix `MORTGAGE_RAG_`. Boolean parsing is explicit
(`{"1","true","yes"}`), because `bool("false")` is `True` in Python and that failure
mode silently disables `use_rerank`.

No config file format. Env vars plus defaults cover local runs and containers; a
file can be added when something needs one.

| Group | Field | Default |
|---|---|---|
| Ingestion | `min_text_length` | `50` |  <!-- empty-page filter -->
| | `do_ocr` | `True` |
| | `do_table_structure` | `True` |
| Chunking | `chunk_max_tokens` | `512` |
| | `chunk_min_tokens` | `50` |
| | `merge_threshold` | `0.75` |
| | `merge_max_tokens` | `600` |
| | `min_chunk_words` | `10` |
| Classification | `use_llm_classification` | `True` |
| | `classify_snippet_chars` | `500` |
| Retrieval | `retrieval_mode` | `"hybrid"` |
| | `top_k` | `10` |
| | `rrf_k` | `60` |
| | `use_rewrite` | `True` |
| | `use_decomposition` | `True` |
| | `max_sub_queries` | `4` |
| Rerank | `use_rerank` | `True` |
| | `rerank_top_n` | `5` |
| | `min_rerank_score` | `None` (see below) |
| | `dedup_threshold` | `0.92` |
| | `max_context_tokens` | `2000` |
| Models | `llm_provider` | `"anthropic"` |
| | `llm_model` | `"claude-sonnet-5"` |
| | `temperature` | `0.1` |
| | `max_new_tokens` | `1024` |
| | `context_window` | `4096` (llama-cpp only) |
| | `embed_model_name` | `"BAAI/bge-small-en-v1.5"` |
| | `cross_encoder_name` | `"BAAI/bge-reranker-v2-m3"` |
| | `cross_encoder_max_length` | `1024` |
| | `gguf_path` | `""` |

### `models.py`

Three getters, each `@lru_cache(maxsize=1)` over the frozen `Settings`:

- `get_llm(cfg)` — branches on `cfg.llm_provider`, returns `Anthropic(...)` or
  `LlamaCPP(...)`. Both are LlamaIndex `LLM` instances; call sites are unchanged.
  This function is the entire "swappable LLM" feature.
- `get_encoder(cfg)` — the shared `SentenceTransformer`.
- `get_cross_encoder(cfg)` — the reranker.

**All heavy imports (`torch`, `sentence_transformers`,
`llama_index.embeddings.huggingface`, `docling`) happen inside function bodies, not
at module scope.** This is a hard requirement, enforced by a test, and it is what
makes cold start and CI viable.

### `ingest.py`

`load_pdf(path, cfg) -> list[PageData]` using Docling's `DocumentConverter` with
`do_ocr` and `do_table_structure` from config. Pages whose extracted text is shorter than `min_text_length` are dropped as
blank — the only remaining use of that field, since Docling now decides
internally when a page needs OCR. Returns the same page dict shape as
the notebook (`filename`, `page_number`, `text`, `source_type`, `doc_type`) so
downstream code is unchanged. Tables are emitted as Markdown within page text.

Deleted: `preprocess_image`, `ocr_with_paddle`, `ocr_with_tesseract`, and the
three-tier fallback.

### `chunking.py`, `classify.py`, `retrieval.py`, `rerank.py`

Near-1:1 ports of the notebook functions, with two changes: constants come from
`cfg`, and functions needing an encoder or LLM take it as an argument rather than
reading a global. Two bug fixes are described below.

### `pipeline.py`

`ask(query, store, cfg) -> dict` — one entry point, replacing both `ask()` and
`ask_v3()`. Return shape matches `ask_v3`.

---

## The ablation seam

`ask()` reads four fields, which sub-project 2 varies with
`dataclasses.replace()`:

| Arm | `retrieval_mode` | `use_rewrite` | `use_decomposition` | `use_rerank` |
|---|---|---|---|---|
| baseline | `vector` | `False` | `False` | `False` |
| +hybrid | `hybrid` | `False` | `False` | `False` |
| +rerank | `hybrid` | `False` | `False` | `True` |
| full | `hybrid` | `True` | `True` | `True` |

Each arm is a printable, reproducible config that goes into the results table.

---

## Bug fixes

### 1. RRF collapses distinct chunks

`reciprocal_rank_fusion` keys documents on `doc.text[:200]`. Two chunks sharing a
200-character prefix — common in mortgage boilerplate — merge into one, silently
reducing recall.

**Fix:** key on `(filename, page_number, chunk_id)`, already present in metadata.
The same prefix-keyed dedup appears in `ask_v3`'s cross-sub-query dedup and gets
the same fix.

### 2. The reranker sees ~25% of each chunk

`rerank_results` builds pairs with `doc.text[:512]` — 512 **characters**
(~128 tokens) — while the CrossEncoder is constructed with `max_length=512`
**tokens** and chunks run to 600 tokens. Roughly three quarters of each chunk is
discarded before scoring.

**Fix:** pass full chunk text and let the tokenizer truncate at
`cfg.cross_encoder_max_length`. `cross_encoder_max_length` defaults to `1024` rather than the notebook's `512`:
chunks run to 600 tokens, so 1024 comfortably fits query plus full chunk, and
bge-reranker-v2-m3 accepts up to 8192 if that ever needs raising. This may matter more than any model swap.

---

## Behavior changes from the notebook

Defaults otherwise reproduce the notebook exactly, so movement measured in
sub-project 2 is attributable. These are the deliberate exceptions:

1. **LLM** — `claude-sonnet-5` via API instead of local Mistral-7B GGUF.
2. **Ingestion** — Docling instead of the PyMuPDF/Paddle/Tesseract chain. Extracted
   text will differ, especially on tables.
3. **Reranker** — `bge-reranker-v2-m3` instead of `ms-marco-MiniLM-L-6-v2`.
4. **`min_rerank_score` defaults to `None` (no filtering).** The notebook's `-5.0`
   was calibrated against ms-marco-MiniLM's logit scale. bge-reranker-v2-m3 scores
   on a different scale, so carrying the constant over would drop chunks for
   arbitrary reasons. Disabling the filter is the honest default until sub-project 2
   has data to calibrate against. `rerank_top_n` still bounds the candidate count.
5. **`max_new_tokens` raised 512 → 1024.** 512 is tight for an answer carrying
   citations; truncation mid-citation is a failure mode worth avoiding.
6. **Both bug fixes above** change retrieval results by design.

`max_context_tokens` stays at `2000` for continuity, but it was chosen for a
4096-token local context window. Sonnet 5 has far more room, so this is a strong
ablation candidate for sub-project 2 — noted here, not changed now.

---

## Testing

Hard constraint: nothing in the default test run may download a model, load
`torch`, or make a network call.

### Unit tests

| File | Covers |
|---|---|
| `test_rrf.py` | Regression for bug 1: two chunks sharing a 200-char prefix but differing `chunk_id` survive fusion as two entries. Plus rank ordering and `k` damping. |
| `test_context_cleanup.py` | Score filter, the keep-at-least-one fallback, token budget truncation, dedup. Uses a stub encoder returning fixed vectors. |
| `test_chunking.py` | `recursive_chunk`: header splitting, oversized-section paragraph fallback, the min-word filter, no chunk exceeds `max_tokens`. |
| `test_table_detection.py` | `detect_table_content` true on a currency table, false on prose. A four-indicator threshold with no test is a threshold nobody will tune. |
| `test_config.py` | Defaults match the notebook constants; `from_env` casts int/float/bool correctly, including the `"false"` case. |

### Tier 1 performance tests (deterministic)

| File | Covers |
|---|---|
| `test_import_purity.py` | Subprocess-imports `mortgage_rag.pipeline` and asserts `torch`, `docling`, and `sentence_transformers` are absent from `sys.modules`. Guards the lazy-loading invariant that governs cold start. |
| `test_call_budget.py` | With a fake LLM counting `.complete()` calls: full config makes exactly 3 (rewrite, decompose, answer); baseline makes exactly 1. Analogous budgets for encoder and cross-encoder invocations. |

Call-count budgets are the highest-value latency guard available, because query
latency is dominated by LLM round trips. They make the current worst case visible:
with decomposition on, retrieval runs once per sub-query (up to 4×), so a full query
is 3 LLM calls, up to 8 retrievals, and a rerank over up to 40 candidates.

Wall-clock benchmarks are deliberately excluded. CI runners vary 2–3× between runs;
a threshold tight enough to catch a 30% regression fires on noise, and a perf gate
that cries wolf gets disabled. Real latency is measured on the deploy host in
sub-project 3 via per-stage logging, and published as a p50/p95 table in the README.

### Integration smoke test

One `@pytest.mark.integration` test — ingest the fixture PDF, run `ask()`, assert a
grounded answer with a page citation. Requires Docling models and an API key, so it
is opt-in (`pytest -m integration`) and excluded from CI.

**Fixture:** CFPB form H-25(B), "Closing Disclosure – Fixed Rate Loan Sample"
(`files.consumerfinance.gov/f/201403_cfpb_closing-disclosure_cover-H25B.pdf`,
94KB, 6 pages). A US government work, therefore public domain — no licensing
question. It is a *completed* sample, so it carries real figures the smoke test
asserts on ($162,000 loan amount, 3.875% rate, $3,240 prepayment penalty); the
blank H-25 model forms would make the test assert nothing. It also seeds
sub-project 2's corpus.

It has a digital text layer, so it exercises Docling's layout and table-structure
handling but not its OCR path. OCR stays uncovered until sub-project 2 introduces
scanned documents — a known gap, recorded here rather than papered over.

### CI

One GitHub Actions workflow: checkout, Python, `pip install -e .[dev]`,
`pytest -m "not integration"`. Pulled forward from the original Phase C because it
is ~15 lines and prevents the repo from silently rotting. The eval job lands in
sub-project 2, where there is something to evaluate.

---

## Dependencies

**Base:** `llama-index-core`, `llama-index-llms-anthropic`,
`llama-index-embeddings-huggingface`, `docling`, `rank-bm25`, `nltk`,
`scikit-learn`, `sentence-transformers`.

**Extras:** `[local-llm]` → `llama-cpp-python`. `[dev]` → `pytest`.

All versions pinned. The notebook pins nothing, and `llama-index` reorganizes
namespaces often enough that an unpinned rebuild months later is a coin flip.

---

## Verification

There is no clean before/after for this port. The notebook cannot run outside
Colab, so no baseline exists, and the Docling switch changes extracted text by
design. What is verifiable:

- Config defaults provably equal the notebook's constants.
- The two bug fixes have tests proving the previous behavior was wrong.
- The smoke test produces a grounded, cited answer on a known document.

The claim for this sub-project is "it runs outside Colab and the logic is intact,"
not "it is as good as before." Quality measurement begins in sub-project 2.

## Acceptance criteria

1. `pip install -e .[dev]` and `pytest` pass on a clean machine with no GPU, no API
   key, and no model downloads.
2. `python -c "import mortgage_rag.pipeline"` returns without loading `torch`,
   `docling`, or `sentence_transformers`.
3. `ask()` accepts a `Settings` whose four ablation switches measurably change
   which stages run (proven by the call-budget test).
4. The smoke test answers a question about the fixture PDF with a page citation.
5. CI is green on push.
6. `notebooks/demo.ipynb` runs top to bottom and contains no pipeline logic.

## Out of scope

FastAPI service; index persistence; the Gradio rewrite (it stays in the notebook
export, unported, because sub-project 3 rewrites it against the service and porting
it now is work done twice); Dockerfile; the ground-truth dataset; RAGAS; the
README; deployment.

## Known risks

- **Docling output quality is unvalidated on these documents.** It is the single
  largest behavioral change and its extraction differs from the current chain. The
  smoke test confirms it produces usable text; whether it produces *better* text is
  a sub-project 2 question. Mitigation: ingestion is one module behind a stable
  return shape, so reverting is contained.
- **`min_rerank_score` is disabled**, so low-relevance chunks reach the context
  budget. Bounded by `rerank_top_n` and `max_context_tokens`. Calibrated in
  sub-project 2.
- **Docling adds model downloads on first ingest**, affecting real cold start (not
  CI, which never reaches it). Quantified in sub-project 3.
