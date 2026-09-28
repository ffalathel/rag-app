# Mortgage RAG Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `complete_mortgage_rag_pipeline.py` (a Colab notebook export driven by module-level globals) into an installable Python package with a config seam, a swappable LLM backend, Docling ingestion, and a test suite that runs with no GPU, no API key, and no model downloads.

**Architecture:** One package, `src/mortgage_rag/`, of flat single-responsibility modules. A frozen `Settings` dataclass carries every tuning constant and the four ablation switches. Models load lazily through `lru_cache`d getters whose heavy imports live inside the function bodies. A `Store` object replaces five mutable globals. `ask(query, store, cfg)` is the only entry point.

**Tech Stack:** Python 3.11+, LlamaIndex (core, `llama-index-llms-anthropic`, `llama-index-embeddings-huggingface`), Docling, sentence-transformers, rank-bm25, nltk, scikit-learn, pytest.

**Spec:** `docs/superpowers/specs/2026-09-27-mortgage-rag-packaging-design.md`

**Source material:** `complete_mortgage_rag_pipeline.py` at the repo root is the notebook export being ported. Tasks reference it by function name; read that function before porting it.

## Global Constraints

- Python 3.11+. All dependency versions pinned in `pyproject.toml`.
- Package lives at `src/mortgage_rag/`; `pyproject.toml` uses a src layout.
- **No module-scope import of `torch`, `docling`, `sentence_transformers`, or `llama_index.embeddings.huggingface` anywhere in the package.** These import only inside function bodies. Enforced by `tests/test_import_purity.py`.
- **No test in the default run may download a model, load `torch`, or make a network call.** Tests needing those are marked `@pytest.mark.integration` and excluded from CI.
- No `llama_index.core.Settings` global mutation anywhere. `embed_model` is passed explicitly.
- Every tuning constant comes from `Settings`. No magic numbers in function bodies.
- Functions needing an encoder, cross-encoder, or LLM take it as a parameter. No module-level model objects.
- Config defaults reproduce the notebook's constants except for the six deliberate changes listed in the spec's "Behavior changes" section.
- Register the `integration` marker in `pyproject.toml` under `[tool.pytest.ini_options]` so unmarked-marker warnings don't appear.

## Review Focus

These are the failure modes the spec implies but no obvious task test covers. Each has a test assigned to the task owning the code.

1. **A single chunk larger than `max_context_tokens`** — `clean_context`'s budget loop `break`s on the first chunk that doesn't fit, so an oversized first chunk yields an empty context and the LLM answers with nothing. A reasonable person expects at least one chunk through. → Task 8.
2. **Retrieval returns zero results** (empty corpus, or a query matching nothing above BM25's score floor) — `ask()` must return the documented no-results shape, not raise on `np.mean([])`. → Task 9.
3. **An invalid `retrieval_mode` or `llm_provider` from the environment** — a typo like `MORTGAGE_RAG_RETRIEVAL_MODE=vektor` must fail loudly at config construction, not silently fall through to one branch and quietly corrupt an ablation arm. → Task 1.
4. **Empty or whitespace-only page text reaching the chunker** — `recursive_chunk` on `""` must return `[]`, not a one-element list of nothing or an index error on `breaks[0]`. → Task 3.
5. **Missing `ANTHROPIC_API_KEY`** — must fail with a message naming the missing variable at `get_llm` time, not with a cryptic auth error surfacing mid-query after a user has already uploaded documents. → Task 2.

---

### Task 1: Project scaffolding and configuration

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/mortgage_rag/__init__.py`, `src/mortgage_rag/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Settings` — a frozen, hashable dataclass with the fields in the spec's config table, plus `Settings.from_env(**overrides) -> Settings`. Every later task takes a `Settings` named `cfg`.

`__init__.py` stays empty — no re-exports. Anything it imports would be imported by `import mortgage_rag`, which is exactly what Task 2's purity test forbids.

- [ ] **Step 1: Write the failing tests**

```python
def test_defaults_match_notebook_constants():
    cfg = Settings()
    assert cfg.chunk_max_tokens == 512
    assert cfg.merge_threshold == 0.75
    assert cfg.dedup_threshold == 0.92
    assert cfg.max_context_tokens == 2000
    assert cfg.rrf_k == 60
    assert cfg.top_k == 10
    assert cfg.rerank_top_n == 5
    assert cfg.min_rerank_score is None
    assert cfg.llm_model == "claude-sonnet-5"
    assert cfg.cross_encoder_name == "BAAI/bge-reranker-v2-m3"
    assert cfg.cross_encoder_max_length == 1024
    assert cfg.max_new_tokens == 1024

def test_from_env_casts_by_field_type(monkeypatch):
    monkeypatch.setenv("MORTGAGE_RAG_TOP_K", "25")
    monkeypatch.setenv("MORTGAGE_RAG_MERGE_THRESHOLD", "0.9")
    cfg = Settings.from_env()
    assert cfg.top_k == 25 and isinstance(cfg.top_k, int)
    assert cfg.merge_threshold == 0.9

@pytest.mark.parametrize("raw,expected", [
    ("false", False), ("False", False), ("0", False), ("no", False),
    ("true", True), ("True", True), ("1", True), ("yes", True),
])
def test_from_env_parses_bools_explicitly(monkeypatch, raw, expected):
    monkeypatch.setenv("MORTGAGE_RAG_USE_RERANK", raw)
    assert Settings.from_env().use_rerank is expected

def test_from_env_overrides_take_precedence(monkeypatch):
    monkeypatch.setenv("MORTGAGE_RAG_TOP_K", "25")
    assert Settings.from_env(top_k=3).top_k == 3

def test_settings_is_hashable_for_lru_cache():
    assert hash(Settings()) == hash(Settings())

# Review Focus 3
def test_invalid_retrieval_mode_raises():
    with pytest.raises(ValueError, match="retrieval_mode"):
        Settings(retrieval_mode="vektor")

def test_invalid_llm_provider_raises():
    with pytest.raises(ValueError, match="llm_provider"):
        Settings(llm_provider="openai")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mortgage_rag'`

- [ ] **Step 3: Write `pyproject.toml` and `.gitignore`**

`pyproject.toml`: src layout, `requires-python = ">=3.11"`, pinned base dependencies (`llama-index-core`, `llama-index-llms-anthropic`, `llama-index-embeddings-huggingface`, `docling`, `rank-bm25`, `nltk`, `scikit-learn`, `sentence-transformers`), extras `local-llm = ["llama-cpp-python"]` and `dev = ["pytest"]`, and `[tool.pytest.ini_options]` registering the `integration` marker with `testpaths = ["tests"]`.

`.gitignore`: `*.gguf`, `indexes/`, `data/documents/`, `.env`, plus standard Python entries.

- [ ] **Step 4: Implement `Settings` in `src/mortgage_rag/config.py`**

`@dataclass(frozen=True)` with the exact fields, types, and defaults from the spec's config table. Two methods:

- `__post_init__(self) -> None` — validates `retrieval_mode in {"vector","bm25","hybrid"}` and `llm_provider in {"anthropic","llamacpp"}`, raising `ValueError` naming the offending field.
- `from_env(cls, **overrides) -> "Settings"` — walk `dataclasses.fields(cls)`, read `MORTGAGE_RAG_<FIELD_NAME_UPPER>`, cast by `field.type`, and let `overrides` win over the environment. Booleans parse against the literal sets in the test; do not use `bool(str)`. `Optional[float]` fields (`min_rerank_score`) treat an empty string as `None`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS (all)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore src/mortgage_rag/__init__.py src/mortgage_rag/config.py tests/test_config.py
git commit -m "feat: add package scaffolding and Settings config"
```

---

### Task 2: Lazy model getters and import purity

**Files:**
- Create: `src/mortgage_rag/models.py`
- Test: `tests/test_models.py`, `tests/test_import_purity.py`

**Interfaces:**
- Consumes: `Settings` from Task 1.
- Produces:
  - `get_llm(cfg: Settings) -> LLM` (LlamaIndex `LLM`; `Anthropic` or `LlamaCPP`)
  - `get_embed_model(cfg: Settings) -> HuggingFaceEmbedding`
  - `get_encoder(cfg: Settings) -> SentenceTransformer` — **the SentenceTransformer underlying `get_embed_model(cfg)`, not a second load**
  - `get_cross_encoder(cfg: Settings) -> CrossEncoder`

All three are `@lru_cache(maxsize=1)`. Later tasks call these at the edges (Task 10) and take the returned objects as parameters everywhere else.

- [ ] **Step 1: Write the failing tests**

`tests/test_import_purity.py` — runs in a subprocess so another test cannot pre-import the forbidden modules:

```python
HEAVY = {"torch", "docling", "sentence_transformers"}

@pytest.mark.parametrize("module", [
    "mortgage_rag", "mortgage_rag.config", "mortgage_rag.models",
    "mortgage_rag.pipeline", "mortgage_rag.ingest",
])
def test_import_does_not_load_heavy_deps(module):
    code = (
        f"import {module}, sys; "
        f"loaded = {HEAVY!r} & set(sys.modules); "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
```

Parametrised modules that do not exist yet will fail; that is expected until the task that creates them lands. Mark the not-yet-existing ones with `pytest.importorskip` in this task and remove the skips in Task 9.

`tests/test_models.py`:

```python
def test_get_llm_is_cached(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    cfg = Settings()
    assert get_llm(cfg) is get_llm(cfg)

# Review Focus 5
def test_get_llm_without_api_key_raises_naming_the_variable(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_llm.cache_clear()
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        get_llm(Settings())

def test_get_llm_llamacpp_without_gguf_path_raises():
    get_llm.cache_clear()
    with pytest.raises(RuntimeError, match="gguf_path"):
        get_llm(Settings(llm_provider="llamacpp", gguf_path=""))

@pytest.mark.integration
def test_encoder_and_embed_model_share_one_loaded_model():
    cfg = Settings()
    assert get_encoder(cfg) is _sentence_transformer_of(get_embed_model(cfg))
```

The last test is `integration`-marked because it must actually load weights. It is
the guard on the spec's "bge-small is loaded once, not twice" requirement; without
it the two getters silently drift into two copies of the same 130MB model.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models.py tests/test_import_purity.py -v`
Expected: FAIL — `mortgage_rag.models` does not exist.

- [ ] **Step 3: Implement the three getters in `src/mortgage_rag/models.py`**

Each is `@lru_cache(maxsize=1)` over `cfg`. Every heavy import (`from llama_index.llms.anthropic import Anthropic`, `from llama_index.llms.llama_cpp import LlamaCPP`, `from sentence_transformers import SentenceTransformer, CrossEncoder`) goes **inside** the function body.

`get_llm` branches on `cfg.llm_provider`. Before constructing `Anthropic`, check `os.environ.get("ANTHROPIC_API_KEY")` and raise `RuntimeError` naming the variable if absent. Before constructing `LlamaCPP`, check `cfg.gguf_path` is non-empty and the file exists, raising `RuntimeError` naming `gguf_path` otherwise. Pass `model`, `temperature`, `max_tokens` from cfg to `Anthropic`; `model_path`, `temperature`, `max_new_tokens`, `context_window`, and `model_kwargs={"n_gpu_layers": -1}` to `LlamaCPP`.

`get_cross_encoder` passes `max_length=cfg.cross_encoder_max_length`.

`get_embed_model` constructs `HuggingFaceEmbedding(model_name=cfg.embed_model_name)`.
`get_encoder` must **not** construct its own `SentenceTransformer` — it returns the
one `HuggingFaceEmbedding` already holds, via a module-private helper
`_sentence_transformer_of(embed_model)`. Check the installed llama-index version for
the attribute holding it (`._model` at time of writing) and raise a clear
`RuntimeError` if it is absent rather than silently falling back to a second load.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models.py tests/test_import_purity.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mortgage_rag/models.py tests/test_models.py tests/test_import_purity.py
git commit -m "feat: add lazy cached model getters with import purity test"
```

---

### Task 3: Chunking

**Files:**
- Create: `src/mortgage_rag/chunking.py`
- Test: `tests/test_chunking.py`, `tests/test_table_detection.py`

**Source:** port `SECTION_PATTERNS`, `find_section_breaks`, `recursive_chunk`, `semantic_merge`, `smart_chunk_page`, `detect_table_content` from the notebook.

**Interfaces:**
- Consumes: `Settings`.
- Produces:
  - `find_section_breaks(text: str) -> list[tuple[int, str]]`
  - `recursive_chunk(text: str, cfg: Settings) -> list[tuple[str, str]]` — `(chunk_text, section_title)`
  - `semantic_merge(chunks: list[tuple[str, str]], encoder, cfg: Settings) -> list[tuple[str, str]]`
  - `chunk_page(text: str, encoder, cfg: Settings) -> list[tuple[str, str]]`
  - `detect_table_content(text: str) -> bool`

`encoder` is any object with `.encode(list[str], normalize_embeddings=True) -> np.ndarray`. Tests pass a stub; production passes `get_encoder(cfg)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_splits_on_numbered_section_headers():
    text = "1. PAYMENTS\nBorrower shall pay monthly.\n\n2. DEFAULT\nLender may accelerate."
    chunks = recursive_chunk(text, Settings())
    assert len(chunks) == 2
    assert "PAYMENTS" in chunks[0][1] and "DEFAULT" in chunks[1][1]

def test_no_chunk_exceeds_max_tokens():
    text = "\n\n".join("word " * 200 for _ in range(10))
    cfg = Settings(chunk_max_tokens=512)
    for chunk_text, _ in recursive_chunk(text, cfg):
        assert len(chunk_text.split()) * 1.3 <= cfg.chunk_max_tokens

def test_drops_chunks_below_min_chunk_words():
    chunks = recursive_chunk("1. A\nshort\n\n2. B\n" + "word " * 50, Settings())
    assert all(len(t.split()) >= Settings().min_chunk_words for t, _ in chunks)

def test_unheadered_text_becomes_one_full_page_chunk():
    chunks = recursive_chunk("word " * 50, Settings())
    assert len(chunks) == 1 and chunks[0][1] == "Full Page"

# Review Focus 4
@pytest.mark.parametrize("text", ["", "   ", "\n\n\t\n"])
def test_empty_or_whitespace_text_returns_no_chunks(text):
    assert recursive_chunk(text, Settings()) == []

def test_semantic_merge_combines_similar_adjacent_chunks(stub_encoder):
    # stub_encoder returns identical vectors → cosine similarity 1.0
    merged = semantic_merge([("a " * 20, "S1"), ("b " * 20, "S1")], stub_encoder, Settings())
    assert len(merged) == 1

def test_semantic_merge_respects_merge_max_tokens(stub_encoder):
    big = "word " * 400
    merged = semantic_merge([(big, "S1"), (big, "S1")], stub_encoder, Settings())
    assert len(merged) == 2
```

`tests/test_table_detection.py`:

```python
def test_detects_currency_table():
    text = "Loan Amount   $250,000.00   $1,234.56\nTaxes   $3,200.00   $266.67\nInsurance   $1,800.00   $150.00"
    assert detect_table_content(text) is True

def test_prose_is_not_a_table():
    text = "Borrower shall pay to Lender the principal sum of two hundred fifty thousand dollars."
    assert detect_table_content(text) is False
```

Define `stub_encoder` in `tests/conftest.py` as a fixture returning an object whose `.encode()` yields a fixed `np.ndarray` of identical unit vectors.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chunking.py tests/test_table_detection.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `src/mortgage_rag/chunking.py`**

Port the notebook functions with three changes: constants read from `cfg` (`chunk_max_tokens`, `chunk_min_tokens`, `min_chunk_words`, `merge_threshold`, `merge_max_tokens`); `semantic_merge` takes `encoder` as a parameter instead of using a global; `recursive_chunk` guards empty/whitespace input before touching `breaks[0]`. Keep the notebook's `words * 1.3` token estimate and the existing `SECTION_PATTERNS` regexes verbatim. `chunk_page` is `recursive_chunk` then `semantic_merge`.

Use `numpy` for the cosine similarity between normalised vectors (a dot product), not `sklearn.metrics.pairwise.cosine_similarity` on reshaped single rows.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chunking.py tests/test_table_detection.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mortgage_rag/chunking.py tests/test_chunking.py tests/test_table_detection.py tests/conftest.py
git commit -m "feat: add chunking with config-driven constants and injected encoder"
```

---

### Task 4: Page classification

**Files:**
- Create: `src/mortgage_rag/classify.py`
- Test: `tests/test_classify.py`

**Source:** port `classify_doc_type_heuristic`, `classify_page_with_llm`, `classify_all_pages`.

**Interfaces:**
- Consumes: `Settings`.
- Produces:
  - `classify_doc_type_heuristic(text: str) -> str`
  - `classify_page_with_llm(text: str, llm, cfg: Settings) -> str`
  - `classify_pages(pages: list[dict], llm, cfg: Settings) -> list[dict]` — sets `doc_type` on each page dict in place and returns the list.

`llm` is any object with `.complete(prompt) -> object` whose `str()` is the response. Tests pass a fake; Task 10 passes `get_llm(cfg)`.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("text,expected", [
    ("This CLOSING DISCLOSURE describes...", "Closing Disclosure"),
    ("PROMISSORY NOTE dated...", "Promissory Note"),
    ("Nothing recognisable here at all", "Unknown"),
])
def test_heuristic_classification(text, expected):
    assert classify_doc_type_heuristic(text) == expected

def test_llm_used_only_when_heuristic_returns_unknown(fake_llm):
    pages = [
        {"text": "CLOSING DISCLOSURE ...", "page_number": 1, "filename": "a.pdf"},
        {"text": "x" * 100, "page_number": 2, "filename": "a.pdf"},
    ]
    classify_pages(pages, fake_llm, Settings())
    assert fake_llm.calls == 1

def test_llm_classification_skipped_when_disabled(fake_llm):
    pages = [{"text": "x" * 100, "page_number": 1, "filename": "a.pdf"}]
    classify_pages(pages, fake_llm, Settings(use_llm_classification=False))
    assert fake_llm.calls == 0
    assert pages[0]["doc_type"] == "Unknown"

def test_overlong_llm_label_falls_back_to_other(fake_llm_returning):
    llm = fake_llm_returning("x" * 80)
    assert classify_page_with_llm("some page text here", llm, Settings()) == "Other"
```

Add `fake_llm` to `tests/conftest.py`: an object with a `calls` counter and a `.complete(prompt)` returning a fixed string. Add `fake_llm_returning(text)` as a factory fixture.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_classify.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `src/mortgage_rag/classify.py`**

Port verbatim, with `llm` and `cfg` as parameters. Keep the notebook's keyword list and its ordering — ordering is load-bearing, since `"disclosure"` must not shadow `"closing disclosure"`. Read the snippet length from `cfg.classify_snippet_chars`. Replace the notebook's bare `except:` clauses with `except Exception`. Drop the per-page `print` calls.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_classify.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mortgage_rag/classify.py tests/test_classify.py tests/conftest.py
git commit -m "feat: add page classification with injected LLM"
```

---

### Task 5: Docling ingestion

**Files:**
- Create: `src/mortgage_rag/ingest.py`, `tests/fixtures/` (with the sample PDF)
- Test: `tests/test_ingest.py`

**Replaces:** the notebook's `preprocess_image`, `ocr_with_paddle`, `ocr_with_tesseract`, `ingest_pdf`, `ingest_all_pdfs`. None of those port.

**Interfaces:**
- Consumes: `Settings`.
- Produces:
  - `load_pdf(path: str | Path, cfg: Settings) -> list[dict]`
  - `load_directory(folder: str | Path, cfg: Settings) -> list[dict]`

Each dict has exactly these keys: `filename: str`, `page_number: int` (1-indexed), `text: str`, `source_type: str` (`"docling"`), `doc_type: str` (`"unknown"` until Task 4 runs).

- [ ] **Step 1: Download the fixture PDF**

Fetch CFPB form **H-25(B), "Mortgage Loan Transaction Closing Disclosure – Fixed Rate Loan Sample"** — a completed (not blank) sample. A US government work, therefore public domain.

```bash
mkdir -p tests/fixtures
curl -sSL -o tests/fixtures/cfpb_closing_disclosure.pdf \
  https://files.consumerfinance.gov/f/201403_cfpb_closing-disclosure_cover-H25B.pdf
shasum -a 256 tests/fixtures/cfpb_closing_disclosure.pdf
```

Expected: `606a93c8baaca815439822df5cf8c78cbb2dcf6cc4af5aa291a459c7917e4173`, 94,194 bytes, 6 pages.

Verified facts in this document, used as assertions in Task 10:

| Field | Value |
|---|---|
| Loan Amount | `$162,000` |
| Interest Rate | `3.875%` |
| Closing Costs | `$9,712.10` |
| Cash to Close | `$14,147.26` |
| Prepayment Penalty | `YES` — as high as `$3,240` in the first 2 years |

If the checksum differs, CFPB has revised the form: keep the download, re-derive the table above from the new file, and update Task 10's assertions to match. Do not substitute a different form — H-25(A), (C), (D), (H) and the PACE form are **blank** models whose figures are zeros, and H-25(F1) is a two-page excerpt. Those make the smoke test assert nothing.

**Note on coverage:** this PDF has a digital text layer, so it exercises Docling's layout and table-structure handling but **not** its OCR path. OCR goes uncovered until sub-project 2 adds scanned documents. That is a known gap, not an oversight.

- [ ] **Step 2: Write the failing tests**

```python
def test_page_dict_shape_is_stable():
    # unit-level: no Docling, no network
    assert PAGE_KEYS == {"filename", "page_number", "text", "source_type", "doc_type"}

def test_blank_pages_below_min_text_length_are_dropped():
    pages = _pages_from_texts("a.pdf", ["", "   ", "x" * 100], Settings(min_text_length=50))
    assert [p["page_number"] for p in pages] == [3]

@pytest.mark.integration
def test_load_pdf_extracts_text_and_pages():
    pages = load_pdf("tests/fixtures/cfpb_closing_disclosure.pdf", Settings())
    assert len(pages) >= 1
    assert all(set(p) == PAGE_KEYS for p in pages)
    assert any("Closing Disclosure" in p["text"] for p in pages)
```

`_pages_from_texts(filename, texts, cfg)` is the module-internal helper that turns extracted per-page text into page dicts and applies the `min_text_length` filter. Testing it directly is what keeps the blank-page rule covered without Docling.

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_ingest.py -v -m "not integration"`
Expected: FAIL — module does not exist.

- [ ] **Step 4: Implement `src/mortgage_rag/ingest.py`**

`load_pdf` imports Docling **inside the function body**, builds a `DocumentConverter` configured from `cfg.do_ocr` and `cfg.do_table_structure`, converts the file, and emits one dict per page via `_pages_from_texts`. Export per-page text as Markdown so table structure survives. `load_directory` globs `*.pdf` case-insensitively, sorted, and concatenates.

Module-level constant `PAGE_KEYS: frozenset[str]` holds the five keys, so downstream tasks and tests assert against one definition.

Consult the installed Docling version's API for the page-level export call rather than guessing; if only whole-document export is available, split on Docling's page provenance metadata.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_ingest.py -v -m "not integration"` then `pytest tests/test_ingest.py -v -m integration`
Expected: PASS (both). The integration run downloads Docling models on first execution.

- [ ] **Step 6: Commit**

```bash
git add src/mortgage_rag/ingest.py tests/test_ingest.py tests/fixtures/
git commit -m "feat: replace OCR chain with Docling ingestion"
```

---

### Task 6: Nodes, BM25 index, and Store

**Files:**
- Create: `src/mortgage_rag/index.py`
- Test: `tests/test_index.py`

**Source:** port `create_documents_from_pages`, `build_phase2_documents`, `BM25Index`, `tokenize_for_bm25`. The notebook's `USE_LLAMAINDEX_BM25` dual path and `llamaindex_bm25` do **not** port — keep only the manual `BM25Okapi` implementation.

**Interfaces:**
- Consumes: `Settings`, `chunk_page`, `detect_table_content` (Task 3).
- Produces:
  - `tokenize_for_bm25(text: str) -> list[str]`
  - `class BM25Index` — `__init__(self, nodes: list[TextNode])`, `search(self, query: str, top_k: int) -> list[tuple[TextNode, float]]`
  - `build_nodes(pages: list[dict], encoder, cfg: Settings) -> list[TextNode]`
  - `@dataclass class Store` — fields `vector_index: VectorStoreIndex`, `bm25: BM25Index`, `nodes: list[TextNode]`
  - `build_store(pages: list[dict], cfg: Settings) -> Store`

Each `TextNode.metadata` carries: `filename`, `page_number`, `doc_type`, `source_type`, `section_title`, `table_present`, `chunk_id`. `chunk_id` is unique across the corpus and is part of the RRF key in Task 7.

- [ ] **Step 1: Write the failing tests**

```python
def test_nodes_carry_required_metadata(stub_encoder):
    pages = [{"filename": "a.pdf", "page_number": 1, "text": "1. PAYMENTS\n" + "word " * 50,
              "source_type": "docling", "doc_type": "Promissory Note"}]
    nodes = build_nodes(pages, stub_encoder, Settings())
    assert nodes and set(NODE_METADATA_KEYS) <= set(nodes[0].metadata)

def test_chunk_ids_are_unique_across_pages(stub_encoder):
    pages = [{"filename": "a.pdf", "page_number": n, "text": "word " * 50,
              "source_type": "docling", "doc_type": "Unknown"} for n in (1, 2, 3)]
    ids = [n.metadata["chunk_id"] for n in build_nodes(pages, stub_encoder, Settings())]
    assert len(ids) == len(set(ids))

def test_bm25_tokenizer_keeps_currency_and_percent():
    tokens = tokenize_for_bm25("The rate is 6.5% and the payment is $1,234")
    assert any("%" in t for t in tokens) and any("$" in t for t in tokens)

def test_bm25_tokenizer_drops_stopwords():
    assert "the" not in tokenize_for_bm25("the interest rate")

def test_bm25_search_ranks_exact_term_match_first(text_nodes):
    idx = BM25Index(text_nodes)
    top = idx.search("prepayment penalty", top_k=3)
    assert "prepayment" in top[0][0].get_content().lower()

def test_bm25_search_on_empty_corpus_returns_empty():
    assert BM25Index([]).search("anything", top_k=5) == []
```

Add a `text_nodes` fixture to `conftest.py`: a handful of `TextNode`s with distinct text and full metadata, reusable by Tasks 7–9.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_index.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `src/mortgage_rag/index.py`**

`build_nodes` loops pages → `chunk_page` → one `TextNode` per chunk with the metadata above, `table_present` from `detect_table_content`, and a corpus-wide incrementing `chunk_id`.

`BM25Index` wraps `BM25Okapi` over tokenized node text; `search` returns `(node, score)` sorted descending, dropping zero scores, and returns `[]` for an empty corpus (guard before constructing `BM25Okapi`, which rejects an empty corpus).

`build_store` calls `get_encoder(cfg)` and `build_nodes`, then constructs `VectorStoreIndex(nodes=..., embed_model=get_embed_model(cfg))` — **pass `embed_model` explicitly; never set `llama_index.core.Settings`.** Because nodes are pre-chunked, use the `VectorStoreIndex(nodes=...)` constructor, not `from_documents`, so no re-splitting occurs. Both getters come from Task 2 and resolve to one loaded model.

Module-level `NODE_METADATA_KEYS: tuple[str, ...]` names the seven keys.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_index.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mortgage_rag/index.py tests/test_index.py tests/conftest.py
git commit -m "feat: add node building, BM25 index, and Store replacing globals"
```

---

### Task 7: Retrieval, RRF, and query processing

**Files:**
- Create: `src/mortgage_rag/retrieval.py`
- Test: `tests/test_rrf.py`, `tests/test_retrieval.py`

**Source:** port `vector_search`, `reciprocal_rank_fusion`, `hybrid_retrieve`, `rewrite_query`, `decompose_query`. **Fixes bug 1** from the spec.

**Interfaces:**
- Consumes: `Settings`, `Store`, `BM25Index` (Task 6).
- Produces:
  - `node_key(node: TextNode) -> tuple[str, int, int]` — `(filename, page_number, chunk_id)`
  - `vector_search(query: str, store: Store, cfg: Settings) -> list[tuple[TextNode, float]]`
  - `reciprocal_rank_fusion(result_lists: list[list[tuple[TextNode, float]]], k: int) -> list[tuple[TextNode, float]]`
  - `retrieve(query: str, store: Store, cfg: Settings) -> list[tuple[TextNode, float]]` — dispatches on `cfg.retrieval_mode`
  - `rewrite_query(query: str, llm) -> str`
  - `decompose_query(query: str, llm, cfg: Settings) -> list[str]`

- [ ] **Step 1: Write the failing tests**

`tests/test_rrf.py` — the bug-1 regression is the first test:

```python
def test_chunks_sharing_a_200_char_prefix_are_not_collapsed():
    shared = "WHEREAS the Borrower and the Lender agree as follows " * 5  # >200 chars
    a = make_node(text=shared + " FIRST DISTINCT TAIL", chunk_id=1, page_number=1)
    b = make_node(text=shared + " SECOND DISTINCT TAIL", chunk_id=2, page_number=1)
    fused = reciprocal_rank_fusion([[(a, 0.9), (b, 0.8)]], k=60)
    assert len(fused) == 2

def test_same_node_in_two_lists_is_fused_once_with_summed_score():
    a = make_node(text="alpha", chunk_id=1)
    b = make_node(text="beta", chunk_id=2)
    fused = reciprocal_rank_fusion([[(a, 0.9), (b, 0.1)], [(a, 0.5)]], k=60)
    assert len(fused) == 2
    assert fused[0][0].metadata["chunk_id"] == 1
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 61)

def test_rank_order_is_by_fused_score_descending():
    ...
def test_empty_input_returns_empty():
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []
```

`tests/test_retrieval.py`:

```python
@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
def test_retrieve_dispatches_on_mode(mode, fake_store, monkeypatch):
    # assert only the expected search function(s) were called
    ...

def test_rewrite_falls_back_to_original_on_empty_llm_output(fake_llm_returning):
    q = "what is the rate?"
    assert rewrite_query(q, fake_llm_returning("")) == q

def test_rewrite_falls_back_when_output_absurdly_long(fake_llm_returning):
    q = "what is the rate?"
    assert rewrite_query(q, fake_llm_returning("x" * 600)) == q

def test_decompose_returns_original_when_llm_returns_nothing(fake_llm_returning):
    assert decompose_query("q?", fake_llm_returning(""), Settings()) == ["q?"]

def test_decompose_caps_at_max_sub_queries(fake_llm_returning):
    llm = fake_llm_returning("\n".join(f"sub question number {i}?" for i in range(10)))
    assert len(decompose_query("q?", llm, Settings(max_sub_queries=4))) == 4
```

Add `make_node(**kw)` and `fake_store` to `conftest.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rrf.py tests/test_retrieval.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `src/mortgage_rag/retrieval.py`**

Port the notebook functions with these changes:

- `reciprocal_rank_fusion` keys its accumulator on `node_key(node)`, **not** `doc.text[:200]`. This is the bug fix; the first test is its regression guard.
- Everything returns `TextNode`, never re-wrapped `Document`. The notebook's `Document(text=n.get_text(), metadata=n.metadata)` round-trip is deleted.
- `retrieve` dispatches on `cfg.retrieval_mode`: `vector` → `vector_search` alone; `bm25` → `store.bm25.search` alone; `hybrid` → both, fused through RRF with `cfg.rrf_k`. All three truncate to `cfg.top_k`.
- `vector_search` uses `VectorIndexRetriever(index=store.vector_index, similarity_top_k=cfg.top_k)`.
- `decompose_query` caps at `cfg.max_sub_queries`. Keep the notebook's bullet/number stripping and its `len(line) > 10` filter.
- Replace bare `except:` with `except Exception`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rrf.py tests/test_retrieval.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mortgage_rag/retrieval.py tests/test_rrf.py tests/test_retrieval.py tests/conftest.py
git commit -m "fix: key RRF on node identity instead of text prefix

Two chunks sharing a 200-character prefix collapsed into one, silently
reducing recall on boilerplate-heavy mortgage instruments."
```

---

### Task 8: Reranking and context cleanup

**Files:**
- Create: `src/mortgage_rag/rerank.py`
- Test: `tests/test_rerank.py`, `tests/test_context_cleanup.py`

**Source:** port `rerank_results`, `clean_context`. **Fixes bug 2** from the spec.

**Interfaces:**
- Consumes: `Settings`, `TextNode`.
- Produces:
  - `rerank(query: str, results: list[tuple[TextNode, float]], cross_encoder, cfg: Settings) -> list[tuple[TextNode, float]]`
  - `clean_context(reranked: list[tuple[TextNode, float]], encoder, cfg: Settings) -> list[tuple[TextNode, float]]`

- [ ] **Step 1: Write the failing tests**

`tests/test_rerank.py` — the bug-2 regression is the first test:

```python
def test_cross_encoder_receives_full_chunk_text_not_512_chars(recording_cross_encoder):
    long_text = "word " * 400            # ~2000 characters
    node = make_node(text=long_text)
    rerank("query", [(node, 0.0)], recording_cross_encoder, Settings())
    _, passed_text = recording_cross_encoder.pairs[0]
    assert passed_text == long_text      # not long_text[:512]

def test_rerank_sorts_descending_and_truncates_to_top_n(recording_cross_encoder):
    ...
def test_rerank_on_empty_results_returns_empty(recording_cross_encoder):
    assert rerank("q", [], recording_cross_encoder, Settings()) == []
```

`tests/test_context_cleanup.py`:

```python
def test_score_filter_applies_when_min_rerank_score_is_set(stub_encoder):
    kept = clean_context([(make_node(chunk_id=1), 5.0), (make_node(chunk_id=2), -9.0)],
                         stub_encoder, Settings(min_rerank_score=0.0))
    assert [n.metadata["chunk_id"] for n, _ in kept] == [1]

def test_no_score_filter_by_default(stub_encoder):
    pairs = [(make_node(chunk_id=1), -50.0), (make_node(chunk_id=2), -60.0)]
    assert len(clean_context(pairs, stub_encoder, Settings())) == 2

def test_keeps_highest_scoring_chunk_when_all_below_threshold(stub_encoder):
    pairs = [(make_node(chunk_id=1), -1.0), (make_node(chunk_id=2), -2.0)]
    kept = clean_context(pairs, stub_encoder, Settings(min_rerank_score=10.0))
    assert [n.metadata["chunk_id"] for n, _ in kept] == [1]

def test_near_duplicates_are_removed(identical_vector_encoder):
    pairs = [(make_node(chunk_id=1, text="a " * 20), 1.0),
             (make_node(chunk_id=2, text="a " * 20), 0.9)]
    assert len(clean_context(pairs, identical_vector_encoder, Settings())) == 1

def test_token_budget_truncates_the_tail(distinct_vector_encoder):
    pairs = [(make_node(chunk_id=i, text="word " * 500), 1.0) for i in range(4)]
    kept = clean_context(pairs, distinct_vector_encoder, Settings(max_context_tokens=2000))
    assert len(kept) < 4

# Review Focus 1
def test_single_oversized_chunk_is_still_returned(distinct_vector_encoder):
    pairs = [(make_node(chunk_id=1, text="word " * 5000), 1.0)]
    kept = clean_context(pairs, distinct_vector_encoder, Settings(max_context_tokens=2000))
    assert len(kept) == 1
```

Add `recording_cross_encoder` (records the `(query, text)` pairs it was given, returns descending scores), `identical_vector_encoder`, and `distinct_vector_encoder` to `conftest.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rerank.py tests/test_context_cleanup.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `src/mortgage_rag/rerank.py`**

`rerank` builds pairs from **full** `node.get_content()` — no character slice; the cross-encoder's own `max_length` truncates. Sort descending, truncate to `cfg.rerank_top_n`.

`clean_context` runs three stages in order: score filter (skipped entirely when `cfg.min_rerank_score is None`; when set and everything falls below, keep the single highest-scoring item), near-duplicate removal against `cfg.dedup_threshold` using the injected encoder, then the `cfg.max_context_tokens` budget — **which always admits the first chunk regardless of size**, then breaks on the first subsequent chunk that would exceed the budget. That first-chunk guarantee is Review Focus 1; without it an oversized top chunk yields an empty context.

Use a numpy dot product for cosine similarity on normalised vectors.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rerank.py tests/test_context_cleanup.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mortgage_rag/rerank.py tests/test_rerank.py tests/test_context_cleanup.py tests/conftest.py
git commit -m "fix: send full chunk text to the cross-encoder

The reranker received a 512-CHARACTER slice while its max_length is 512
TOKENS, discarding roughly three quarters of every chunk before scoring."
```

---

### Task 9: Pipeline entry point and call budgets

**Files:**
- Create: `src/mortgage_rag/pipeline.py`
- Modify: `tests/test_import_purity.py` (remove the `importorskip` guards from Task 2)
- Test: `tests/test_pipeline.py`, `tests/test_call_budget.py`

**Source:** port `ask_v3`. The notebook's separate `ask()` does not port — one entry point replaces both.

**Interfaces:**
- Consumes: everything from Tasks 1–8.
- Produces:
  - `ask(query: str, store: Store, cfg: Settings, llm=None, encoder=None, cross_encoder=None) -> dict`
  - `build_prompt(query: str, cleaned: list[tuple[TextNode, float]]) -> str`

The three model parameters default to `None`, in which case `ask` resolves them from `models.py`. Tests pass fakes. Return shape:

```python
{"answer": str, "sources": list[dict], "num_chunks_used": int,
 "confidence": float, "debug": dict}
```

Each `sources` entry: `filename`, `page_number`, `doc_type`, `section_title`, `score`, `preview`. `debug` holds `original_query`, `rewritten_query`, `sub_queries`.

Two new `conftest.py` fixtures this task needs: `fake_models` — a dict
`{"encoder": ..., "cross_encoder": ...}` of call-counting stubs, splatted into `ask`;
and `empty_store` — a `Store` whose vector index and BM25 index both hold zero nodes.

- [ ] **Step 1: Write the failing tests**

`tests/test_call_budget.py`:

```python
def test_full_config_makes_exactly_three_llm_calls(fake_llm, fake_store, fake_models):
    ask("what is the interest rate?", fake_store, Settings(), llm=fake_llm, **fake_models)
    assert fake_llm.calls == 3          # rewrite + decompose + answer

def test_baseline_config_makes_exactly_one_llm_call(fake_llm, fake_store, fake_models):
    cfg = Settings(retrieval_mode="vector", use_rewrite=False,
                   use_decomposition=False, use_rerank=False)
    ask("what is the interest rate?", fake_store, cfg, llm=fake_llm, **fake_models)
    assert fake_llm.calls == 1          # answer only

def test_rerank_disabled_skips_the_cross_encoder(fake_llm, fake_store, fake_models):
    cfg = Settings(use_rerank=False)
    ask("q", fake_store, cfg, llm=fake_llm, **fake_models)
    assert fake_models["cross_encoder"].calls == 0
```

`tests/test_pipeline.py`:

```python
# Review Focus 2
def test_zero_retrieval_results_returns_documented_shape(fake_llm, empty_store, fake_models):
    result = ask("anything", empty_store, Settings(), llm=fake_llm, **fake_models)
    assert result["num_chunks_used"] == 0
    assert result["sources"] == []
    assert result["confidence"] == 0.0
    assert isinstance(result["answer"], str)

def test_sources_carry_citation_metadata(fake_llm, fake_store, fake_models):
    result = ask("q", fake_store, Settings(), llm=fake_llm, **fake_models)
    assert set(result["sources"][0]) == {
        "filename", "page_number", "doc_type", "section_title", "score", "preview"}

def test_debug_records_rewritten_query_and_sub_queries(fake_llm, fake_store, fake_models):
    result = ask("q", fake_store, Settings(), llm=fake_llm, **fake_models)
    assert set(result["debug"]) == {"original_query", "rewritten_query", "sub_queries"}

def test_prompt_contains_grounding_and_refusal_instruction(fake_store):
    prompt = build_prompt("q", [(make_node(text="ctx"), 1.0)])
    assert "ONLY from the provided context" in prompt
    assert "cannot find this information" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py tests/test_call_budget.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `src/mortgage_rag/pipeline.py`**

`ask` runs the documented flow: optional rewrite → optional decompose → `retrieve` per sub-query → dedup across sub-queries on `node_key` → optional `rerank` → `clean_context` → `build_prompt` → `llm.complete`. Return early with the zero-results shape when retrieval yields nothing, **before** computing `np.mean` on an empty list.

`build_prompt` keeps the notebook's prompt text verbatim, including the `[Source N: filename, Page N, Type: ..., Section: ...]` header per chunk and the refusal instruction. That wording is the grounding contract sub-project 2 measures; do not reword it.

`confidence` is the mean of the cleaned chunks' scores, rounded to 4 places.

- [ ] **Step 4: Remove the `importorskip` guards in `tests/test_import_purity.py`**

All five parametrised modules now exist.

- [ ] **Step 5: Run the full suite**

Run: `pytest -m "not integration" -v`
Expected: PASS (all tests, all tasks)

- [ ] **Step 6: Commit**

```bash
git add src/mortgage_rag/pipeline.py tests/test_pipeline.py tests/test_call_budget.py tests/test_import_purity.py
git commit -m "feat: add ask() entry point with call-budget tests"
```

---

### Task 10: Smoke test, demo notebook, and CI

**Files:**
- Create: `tests/test_smoke.py`, `notebooks/demo.ipynb`, `.github/workflows/test.yml`

**Interfaces:**
- Consumes: everything.
- Produces: nothing other tasks depend on.

- [ ] **Step 1: Write the integration smoke test**

```python
@pytest.fixture(scope="module")
def real_store():
    cfg = Settings()
    pages = load_pdf("tests/fixtures/cfpb_closing_disclosure.pdf", cfg)
    pages = classify_pages(pages, get_llm(cfg), cfg)
    return build_store(pages, cfg)

@pytest.mark.integration
@pytest.mark.parametrize("question,expected", [
    ("What is the loan amount?", "162,000"),
    ("What is the interest rate?", "3.875"),
    ("Is there a prepayment penalty?", "3,240"),
])
def test_end_to_end_answers_contain_the_correct_figure(real_store, question, expected):
    result = ask(question, real_store, Settings())
    assert expected in result["answer"]
    assert result["num_chunks_used"] > 0
    assert result["sources"][0]["page_number"] >= 1

@pytest.mark.integration
def test_refuses_when_the_answer_is_absent(real_store):
    result = ask("What is the borrower's credit score?", real_store, Settings())
    assert "cannot find" in result["answer"].lower()
```

The refusal test is the one that matters most: it is the only check in this
sub-project that the grounding instruction in `build_prompt` actually holds, and
a pipeline that confidently invents a credit score is worse than one that
retrieves badly.

- [ ] **Step 2: Run the smoke test**

Run: `ANTHROPIC_API_KEY=<key> pytest tests/test_smoke.py -m integration -v`
Expected: PASS (4 tests). Downloads Docling and embedding models on first run; several minutes is normal.

- [ ] **Step 3: Write `.github/workflows/test.yml`**

Trigger on push and pull_request. One job: checkout, `actions/setup-python` at 3.11, `pip install -e .[dev]`, `pytest -m "not integration"`. No API key secret, no model cache — if the workflow ever needs one, the import-purity rule has been broken.

- [ ] **Step 4: Write `notebooks/demo.ipynb`**

Roughly ten cells, importing only from `mortgage_rag`, defining no pipeline logic: build a `Settings`, `load_pdf` the fixture, `classify_pages`, `build_store`, print the page/chunk/doc-type breakdown, then `ask` three questions printing answers with citations. Suggested questions: "What is the loan amount?", "What are the closing costs?", "Is there a prepayment penalty?".

- [ ] **Step 5: Verify the notebook runs top to bottom**

Run: `jupyter nbconvert --execute --to notebook --inplace notebooks/demo.ipynb`
Expected: exit 0.

- [ ] **Step 6: Verify the acceptance criteria**

Run each and confirm:
- `pip install -e .[dev] && pytest -m "not integration"` → passes with no GPU, no API key, no downloads
- `python -c "import mortgage_rag.pipeline"` → returns without loading torch/docling/sentence_transformers
- `git status` → clean

- [ ] **Step 7: Commit**

```bash
git add tests/test_smoke.py notebooks/demo.ipynb .github/workflows/test.yml
git commit -m "feat: add smoke test, demo notebook, and test CI"
```

---

## Not in this plan

FastAPI service; index persistence; the Gradio rewrite (it stays unported in the notebook export — sub-project 3 rewrites it against the service); Dockerfile; ground-truth dataset; RAGAS; README; deployment. `complete_mortgage_rag_pipeline.py` stays at the repo root untouched as the reference source throughout; deleting it is sub-project 4's call.
