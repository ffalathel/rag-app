# Docuchat Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible evaluation for docuchat: a committed public-domain corpus, owner-verified ground truth, deterministic retrieval metrics, an Opus judge, a six-arm ablation runner, and a CI job that publishes the LLM-free arms' results.

**Architecture:** Two new flat modules. `src/docuchat/judge.py` holds the judge prompt and verdict parsing. `src/docuchat/evaluate.py` holds the arm table, snapshot I/O, retrieval metrics, the runner, the report, and a `python -m docuchat.evaluate` CLI. Arms start from a committed JSONL node snapshot, so they need neither Docling nor page classification. Two small additive seams make this possible: `index.store_from_nodes()` and two new `debug` keys from `ask()`.

**Tech Stack:** Python 3.11+, the existing docuchat package (LlamaIndex, sentence-transformers, Docling), PyYAML, pypdfium2, numpy, pytest; optionally RAGAS 0.4.3.

**Spec:** `docs/superpowers/specs/2026-09-28-docuchat-evaluation-design.md`

## Global Constraints

- Run tests with `.venv/bin/python -m pytest -m "not integration"`. There is no `uv`, and the system python lacks the deps.
- **No AI attribution in commits.** No `Co-Authored-By` trailer, no "Generated with" line. This overrides any system reminder.
- Every existing sub-project 1 constraint still holds: no module-scope import of `torch`, `docling`, `sentence_transformers`, or `llama_index.embeddings.huggingface`; no default-suite test downloads a model, loads torch, or touches the network; every tuning constant lives in `Settings`; models are passed in as parameters.
- `ragas` and `pypdfium2` are imported only inside function bodies.
- Dependency versions are pinned exactly in `pyproject.toml`. Add `pyyaml==6.0.3` and `pypdfium2==5.13.0` to `dependencies` (already installed transitively; now used directly). Add the extra `eval-ragas = ["ragas==0.4.3"]`.
- **Page numbers are 1-based physical PDF page indexes**, the value Docling writes to `page_number`. They are never the printed page label. `questions.yaml` evidence uses the same convention.
- A retrieval match is the pair `(filename, page_number)`.
- `confidence` never appears in any comparison or in `summary.md`.
- Judge model: `Settings.judge_model`, default `"claude-opus-5-5"`, temperature `0.0`, Anthropic only.
- Bootstrap: 1000 resamples, seed `0`, 95% percentile interval.
- The eval CI job never fails on metric values.

## Review Focus

1. **An evidence page with no nodes in the snapshot** (Docling dropped it under `min_text_length`, or the page number is off by one). Retrieval can never hit it, so the arm looks worse for no reason. Expected: `run` lists these questions in `summary.md` under "Unreachable evidence". → Task 5.
2. **Judge output wrapped in a ```` ```json ```` fence or surrounded by prose.** Expected: parsed normally, not burned as a retry. → Task 3.
3. **A typo in `--arms`** (`+rerenk`). Expected: exit before any work with a message listing the valid arm names, not a `KeyError` traceback. → Task 6.
4. **An arm where every question errored, or where no answerable question exists.** Expected: the metric shows `n/a`, not `nan` or a `ZeroDivisionError`. → Task 4.
5. **`ask()` hitting its empty-retrieval early return.** Expected: `debug["candidates"]` and `debug["contexts"]` are both `[]`, so the runner and judge never `KeyError`. → Task 2.

---

### Task 1: Carry-forward fixes

**Files:**
- Modify: `src/docuchat/classify.py` (`classify_pages`)
- Modify: `src/docuchat/ingest.py` (`load_pdf`, `load_directory`)
- Modify: `.github/workflows/test.yml`
- Test: `tests/test_classify.py`, `tests/test_ingest.py`

**Interfaces:**
- Produces: `classify_pages(pages, llm, cfg) -> list[dict]` (signature unchanged; the behavior is now per document). `load_pdf(path, cfg, converter=None) -> list[dict]`. `_make_converter(cfg)` in `ingest.py`, which returns a Docling `DocumentConverter`.

- [ ] **Step 1: Write the failing classification tests** in `tests/test_classify.py`

```python
def test_label_is_per_document_not_per_page(fake_llm):
    pages = [
        {"text": "CLOSING DISCLOSURE loan terms", "page_number": 1, "filename": "cd.pdf"},
        {"text": "Escrow account details " + "x" * 80, "page_number": 2, "filename": "cd.pdf"},
        {"text": "Loan Estimate comparison " + "x" * 80, "page_number": 3, "filename": "cd.pdf"},
    ]
    classify_pages(pages, fake_llm, Settings())
    assert [p["doc_type"] for p in pages] == ["Closing Disclosure"] * 3
    assert fake_llm.calls == 0


def test_each_document_classified_once(fake_llm):
    pages = [
        {"text": "x" * 100, "page_number": 1, "filename": "a.pdf"},
        {"text": "x" * 100, "page_number": 2, "filename": "a.pdf"},
        {"text": "PROMISSORY NOTE", "page_number": 1, "filename": "b.pdf"},
    ]
    classify_pages(pages, fake_llm, Settings())
    assert fake_llm.calls == 1  # a.pdf's first page only; b.pdf hits the heuristic
    assert [p["doc_type"] for p in pages] == ["Other", "Other", "Promissory Note"]
```

Update `test_llm_used_only_when_heuristic_returns_unknown`. Both of its pages share `a.pdf`, so the second page now inherits "Closing Disclosure": assert `fake_llm.calls == 0` and rename the test to `test_later_pages_inherit_the_first_pages_label`.

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_classify.py -v`. Expected: the two new tests and the renamed one FAIL.

- [ ] **Step 3: Implement per-document `classify_pages`.** Group pages by `filename`, preserving order. Classify the page with the lowest `page_number` in each group, using the existing heuristic-then-LLM logic, and assign that label to every page in the group. Add a one-line `# ponytail:` comment: a PDF that bundles several document types gets one label; split per section if the corpus ever contains loan packages. Update the module docstring.

- [ ] **Step 4: Write the failing converter test** in `tests/test_ingest.py`. Monkeypatch `docuchat.ingest._make_converter` with a counting stub and `docuchat.ingest.load_pdf` with a stub that records the `converter` kwarg it receives. Create two empty `.pdf` files in `tmp_path`, call `load_directory`, and assert that `_make_converter` was called once and that both `load_pdf` calls received the same object.

- [ ] **Step 5: Implement.** Extract the converter construction from `load_pdf` into `_make_converter(cfg)` (Docling imports stay inside it). `load_pdf` takes `converter=None` and builds one only when none is given. `load_directory` builds one and passes it to every `load_pdf` call.

- [ ] **Step 6: CPU torch in CI.** In `.github/workflows/test.yml`, add `- run: pip install torch --index-url https://download.pytorch.org/whl/cpu` before `pip install -e .[dev]`.

- [ ] **Step 7: Run the full suite.** `.venv/bin/python -m pytest -m "not integration" -q`. Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/docuchat/classify.py src/docuchat/ingest.py tests/test_classify.py tests/test_ingest.py .github/workflows/test.yml
git commit -m "fix: classify per document, share one Docling converter, CPU torch in CI"
```

---

### Task 2: Pipeline seams and settings

**Files:**
- Modify: `src/docuchat/index.py` (`build_store`), `src/docuchat/pipeline.py` (`ask`), `src/docuchat/config.py` (`Settings`), `src/docuchat/models.py`, `pyproject.toml`
- Test: `tests/test_index.py`, `tests/test_pipeline.py`, `tests/test_models.py`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `store_from_nodes(nodes: list[TextNode], cfg: Settings) -> Store` in `index.py`. `build_store(pages, cfg)` becomes `store_from_nodes(build_nodes(pages, get_encoder(cfg), cfg), cfg)`.
  - `ask()` adds `debug["candidates"]: list[tuple[str, int]]`, the `(filename, page_number)` of the deduplicated pool before rerank/truncation, in pool order, with duplicate pages kept. It also adds `debug["contexts"]: list[str]`, the full text of each chunk in the final context, in the same order as `sources`. Both are `[]` on the empty-retrieval return.
  - `Settings.judge_model: str = "claude-opus-5-5"`, in a new `# Evaluation` group.
  - `get_judge_llm(cfg) -> Anthropic` in `models.py`. It raises `RuntimeError` naming `ANTHROPIC_API_KEY` when the key is unset. Otherwise it returns a separately `lru_cache`d `_load_judge_llm(judge_model)` built with `temperature=0.0` and `max_tokens=cfg.max_new_tokens`. Its cache is separate so judging never evicts the answer LLM from `_load_llm`'s `maxsize=1` cache.
  - `pyproject.toml` gets the dependency additions from Global Constraints.

- [ ] **Step 1: Write the failing tests**
  - `test_store_from_nodes_keeps_node_identity(make_node, monkeypatch)`: monkeypatch `docuchat.index.get_embed_model` to return `MockEmbedding(embed_dim=8)`. Pass two nodes and assert `store.nodes` is the same list and `store.bm25.nodes` has length 2.
  - `test_ask_records_candidates_and_contexts(fake_store, fake_llm, fake_models)`: use `Settings(use_rewrite=False, use_decomposition=False, use_rerank=True, retrieval_mode="bm25")`. Assert `debug["candidates"] == [("a.pdf", 1)] * 3`, `len(debug["contexts"]) == result["num_chunks_used"]`, and that every `debug["contexts"]` entry is one of the full node texts `{"doc 0", "doc 1", "doc 2"}` (not a truncated preview).
  - `test_empty_retrieval_has_empty_candidates_and_contexts(empty_store, fake_llm, fake_models)`: both keys equal `[]`.
  - `test_judge_model_default`: `Settings().judge_model == "claude-opus-5-5"`, and `DOCUCHAT_JUDGE_MODEL` overrides it through `from_env`.
  - `test_get_judge_llm_requires_key(monkeypatch)`: with `ANTHROPIC_API_KEY` deleted, `pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY")`.

- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement the four interface items above.** Keep `import_purity` intact: the Anthropic import stays inside `_load_judge_llm`.
- [ ] **Step 4: Run the full suite.** Expected: all pass. The existing `build_store` and `ask` tests are unchanged.
- [ ] **Step 5: Commit** with the message `feat: add store_from_nodes, ask() candidates/contexts, and judge_model setting`.

---

### Task 3: Judge

**Files:**
- Create: `src/docuchat/judge.py`
- Test: `tests/test_judge.py`; add `"docuchat.judge"` to the list in `tests/test_import_purity.py`

**Interfaces:**
- Consumes: any object with `.complete(prompt) -> str-able` (the Task 2 judge LLM, or `_FakeLLM` in tests).
- Produces: `judge_answer(question: str, reference_answer: str, contexts: list[str], answer: str, llm) -> dict`. On success it returns `{"refused": bool, "correctness": 0|1|2, "faithful": bool, "rationale": str}`. On failure it returns `{"judge_error": str}`. For an unanswerable question, the runner passes `reference_answer = "UNANSWERABLE: the documents do not contain this information."`, which is the constant `UNANSWERABLE_REFERENCE` exported from this module.

- [ ] **Step 1: Write the failing tests** using `fake_llm_returning` (the fixture returns the `_FakeLLM` class). Add a local two-response fake for the retry cases.
  - `test_parses_bare_json`: response `'{"refused": false, "correctness": 2, "faithful": true, "rationale": "ok"}'` gives that dict, and `llm.calls == 1`.
  - `test_parses_fenced_json_without_retry`: the same JSON inside a ```` ```json ... ``` ```` block with a sentence before it parses with `llm.calls == 1` (Review Focus 2).
  - `test_retries_once_then_succeeds`: first response `"not json"`, then valid JSON. The result is valid, and `calls == 2`.
  - `test_two_failures_give_judge_error`: `"not json"` twice gives a dict with the key `judge_error`, `calls == 2`.
  - `test_out_of_range_correctness_is_a_failure`: `correctness: 3` is treated as malformed, so it retries.
  - `test_prompt_contains_all_inputs`: the question, reference, every context string, and the answer all appear in `llm.prompts[0]`.

- [ ] **Step 2: Run them.** Expected: FAIL (module missing).

- [ ] **Step 3: Implement `judge.py`.** Parse by taking the substring from the first `{` to the last `}` and passing it to `json.loads`. Validate that `refused` and `faithful` are `bool`, `correctness` is an `int` in `{0, 1, 2}`, and `rationale` is a `str`. Anything else counts as malformed. The prompt text is fixed here:

```text
You are grading an answer produced by a document question-answering system.

Question: {question}

Reference answer: {reference_answer}

Retrieved context given to the system:
{contexts joined with "\n\n---\n\n"}

System answer: {answer}

Grade the system answer. Respond with ONLY a JSON object with these keys:
- "refused": true if the answer declines to answer or says the information is not available, else false.
- "correctness": 2 if it matches the reference answer's facts, 1 if partially correct or incomplete, 0 if wrong. If the reference is UNANSWERABLE, a refusal scores 2 and any substantive answer scores 0.
- "faithful": true if every factual claim in the answer is supported by the retrieved context (a refusal is faithful), else false.
- "rationale": one sentence explaining the grade.
```

- [ ] **Step 4: Run the tests.** Expected: PASS, including import purity.
- [ ] **Step 5: Commit** with the message `feat: add LLM judge with JSON verdict parsing and one retry`.

---

### Task 4: Metrics and aggregation

**Files:**
- Create: `src/docuchat/evaluate.py` (metrics section)
- Test: `tests/test_eval_metrics.py`; add `"docuchat.evaluate"` to `tests/test_import_purity.py`

**Interfaces:**
- Produces, all pure functions in `evaluate.py`:
  - `retrieval_metrics(evidence: list[tuple[str, int]], context: list[tuple[str, int]], candidates: list[tuple[str, int]]) -> dict` with the keys `hit` (0.0/1.0), `recall`, `mrr`, and `candidate_recall`. `recall` is the number of distinct evidence pages present in `context` divided by the number of distinct evidence pages. `mrr` is `1/rank` of the first `context` position (1-based) that is an evidence page, or `0.0`.
  - `bootstrap_ci(values: list[float], n: int = 1000, seed: int = 0) -> tuple[float, float] | None`. Returns `None` for an empty list. Resample indices with `np.random.default_rng(seed)` and take the 2.5/97.5 percentiles of the resample means.
  - `win_loss(prev: dict[str, float], cur: dict[str, float]) -> tuple[int, int, int]`. Both are keyed by question id; only ids present in both are compared.
  - `aggregate(records: list[dict]) -> dict[str, dict]`: metric name → `{"mean": float | None, "ci": tuple | None, "n": int}`. The metric names and their inclusion rules:
    - `hit`, `recall`, `mrr`, `candidate_recall`: answerable (`kind != "unanswerable"`), non-errored records.
    - `correct_rate` (share with `correctness == 2`), `mean_correctness`, `faithful_rate`: non-errored records with a verdict and no `judge_error`, all kinds.
    - `correct_refusal_rate`: `unanswerable` records with a valid verdict, `refused == True`.
    - `false_refusal_rate`: answerable records with a valid verdict, `refused == True`.
    - `llm_calls`: mean over non-errored records.
    - Also the counts `errors` and `judge_errors`.
  - A record is the per-question dict from Task 6 (see there). This task reads only its keys `id`, `kind`, `error`, `retrieval`, and `verdict`.

- [ ] **Step 1: Write the failing tests.**
  - `retrieval_metrics([("a", 1), ("b", 2)], [("a", 1), ("c", 3)], [("a", 1), ("b", 2)])` returns `{"hit": 1.0, "recall": 0.5, "mrr": 1.0, "candidate_recall": 1.0}`.
  - MRR rank: evidence `[("a", 3)]`, context `[("x", 1), ("y", 1), ("a", 3)]` gives `mrr == pytest.approx(1/3)`.
  - Duplicate context pages count once toward recall.
  - `bootstrap_ci([1.0] * 10) == (1.0, 1.0)`; the same input and seed give the same output twice; `bootstrap_ci([]) is None`.
  - `win_loss({"q1": 1, "q2": 0, "q3": 1}, {"q1": 1, "q2": 1, "q3": 0}) == (1, 1, 1)`.
  - `aggregate` excludes unanswerable records from `recall` and errored records from everything, and counts `errors`.
  - Review Focus 4: `aggregate` over only errored records returns `{"mean": None, "ci": None, "n": 0}` for `recall`, and it raises no numpy `RuntimeWarning` (run the test under `warnings.simplefilter("error")`). `aggregate` over only unanswerable records gives `recall` n=0.

- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement** the functions above. Use `numpy` only; no pandas or scipy.
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat: add retrieval metrics, bootstrap CIs, and aggregation`.

---

### Task 5: Ground-truth loading, validation, and snapshots

**Files:**
- Modify: `src/docuchat/evaluate.py`
- Test: `tests/test_eval_data.py`

**Interfaces:**
- Produces:
  - `EVAL_DIR = Path("eval")`, `CORPUS_DIR = EVAL_DIR / "corpus"`, `QUESTIONS_PATH = EVAL_DIR / "questions.yaml"`, `NODES_DIR = EVAL_DIR / "nodes"`, `RESULTS_DIR = EVAL_DIR / "results"`. Paths are relative to the repo root, and the CLI runs from there.
  - `load_questions(path) -> list[dict]` (`yaml.safe_load`).
  - `validate_questions(questions: list[dict], corpus_dir: Path) -> list[str]` returns human-readable error strings, empty if valid. It checks:
    - the required keys `id`, `question`, `reference_answer`, `evidence`, and `kind`;
    - `kind` is in `{"fact", "table", "multi_hop", "unanswerable"}`;
    - ids are unique;
    - `unanswerable` has empty `evidence` and every other kind has non-empty `evidence`;
    - each evidence `filename` exists in `corpus_dir`;
    - each evidence `page` falls in `1..page_count`, with the page count from `pypdfium2.PdfDocument` (imported inside the function).
  - `INGEST_FIELDS: tuple[str, ...]`: every `Settings` field under the Ingestion, Chunking, Domain, and Classification comment groups, **excluding `embed_model_name`**. Chunking always uses the snapshot's encoder; the embed arm re-embeds only.
  - `corpus_hashes(corpus_dir: Path) -> dict[str, str]`: filename → sha256 hex, for every `*.pdf` (case-insensitive), sorted.
  - `write_snapshot(path: Path, nodes: list[TextNode], corpus_dir: Path, cfg: Settings) -> None`. The header line is `{"corpus": corpus_hashes(...), "settings": {f: getattr(cfg, f) for f in INGEST_FIELDS}}`. Each following line is `{"text": node.get_content(), "metadata": node.metadata}`.
  - `class StaleSnapshotError(Exception)`
  - `load_snapshot(path: Path, corpus_dir: Path, cfg: Settings) -> list[TextNode]`. It raises `StaleSnapshotError` when the header corpus hashes differ from `corpus_hashes(corpus_dir)` or when any `INGEST_FIELDS` value differs from `cfg`. The message names the differing files or fields and the command `python -m docuchat.evaluate ingest --profile <profile>`.
  - `unreachable_evidence(questions: list[dict], nodes: list[TextNode]) -> list[str]`: ids of answerable questions for which at least one evidence `(filename, page)` has no node in `nodes` (Review Focus 1).

- [ ] **Step 1: Write the failing tests.** Use `tmp_path` and a copy of `tests/fixtures/cfpb_closing_disclosure.pdf` as the corpus (6 pages).
  - Validation catches each rule in one parametrized test: a missing key, a bad kind, a duplicate id, an unanswerable question with evidence, an answerable question without evidence, an unknown filename, page 0, and page 7. Each case asserts exactly one error string that mentions the offending question id. A valid question list returns `[]`.
  - Snapshot round-trip: `make_node` nodes are written, then loaded, with equal text and metadata.
  - Stale on corpus: append a byte to the copied PDF after writing, and `load_snapshot` raises `StaleSnapshotError` whose message contains `"ingest"`.
  - Stale on settings: loading with `Settings(chunk_max_tokens=256)` raises.
  - Not stale on embed model: loading with `Settings(embed_model_name="BAAI/bge-base-en-v1.5")` succeeds.
  - `unreachable_evidence` flags a question whose evidence page has no node and ignores unanswerable questions.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them.** Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat: add ground-truth validation and node snapshots`.

---

### Task 6: Arms, runner, report, and CLI

**Files:**
- Modify: `src/docuchat/evaluate.py`
- Test: `tests/test_eval_runner.py`

**Interfaces:**
- Consumes: `ask` (with its Task 2 debug keys), `store_from_nodes`, `get_llm`, `get_judge_llm`, `judge_answer`, `UNANSWERABLE_REFERENCE`, the Task 4 metrics, and the Task 5 loaders.
- Produces:
  - `ARMS: dict[str, tuple[str, dict]]`, where the name maps to `(snapshot profile, Settings overrides)`, in this order. `FULL = {}` means `Settings()` defaults.

    | name | profile | overrides |
    |---|---|---|
    | `baseline` | `mortgage` | `retrieval_mode="vector", use_rewrite=False, use_decomposition=False, use_rerank=False` |
    | `+hybrid` | `mortgage` | `retrieval_mode="hybrid", use_rewrite=False, use_decomposition=False, use_rerank=False` |
    | `+rerank` | `mortgage` | `retrieval_mode="hybrid", use_rewrite=False, use_decomposition=False, use_rerank=True` |
    | `full` | `mortgage` | `{}` |
    | `full/generic` | `generic` | `domain_profile="generic"` |
    | `full/embed-bge-base` | `mortgage` | `embed_model_name="BAAI/bge-base-en-v1.5"` |

  - `arm_settings(name: str, base: Settings) -> Settings`, using `dataclasses.replace(base, **overrides)`.
  - `needs_llm(cfg: Settings) -> bool`: `cfg.use_rewrite or cfg.use_decomposition`.
  - `class StubLLM`: `complete(prompt)` returns `""`. It is used as the answer LLM when not judging and the arm doesn't `needs_llm`.
  - `run_arm(name: str, questions: list[dict], store, cfg: Settings, llm, judge_llm=None) -> list[dict]`. It wraps `llm` in a call counter per question. Each record is `{"id", "kind", "question", "answer", "sources": [[filename, page]], "candidates": [[filename, page]], "retrieval": retrieval_metrics(...), "llm_calls": int, "latency_s": float}`, plus `"verdict"` when `judge_llm` is given, plus `"error": str` in place of everything after `question` when `ask()` or the judge raises. A judge failure is a returned `judge_error` verdict, not an exception.
  - `render_summary(arm_records: dict[str, list[dict]], skipped: dict[str, str], unreachable: list[str], validation: dict | None) -> str`, which returns Markdown:
    - a header with the date, git commit (`git rev-parse --short HEAD`, or `"unknown"`), question count, and judge model if judged;
    - one table row per arm with each `aggregate` metric as `mean [lo, hi]` (3 decimals) or `n/a`;
    - a "vs previous arm" table of win/loss/tie on `recall`, and on `correctness` when judged, where "previous" is the prior entry in `ARMS` order among the arms run;
    - "Skipped arms" with reasons, "Unreachable evidence" ids, and "Judge validation" agreement when `validation` is given.
  - `main(argv: list[str] | None = None) -> int`, with `if __name__ == "__main__": raise SystemExit(main())`. Subcommands:
    - `ingest --profile {mortgage,generic}`: `load_directory(CORPUS_DIR, cfg)` → `classify_pages(pages, get_llm(cfg), cfg)` → `build_nodes(pages, get_encoder(cfg), cfg)` → `write_snapshot(NODES_DIR / f"{profile}.jsonl", ...)`, with `cfg = Settings.from_env(domain_profile=profile)`.
    - `run [--arms a,b,...] [--judge] [--ragas]` (default: all arms). It validates arm names first, printing the valid names to stderr and returning 2 on an unknown name (Review Focus 3). With `--judge` and `get_judge_llm` raising, it prints the error and returns 2 before any question runs. An arm that `needs_llm` while `get_llm` raises is recorded in `skipped` with the error message. Each arm loads its snapshot (a `StaleSnapshotError` prints and returns 2), builds the store, and runs. It writes `RESULTS_DIR / f"{name.replace('/', '_')}.json"` and `RESULTS_DIR / "summary.md"`, reading `RESULTS_DIR / "judge_validation.yaml"` if present.
    - `judge-sample [--n 10]`: loads the result JSONs and draws `n` judged records with a seed-0 RNG, spread across arms and kinds by a round-robin draw over the `(arm, kind)` groups. It writes `RESULTS_DIR / "judge_validation.yaml"`, a list of `{arm, id, question, reference_answer, answer, judge: {correctness, faithful, refused}, human: {correctness: null, faithful: null, refused: null}}`.
  - `judge_agreement(items: list[dict]) -> dict | None`: over items whose `human` fields are all non-null, the share where all three fields match, plus `n`. Returns `None` when no item is filled in.

- [ ] **Step 1: Write the failing tests** using `fake_store`, `fake_models`, `fake_llm_returning`, and `monkeypatch`.
  - `arm_settings("+rerank", Settings())` has `use_rerank=True, use_rewrite=False`. Every `ARMS` override key is a real `Settings` field.
  - `run_arm` with `StubLLM` on two questions, one answerable (evidence `[{"filename": "a.pdf", "page": 1}]`) and one unanswerable, produces records with all keys. `retrieval["hit"] == 1.0` for the first, and `llm_calls == 1`, which is the answer call.
  - With a judge fake returning valid JSON, each record has a `verdict`. The unanswerable question's judge prompt contains `UNANSWERABLE_REFERENCE`.
  - With `monkeypatch` making `docuchat.evaluate.ask` raise, the record has `error` and the next question still runs.
  - `render_summary` over two arms contains both arm names, `n/a` for the unjudged correctness, a win/loss line, and the skipped-arm reason.
  - `main(["run", "--arms", "+rerenk"])` returns 2, and stderr (`capsys`) contains `"+rerank"`.
  - `main(["run", "--judge"])` with `ANTHROPIC_API_KEY` unset returns 2 before `load_snapshot` is called (monkeypatch `load_snapshot` to raise `AssertionError`).
  - `judge_agreement` with one matching, one mismatching, and one unfilled item gives `{"agreement": 0.5, "n": 2}`.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.** `--ragas` raises a clear `SystemExit` message naming `pip install -e .[eval-ragas]` when `ragas` is not importable. The RAGAS computation itself is Task 7.
- [ ] **Step 4: Run the full suite.** Expected: all pass.
- [ ] **Step 5: Commit** with the message `feat: add ablation arms, runner, summary report, and eval CLI`.

---

### Task 7: RAGAS cross-check

**Files:**
- Modify: `src/docuchat/evaluate.py`, `pyproject.toml`
- Test: `tests/test_eval_runner.py`

**Interfaces:**
- Produces: `ragas_scores(records: list[dict], cfg: Settings) -> dict[str, float]` with the keys `ragas_faithfulness` and `ragas_answer_relevancy`. Errored records are skipped. `render_summary` shows these two columns for an arm when its records carry them. `run --ragas` implies `--judge` and stores the result per arm in the arm JSON under `"ragas"`.

- [ ] **Step 1: Install the extra** with `.venv/bin/pip install -e .[eval-ragas]` (the pin is already in `pyproject.toml` from Task 2). Read the installed RAGAS 0.4.3 API for `evaluate`, the `faithfulness` and `answer_relevancy` metrics, the LLM wrapper, and the embeddings wrapper, from its source under `.venv/lib/*/site-packages/ragas/`. Use `cfg.judge_model` via Anthropic as the LLM, and `cfg.embed_model_name` via a HuggingFace wrapper for answer relevancy. Record the exact classes used in a docstring.
- [ ] **Step 2: Write the test** `test_ragas_missing_gives_install_hint`: monkeypatch `builtins.__import__` to raise `ImportError` for `ragas`, and assert that `main(["run", "--ragas"])` raises `SystemExit` whose message contains `eval-ragas`. Mark `test_ragas_scores_on_two_records` `@pytest.mark.integration`: two hand-built records give floats in `[0, 1]`.
- [ ] **Step 3: Implement** `ragas_scores`. Import `ragas` inside the function.
- [ ] **Step 4: Run** the default suite (it must pass without RAGAS installed: temporarily `pip uninstall -y ragas`, run, then reinstall), then run `-m integration -k ragas` with an API key. Expected: both pass.
- [ ] **Step 5: Commit** with the message `feat: add optional RAGAS cross-check columns`.

---

### Task 8: Corpus

**Files:**
- Create: `eval/corpus/*.pdf`, `eval/corpus/SOURCES.md`
- Test: `tests/test_eval_data.py`

**Interfaces:**
- Produces: 10–20 PDFs in `eval/corpus/`, totalling ≤ 25 MB, with lowercase snake_case filenames. `SOURCES.md` is a table with the columns `file | origin URL | sha256 | document type | pages | scanned`.

- [ ] **Step 1: Collect US-government works only** (public domain, no license question). Use CFPB (`files.consumerfinance.gov`) and HUD sources. Aim for:
  - filled Loan Estimate samples (H-24 series) and filled Closing Disclosure samples (H-25 series, beyond the fixture's H-25(B)), for `table` questions;
  - the CFPB "Your home loan toolkit" booklet and one or two other CFPB/HUD consumer guides, for `fact` and `multi_hop` questions;
  - a Loan Estimate / Closing Disclosure pair for the same sample loan where CFPB publishes one, for cross-document `multi_hop` questions;
  - copy the existing fixture in as `cfpb_closing_disclosure_h25b.pdf`.
  Verify each download is a real PDF (`file` reports "PDF document") and opens in `pypdfium2`.
- [ ] **Step 2: Create the scanned document.** Pick one collected document that nothing else duplicates. Rasterize each page at 150 dpi with `pypdfium2` (`page.render(scale=150/72).to_pil()`) and save the images as an image-only PDF (`PIL.Image.save(..., save_all=True)`) named `<name>_scanned.pdf`. Remove the text original from the corpus. Confirm it has no text layer: `pypdfium2` `get_textpage().get_text_range()` is empty on every page. Mark it `scanned: yes (rasterized from <origin URL>)` in `SOURCES.md`.
- [ ] **Step 3: Write `SOURCES.md`.** Fill every column; compute sha256 with `shasum -a 256`.
- [ ] **Step 4: Write the test** `test_sources_md_matches_corpus`. Parse the `SOURCES.md` table and assert that its set of filenames equals the set of PDFs in `eval/corpus/`, that every sha256 matches `corpus_hashes(CORPUS_DIR)`, and that at least one row is scanned. Run it. Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat: add public-domain evaluation corpus`.

---

### Task 9: Snapshots and draft ground truth — STOP for owner verification

**Files:**
- Create: `eval/nodes/mortgage.jsonl`, `eval/nodes/generic.jsonl`, `eval/questions.yaml`
- Test: `tests/test_eval_data.py`

- [ ] **Step 1: Generate the snapshots.** This needs `ANTHROPIC_API_KEY` and the Docling models. Run `.venv/bin/python -m docuchat.evaluate ingest --profile mortgage` and the same with `--profile generic`. Confirm that the scanned document produced non-empty nodes (grep its filename in `mortgage.jsonl`); if it didn't, OCR failed, and you must stop and report. Spot-check that the `doc_type` values look right per file.
- [ ] **Step 2: Draft `eval/questions.yaml`** with about 40 items in the spec's format: about 15 `fact`, 10 `table`, 10 `multi_hop`, and 5 `unanswerable`. Rules:
  - Write each question by reading the page itself, via the snapshot text or the PDF, and cite physical page indexes.
  - Reference answers are short and factual.
  - `multi_hop` questions need two or more evidence pages, and at least 4 of them span two documents.
  - `unanswerable` questions must be plausible for the corpus (e.g. a borrower's credit score) and absent from every document. Grep the snapshots to confirm.
  - Phrase questions the way a borrower would, not by copying the page's wording, so they don't trivially favor BM25.
  - Put anything uncertain in `notes`.
- [ ] **Step 3: Add the real-data tests** to `tests/test_eval_data.py`:
  - `test_questions_yaml_is_valid`: `validate_questions(load_questions(QUESTIONS_PATH), CORPUS_DIR) == []`.
  - `test_question_mix`: counts per kind fall within ±3 of the targets, and there are ≥ 35 items in total.
  - `test_all_evidence_reachable`: `unreachable_evidence(questions, load_snapshot(NODES_DIR / "mortgage.jsonl", CORPUS_DIR, Settings())) == []`.
  Run them. Expected: PASS. Fix the questions, not the tests.
- [ ] **Step 4: Commit** with the message `feat: add evaluation snapshots and draft ground truth`.
- [ ] **Step 5: STOP.** Ask the owner to review `eval/questions.yaml` and correct it. Do not run any arm until they approve. Commit their corrections with the message `fix: owner corrections to ground truth`.

---

### Task 10: First results, judge validation, and eval CI

**Files:**
- Create: `eval/results/*`, `.github/workflows/eval.yml`

- [ ] **Step 1: Free run.** Run `.venv/bin/python -m docuchat.evaluate run --arms baseline,+hybrid,+rerank` with no API key in the environment (`env -u ANTHROPIC_API_KEY`). Expected: three arm JSONs and a `summary.md`, with zero errors and empty "Unreachable evidence".
- [ ] **Step 2: Judged run.** Run `.venv/bin/python -m docuchat.evaluate run --judge --ragas`. Expected: all six arms and zero `judge_errors`. If the run shows any `errors`, read them and fix the cause before continuing.
- [ ] **Step 3: Judge validation.** Run `.venv/bin/python -m docuchat.evaluate judge-sample --n 10`. **STOP:** ask the owner to fill in the `human` fields in `eval/results/judge_validation.yaml`. Then rerun `run --judge --ragas` so the summary includes agreement. If agreement is below 0.8, report the disagreements to the owner and revise the judge prompt in `judge.py` only with their approval.
- [ ] **Step 4: Eval CI.** Create `.github/workflows/eval.yml`:
  - triggers: `push` and `pull_request` with `paths: ["src/**", "eval/**"]`, plus `workflow_dispatch`;
  - Python 3.11, CPU torch as in Task 1, then `pip install -e .`;
  - `actions/cache@v4` on `~/.cache/huggingface`, keyed on the `embed_model_name` and `cross_encoder_name` defaults;
  - run `python -m docuchat.evaluate run --arms baseline,+hybrid,+rerank`;
  - `cat eval/results/summary.md >> "$GITHUB_STEP_SUMMARY"`;
  - `actions/upload-artifact@v4` of `eval/results/`.
  No step checks metric values.
- [ ] **Step 5: Commit.** Commit the judged results (not the free-run files, which the judged run overwrote), `judge_validation.yaml`, and the workflow, with the message `feat: add first evaluation results and eval CI workflow`. Report the summary table to the owner.
