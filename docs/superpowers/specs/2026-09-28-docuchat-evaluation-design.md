# Docuchat — Sub-project 2: Evaluation

**Date:** 2026-09-28
**Status:** Design approved, pending spec review
**Scope:** Sub-project 2 of 4 (see the table in `2026-09-27-docuchat-packaging-design.md`)

---

## Purpose

Produce a measured, reproducible answer to "does each pipeline stage help?" for
docuchat, and make that answer something a reader of the repo can rerun.
Sub-project 4 lifts the results table into the README, so the output of this
sub-project is numbers someone can check, not a claim.

Sub-project 1 built the seam: `ask()` takes a `Settings`, and the ablation arms
are `dataclasses.replace()` calls. This sub-project supplies what the seam was
built for: a corpus, ground truth, metrics, a judge, and a runner.

### Success criteria

1. A public-domain corpus of roughly 10–20 mortgage PDFs is committed under
   `eval/corpus/`, including at least one scanned PDF.
2. About 40 ground-truth questions are committed in `eval/questions.yaml`,
   each labelled with evidence pages and verified by the repo owner.
3. `python -m docuchat.evaluate run --arms baseline,+hybrid,+rerank` reproduces the retrieval metrics
   for the LLM-free arms with no API key and no Docling models.
4. A full judged run produces `eval/results/summary.md`: one row per arm, with
   bootstrap confidence intervals and per-question win/loss counts. The judge's
   agreement rate with a hand-graded sample is reported alongside.
5. CI runs the LLM-free arms on every change to `src/` or `eval/` and publishes
   the summary.

---

## Decisions made during design

| Decision | Choice | Rationale |
|---|---|---|
| Corpus | Public-domain CFPB/HUD/Fannie Mae PDFs, committed | The single 6-page fixture is too small for the arms to differ: BM25 goes flat, and almost every chunk fits in top-k. Government URLs rot, so committing the files beats fetching them. |
| Ground truth | Drafted by the implementer from the corpus, verified by the owner | The notebook's 8 questions have generic answers ("should be stated in the promissory note") rather than facts from a document, so they cannot be scored. LLM-synthesized questions would make the eval circular. |
| Size | ~40 questions | At 20 questions each one moves a metric by 5 points, so arm differences would mostly be noise. |
| Metrics approach | Own metrics, RAGAS as an optional cross-check | Retrieval metrics on page labels are deterministic and free, so they run in CI. An in-repo judge is short enough to audit. RAGAS stays behind an extra, so its API churn and langchain stack never become core dependencies. |
| Judge model | `judge_model` setting, default `claude-opus-5-5` | Amends sub-project 1's `claude-opus-5`, keeping its rationale that a judge should be at least as capable as what it grades. |
| Ingestion in the eval | Committed node snapshot | Arms and CI start from chunked nodes, so they need neither Docling models nor LLM page classification. Rerunning `ingest` regenerates the snapshot. |
| CI gate | Report only, no threshold | Embedding float differences across runner hardware can reorder near-ties. A gate that fires on noise gets disabled. |

### Approaches considered and rejected

- **RAGAS-first.** RAGAS computes everything, including LLM-based context
  precision and recall. Retrieval numbers would cost API calls and be
  non-deterministic, CI could not run them, and RAGAS would be the heaviest
  dependency in the repo.
- **Own metrics, no RAGAS.** The leanest option, but it drops a line item from
  the sub-project table for little saving, since the extra is optional anyway.
- **Fixture-only corpus.** Cheapest, but it would show the harness working
  rather than produce any finding.

---

## Layout

```
eval/
  corpus/              # committed PDFs
    SOURCES.md         # per file: origin URL, sha256, document type, scanned?
  questions.yaml       # ground truth
  nodes/
    mortgage.jsonl     # committed ingestion snapshot, mortgage profile
    generic.jsonl      # same corpus, generic profile
  results/
    <arm>.json         # raw per-question outputs of the latest run
    summary.md         # the results table
src/docuchat/
  evaluate.py          # arms, snapshot I/O, runner, retrieval metrics, report, CLI
  judge.py             # judge prompt, verdict parsing
```

`evaluate.py` and `judge.py` join the import-purity test: importing them must not
load torch, docling, or sentence-transformers.

### Ground-truth format

```yaml
- id: cd-loan-amount
  question: What is the loan amount on the Closing Disclosure?
  reference_answer: $162,000.
  evidence:
    - {filename: cfpb_closing_disclosure.pdf, page: 1}
  kind: fact            # fact | table | multi_hop | unanswerable
  notes: ""
```

The target mix is about 15 `fact`, 10 `table`, 10 `multi_hop` (spanning pages
or documents, to exercise decomposition), and 5 `unanswerable` (to exercise
refusal). Unanswerable questions have an empty `evidence` list.

### Snapshot format

The first line of each JSONL file is a header holding the sha256 of every
corpus file plus the ingestion and chunking settings used. Each following line is
one node: `{"text": ..., "metadata": {...}}`. On load, `run` recomputes the
corpus hashes and refuses a mismatched snapshot with a message naming the
`ingest` command.

---

## Data flow

**`python -m docuchat.evaluate ingest [--profile mortgage|generic]`**
Docling → per-document classification → chunking → write
`eval/nodes/<profile>.jsonl`. This needs Docling models and, when LLM
classification is on, an API key. It runs only when the corpus or the
ingestion code changes.

**`python -m docuchat.evaluate run [--arms ...] [--judge] [--ragas]`**
For each arm: load that arm's snapshot → build a `Store` from the nodes → for
each question, call `ask()` → score retrieval → optionally judge → write
`results/<arm>.json`. After all arms: write `summary.md`.

`build_store` is split so a `Store` can be built from existing nodes:
`store_from_nodes(nodes, cfg)` does the embedding and BM25 build, and
`build_store` becomes chunking followed by `store_from_nodes`. The runner uses
the new function; nothing else changes.

### Arms

| Arm | Snapshot | Changes from `Settings()` | Needs LLM |
|---|---|---|---|
| baseline | mortgage | `retrieval_mode=vector`, rewrite/decompose/rerank off | only with `--judge` |
| +hybrid | mortgage | `retrieval_mode=hybrid`, rewrite/decompose/rerank off | only with `--judge` |
| +rerank | mortgage | hybrid + `use_rerank=True` | only with `--judge` |
| full | mortgage | defaults | yes |
| full/generic | generic | `domain_profile=generic` | yes |
| full/embed-bge-base | mortgage | `embed_model_name=BAAI/bge-base-en-v1.5` | yes |

The generic arm gets its own snapshot because the profile shapes ingestion
(doc-type labels, section titles) as well as prompts. The embedding arm reuses
the mortgage snapshot and only re-embeds, so it measures the embedding model and
nothing else, even though chunking also uses an encoder.

Judging is opt-in (`--judge`). Without it, the first three arms run with a
stub answer LLM: `ask()` still makes its single answer call, but the stub returns
an empty string, and the retrieval metrics don't depend on the answer. So those
arms need no API key. The last three arms need a real LLM for rewrite and
decomposition. Without an API key they are skipped, and the summary says so.

`max_context_tokens` is not an arm yet. Adding one is a single row.

---

## Metrics

### Retrieval (deterministic)

Computed on answerable questions only. A match is `(filename, page_number)`.

- **Context hit:** at least one evidence page appears in the final context the
  LLM sees.
- **Context recall:** the fraction of evidence pages covered by the final
  context. This mainly matters for `multi_hop`.
- **MRR:** reciprocal rank of the first evidence page in the final context.
- **Candidate recall:** context recall over the pre-rerank candidate pool.
  Comparing it with context recall separates "retrieval never found it" from
  "rerank dropped it."

Candidate recall needs one additive change to `ask()`: record
`debug["candidates"]`, the `(filename, page_number)` list of the deduplicated
pool before rerank or truncation. The return shape is otherwise unchanged.

The final context is capped at `rerank_top_n` in every arm (ruling C21), so
these metrics are comparable across arms. `confidence` is excluded from all
comparisons: with rerank off it is a mean of retrieval scores, not cross-encoder
scores.

### Generation (judge)

`judge.py` sends `judge_model` (temperature 0) the question, the reference
answer, the retrieved context, and the answer, and asks for:

```json
{"refused": false, "correctness": 2, "faithful": true, "rationale": "..."}
```

`correctness` is 0 (wrong), 1 (partial), or 2 (correct). The judge decides
refusals; the notebook's phrase list is dropped as brittle. Malformed output gets
one retry, then the question is recorded as `judge_error`. It is counted and
reported, never scored.

Reported per arm:

- **Correct rate** (share scoring 2) and **mean correctness**
- **Faithfulness rate**
- **Correct-refusal rate** on `unanswerable`
- **False-refusal rate** on answerable questions

**Judge validation.** Before results are reported, the owner hand-grades about 10
verdicts sampled across arms and kinds. The agreement rate goes into
`summary.md`. If agreement is poor, the judge prompt is revised before any
numbers are published.

### RAGAS (optional)

`--ragas` implies `--judge` and requires the `eval-ragas` extra (`ragas==0.4.3`, pinned when
implemented). It adds faithfulness and answer-relevancy columns, using the same
judge model. It is a cross-check, not the headline.

### Reporting

`summary.md` has one row per arm with every metric above, the mean number of LLM
calls per query, and the error counts. Each arm-level rate carries a paired
bootstrap 95% CI over questions (numpy, 1000 resamples, fixed seed). Each arm
also gets a win/loss/tie count against the arm before it, per question, on
context recall and on correctness. Wall-clock latency is recorded in the raw
JSON but kept out of the headline, for the same reason sub-project 1 gave.

---

## Carry-forward fixes

These land before the first baseline, so they don't confound the arms.

1. **Per-document classification.** The heuristic currently runs per page and
   takes the first keyword match, so a Closing Disclosure page that mentions
   escrow is labelled "Escrow Statement", and that label reaches the prompt
   headers. Fix: classify each document once, using the heuristic on page 1 with
   the LLM as fallback, then apply that label to every page of the document. A
   test pins "Closing Disclosure" on all six pages of the fixture.
2. **One Docling converter per `load_directory`**, not one per PDF.
3. **CPU torch in CI**, installed from the PyTorch CPU index.

BM25 scoring at or below zero on tiny corpora needs no change: the real corpus
resolves it.

---

## Error handling

- An exception on one question (API error, empty retrieval, and so on) is
  recorded in that question's result as `error`. The arm continues, and
  `summary.md` reports error counts per arm.
- A stale snapshot stops the run with a message naming `ingest` (see Snapshot
  format).
- A missing API key with `--judge` stops the run before any question executes,
  naming the variable. Without `--judge`, an arm that needs an LLM for rewrite
  or decomposition is skipped instead, and `summary.md` lists it with the
  reason.
- There is no resume. An arm is cheap to rerun.

---

## Testing

Unit tests run in the default suite and reuse the fakes in `tests/conftest.py`.

- **Metrics:** hand-built cases covering partial multi-hop recall, exclusion of
  unanswerable questions, rank order for MRR, and bootstrap determinism under a
  fixed seed.
- **Judge parsing** with a fake LLM: valid JSON; malformed then valid; malformed
  twice gives `judge_error`.
- **Ground-truth validation:** every `evidence` filename exists in
  `eval/corpus/`, every page is within that PDF's page count, `unanswerable`
  entries have no evidence, answerable entries have some, and ids are unique.
  This catches typos in the hand-verified set.
- **Snapshot round-trip** and stale-snapshot rejection.
- **Runner smoke test:** a fake store and fake LLM produce well-formed
  per-question results and a summary.
- **Per-document classification** on the fixture pages.
- **Import purity** extended to `docuchat.evaluate` and `docuchat.judge`.

The real run stays `@pytest.mark.integration`.

---

## CI

A second workflow, `.github/workflows/eval.yml`, triggers on changes to `src/**`
or `eval/**` and on manual dispatch. It installs CPU torch, restores a cached
Hugging Face model directory, runs
`python -m docuchat.evaluate run --arms baseline,+hybrid,+rerank`,
writes `summary.md` to the job summary, and uploads it as an artifact. It never
fails on metric values.

---

## Order of work

1. Carry-forward fixes.
2. Harness and tests: `store_from_nodes`, `debug["candidates"]`, snapshot I/O,
   metrics, judge, runner, report, CLI.
3. Corpus assembly and `SOURCES.md`.
4. Draft `questions.yaml`; **the owner verifies it** before any numbers are run.
5. Generate both snapshots.
6. First full judged run, then judge validation with the owner.
7. Commit the results and the eval CI workflow.

---

## Out of scope

- Chunking-parameter arms (they would need a snapshot per setting).
- A `max_context_tokens` arm (a one-row addition once the harness exists).
- Latency benchmarking (sub-project 3).
- The local front end (deferred until after this sub-project).
- README results tables (sub-project 4 lifts `summary.md`).

## Risks

- **Judge bias.** An Opus judge grading Sonnet answers may be lenient on
  fluent but wrong answers. The hand-graded agreement check exists for this.
- **Small sample.** Forty questions give wide intervals. The CIs and win/loss
  counts are there to keep the README honest about it.
- **Corpus availability.** If too few filled-in public samples exist, the corpus
  leans on explanatory documents (CFPB guides) rather than forms. The question
  mix may shift from `table` toward `fact`, and this spec's numbers would move
  with it.
