"""Evaluation metrics and aggregation for docuchat runs.

A "match" is the pair (filename, page_number); page numbers are 1-based
physical PDF page indexes. Functions here are pure and read only plain
dicts/lists so they have no heavy imports.
"""

import argparse
import dataclasses
import hashlib
import json
import random
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import yaml
from llama_index.core.schema import TextNode

from docuchat.classify import classify_pages
from docuchat.config import Settings
from docuchat.index import build_nodes, store_from_nodes
from docuchat.ingest import load_directory
from docuchat.judge import UNANSWERABLE_REFERENCE, judge_answer
from docuchat.models import get_encoder, get_judge_llm, get_llm
from docuchat.pipeline import ask

# --- Metrics ---------------------------------------------------------------


def retrieval_metrics(
    evidence: list[tuple[str, int]],
    context: list[tuple[str, int]],
    candidates: list[tuple[str, int]],
) -> dict:
    """Score retrieved `context`/`candidates` against ground-truth `evidence`.

    - hit: 1.0 if any evidence page is present in context, else 0.0.
    - recall: distinct evidence pages present in context / distinct evidence pages.
    - mrr: 1/rank (1-based) of the first context entry that is an evidence page, else 0.0.
    - candidate_recall: distinct evidence pages present in candidates / distinct evidence pages.
    """
    evidence_set = set(evidence)
    context_set = set(context)
    candidate_set = set(candidates)

    hit = 1.0 if evidence_set & context_set else 0.0

    n_evidence = len(evidence_set)
    recall = len(evidence_set & context_set) / n_evidence if n_evidence else 0.0
    candidate_recall = len(evidence_set & candidate_set) / n_evidence if n_evidence else 0.0

    mrr = 0.0
    for rank, page in enumerate(context, start=1):
        if page in evidence_set:
            mrr = 1.0 / rank
            break

    return {"hit": hit, "recall": recall, "mrr": mrr, "candidate_recall": candidate_recall}


def bootstrap_ci(values: list[float], n: int = 1000, seed: int = 0) -> tuple[float, float] | None:
    """95% percentile bootstrap CI over the mean of `values`, or None if empty."""
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    resample_means = rng.choice(arr, size=(n, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(resample_means, [2.5, 97.5])
    return (float(lo), float(hi))


def win_loss(prev: dict[str, float], cur: dict[str, float]) -> tuple[int, int, int]:
    """(wins, losses, ties) for `cur` vs `prev`, over ids present in both."""
    wins = losses = ties = 0
    for qid in prev.keys() & cur.keys():
        if cur[qid] > prev[qid]:
            wins += 1
        elif cur[qid] < prev[qid]:
            losses += 1
        else:
            ties += 1
    return (wins, losses, ties)


# --- Aggregation -------------------------------------------------------------


def _stat(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "ci": None, "n": 0}
    return {"mean": float(np.mean(values)), "ci": bootstrap_ci(values), "n": len(values)}


def aggregate(records: list[dict]) -> dict[str, dict]:
    """Aggregate per-question records into metric name -> {mean, ci, n}."""
    non_errored = [r for r in records if "error" not in r]
    answerable_ok = [r for r in non_errored if r["kind"] != "unanswerable"]

    retrieval_stats = {
        name: _stat([r["retrieval"][name] for r in answerable_ok])
        for name in ("hit", "recall", "mrr", "candidate_recall")
    }

    valid_verdict_records = [
        r for r in non_errored if "verdict" in r and "judge_error" not in r["verdict"]
    ]
    judge_error_records = [
        r for r in non_errored if "verdict" in r and "judge_error" in r["verdict"]
    ]

    correctness_stats = {
        "correct_rate": _stat([1.0 if r["verdict"]["correctness"] == 2 else 0.0 for r in valid_verdict_records]),
        "mean_correctness": _stat([float(r["verdict"]["correctness"]) for r in valid_verdict_records]),
        "faithful_rate": _stat([1.0 if r["verdict"]["faithful"] else 0.0 for r in valid_verdict_records]),
    }

    unanswerable_valid = [r for r in valid_verdict_records if r["kind"] == "unanswerable"]
    answerable_valid = [r for r in valid_verdict_records if r["kind"] != "unanswerable"]

    refusal_stats = {
        "correct_refusal_rate": _stat([1.0 if r["verdict"]["refused"] else 0.0 for r in unanswerable_valid]),
        "false_refusal_rate": _stat([1.0 if r["verdict"]["refused"] else 0.0 for r in answerable_valid]),
    }

    llm_calls_stat = _stat([float(r["llm_calls"]) for r in non_errored])

    counts = {
        "errors": {"mean": len(records) - len(non_errored), "ci": None, "n": len(records) - len(non_errored)},
        "judge_errors": {"mean": len(judge_error_records), "ci": None, "n": len(judge_error_records)},
    }

    # Optional: only present when records carry RAGAS cross-check scores
    # (added by `run --ragas`); see render_summary.
    ragas_stats = {
        name: _stat([r[name] for r in non_errored if name in r])
        for name in ("ragas_faithfulness", "ragas_answer_relevancy")
        if any(name in r for r in non_errored)
    }

    return {
        **retrieval_stats,
        **correctness_stats,
        **refusal_stats,
        "llm_calls": llm_calls_stat,
        **counts,
        **ragas_stats,
    }


# --- Ground truth, validation, and node snapshots ---------------------------

EVAL_DIR = Path("eval")
CORPUS_DIR = EVAL_DIR / "corpus"
QUESTIONS_PATH = EVAL_DIR / "questions.yaml"
NODES_DIR = EVAL_DIR / "nodes"
RESULTS_DIR = EVAL_DIR / "results"

_REQUIRED_QUESTION_KEYS = ("id", "question", "reference_answer", "evidence", "kind")
_VALID_KINDS = {"fact", "table", "multi_hop", "unanswerable"}

# Every Settings field under Ingestion, Chunking, Domain, and Classification,
# excluding embed_model_name: chunking always uses the snapshot's encoder,
# the embed arm only re-embeds.
INGEST_FIELDS: tuple[str, ...] = (
    "min_text_length",
    "do_ocr",
    "do_table_structure",
    "chunk_max_tokens",
    "chunk_min_tokens",
    "merge_threshold",
    "merge_max_tokens",
    "min_chunk_words",
    "domain_profile",
    "use_llm_classification",
    "classify_snippet_chars",
)


def load_questions(path: Path) -> list[dict]:
    """Load the ground-truth question list from a YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def validate_questions(questions: list[dict], corpus_dir: Path) -> list[str]:
    """Validate ground-truth questions against the corpus. Returns a list of
    human-readable error strings, empty if the question list is valid."""
    import pypdfium2

    errors: list[str] = []
    seen_ids: set = set()
    page_counts: dict[str, int] = {}

    for q in questions:
        qid = q.get("id", "<missing id>")

        missing = [k for k in _REQUIRED_QUESTION_KEYS if k not in q]
        if missing:
            errors.append(f"question {qid!r}: missing key(s) {missing}")
            continue

        if q["kind"] not in _VALID_KINDS:
            errors.append(f"question {qid!r}: kind {q['kind']!r} not in {sorted(_VALID_KINDS)}")
            continue

        if q["id"] in seen_ids:
            errors.append(f"question {qid!r}: duplicate id")
            continue
        seen_ids.add(q["id"])

        if q["kind"] == "unanswerable":
            if q["evidence"]:
                errors.append(f"question {qid!r}: unanswerable question must have empty evidence")
            continue

        if not q["evidence"]:
            errors.append(f"question {qid!r}: answerable question must have non-empty evidence")
            continue

        bad = False
        for ev in q["evidence"]:
            filename = ev["filename"]
            pdf_path = corpus_dir / filename
            if not pdf_path.exists():
                errors.append(f"question {qid!r}: evidence filename {filename!r} not found in corpus")
                bad = True
                continue

            if filename not in page_counts:
                doc = pypdfium2.PdfDocument(pdf_path)
                page_counts[filename] = len(doc)
            page_count = page_counts[filename]

            if not (1 <= ev["page"] <= page_count):
                errors.append(
                    f"question {qid!r}: evidence page {ev['page']} out of range "
                    f"1..{page_count} for {filename!r}"
                )
                bad = True
        if bad:
            continue

    return errors


def corpus_hashes(corpus_dir: Path) -> dict[str, str]:
    """filename -> sha256 hex digest, for every *.pdf (case-insensitive) in
    corpus_dir, sorted by filename."""
    hashes = {}
    for path in sorted(corpus_dir.iterdir(), key=lambda p: p.name):
        if path.suffix.lower() == ".pdf":
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(hashes.items()))


def write_snapshot(path: Path, nodes: list[TextNode], corpus_dir: Path, cfg: Settings) -> None:
    """Write `nodes` to a JSONL snapshot at `path`, prefixed with a header
    line recording the corpus hashes and ingest-relevant settings."""
    header = {
        "corpus": corpus_hashes(corpus_dir),
        "settings": {f: getattr(cfg, f) for f in INGEST_FIELDS},
    }
    with open(path, "w") as f:
        f.write(json.dumps(header) + "\n")
        for node in nodes:
            f.write(json.dumps({"text": node.get_content(), "metadata": node.metadata}) + "\n")


class StaleSnapshotError(Exception):
    """Raised when a node snapshot's corpus or ingest settings no longer
    match the current corpus/config."""


def load_snapshot(path: Path, corpus_dir: Path, cfg: Settings) -> list[TextNode]:
    """Load nodes from a JSONL snapshot, raising StaleSnapshotError if the
    snapshot's corpus hashes or ingest settings differ from the current
    corpus/cfg."""
    with open(path) as f:
        lines = f.readlines()

    header = json.loads(lines[0])
    current_corpus = corpus_hashes(corpus_dir)
    current_settings = {f: getattr(cfg, f) for f in INGEST_FIELDS}

    stale_reasons = []

    changed_files = sorted(
        name
        for name in header["corpus"].keys() | current_corpus.keys()
        if header["corpus"].get(name) != current_corpus.get(name)
    )
    if changed_files:
        stale_reasons.append(f"corpus file(s) changed: {changed_files}")

    changed_fields = sorted(
        f for f in INGEST_FIELDS if header["settings"].get(f) != current_settings[f]
    )
    if changed_fields:
        stale_reasons.append(f"setting(s) changed: {changed_fields}")

    if stale_reasons:
        raise StaleSnapshotError(
            "; ".join(stale_reasons)
            + ". Re-run `python -m docuchat.evaluate ingest --profile <profile>`."
        )

    nodes = []
    for line in lines[1:]:
        record = json.loads(line)
        nodes.append(TextNode(text=record["text"], metadata=record["metadata"]))
    return nodes


def unreachable_evidence(questions: list[dict], nodes: list[TextNode]) -> list[str]:
    """Ids of answerable questions for which at least one evidence
    (filename, page) pair has no matching node."""
    available = {(n.metadata["filename"], n.metadata["page_number"]) for n in nodes}
    result = []
    for q in questions:
        if q["kind"] == "unanswerable":
            continue
        if any((ev["filename"], ev["page"]) not in available for ev in q["evidence"]):
            result.append(q["id"])
    return result


# --- Arms --------------------------------------------------------------------

# name -> (snapshot profile, Settings overrides). Order is significant: it is
# the "vs previous arm" comparison order in render_summary.
ARMS: dict[str, tuple[str, dict]] = {
    "baseline": (
        "mortgage",
        {"retrieval_mode": "vector", "use_rewrite": False, "use_decomposition": False, "use_rerank": False},
    ),
    "+hybrid": (
        "mortgage",
        {"retrieval_mode": "hybrid", "use_rewrite": False, "use_decomposition": False, "use_rerank": False},
    ),
    "+rerank": (
        "mortgage",
        {"retrieval_mode": "hybrid", "use_rewrite": False, "use_decomposition": False, "use_rerank": True},
    ),
    "full": ("mortgage", {}),
    "full/generic": ("generic", {"domain_profile": "generic"}),
    "full/embed-bge-base": ("mortgage", {"embed_model_name": "BAAI/bge-base-en-v1.5"}),
}

# name -> the arm it's compared against in render_summary's "vs" table. An
# arm run without its base also having been run gets no comparison row.
COMPARE_TO: dict[str, str] = {
    "+hybrid": "baseline",
    "+rerank": "+hybrid",
    "full": "+rerank",
    "full/generic": "full",
    "full/embed-bge-base": "full",
}


def arm_settings(name: str, base: Settings) -> Settings:
    """Apply arm `name`'s Settings overrides onto `base`."""
    _, overrides = ARMS[name]
    return dataclasses.replace(base, **overrides)


def needs_llm(cfg: Settings) -> bool:
    """Whether the pipeline itself calls the LLM before generating the
    answer (rewrite or decomposition), independent of judging."""
    return cfg.use_rewrite or cfg.use_decomposition


class StubLLM:
    """No-op LLM: used as the answer LLM when not judging and the arm
    doesn't need a real completion for rewrite/decompose."""

    def complete(self, prompt: str) -> str:
        return ""


class _CountingLLM:
    """Wraps an LLM and counts `complete` calls, so run_arm can report the
    per-question llm_calls the answer pipeline actually made."""

    def __init__(self, llm) -> None:
        self._llm = llm
        self.calls = 0

    def complete(self, prompt: str):
        self.calls += 1
        return self._llm.complete(prompt)


# --- Runner --------------------------------------------------------------------


def run_arm(
    name: str,
    questions: list[dict],
    store,
    cfg: Settings,
    llm,
    judge_llm=None,
) -> list[dict]:
    """Run every question in `questions` through ask() under arm `cfg`,
    scoring retrieval (and, with `judge_llm`, correctness) per question.

    An exception from ask() or the judge is caught and recorded as
    `record["error"]` in place of every key after "question"; the next
    question still runs. A judge parsing/API failure does not raise --
    judge_answer already returns a judge_error verdict for that.
    """
    records = []
    for q in questions:
        record = {"id": q["id"], "kind": q["kind"], "question": q["question"]}
        try:
            counting_llm = _CountingLLM(llm)
            start = time.monotonic()
            result = ask(q["question"], store, cfg, llm=counting_llm)
            latency_s = time.monotonic() - start

            debug = result["debug"]
            candidates = [(f, p) for f, p in debug["candidates"]]
            context_pages = [(s["filename"], s["page_number"]) for s in result["sources"]]
            evidence = [(ev["filename"], ev["page"]) for ev in q["evidence"]]

            record.update(
                {
                    "answer": result["answer"],
                    "contexts": debug["contexts"],
                    "sources": [[f, p] for f, p in context_pages],
                    "candidates": [[f, p] for f, p in candidates],
                    "retrieval": retrieval_metrics(evidence, context_pages, candidates),
                    "llm_calls": counting_llm.calls,
                    "latency_s": latency_s,
                }
            )

            if judge_llm is not None:
                reference = (
                    UNANSWERABLE_REFERENCE if q["kind"] == "unanswerable" else q["reference_answer"]
                )
                record["verdict"] = judge_answer(
                    q["question"], reference, debug["contexts"], result["answer"], judge_llm
                )
        except Exception as exc:  # noqa: BLE001 -- recorded, not raised, so the run continues
            record["error"] = str(exc)
        records.append(record)
    return records


# --- RAGAS cross-check -------------------------------------------------------


def ragas_scores(records: list[dict], cfg: Settings) -> dict[str, float]:
    """Mean RAGAS faithfulness and answer-relevancy over `records` (each
    needs "question", "answer", "contexts"; records with "error" are
    skipped).

    Uses ragas 0.4.3's `ragas.evaluate` with the `ragas.metrics.faithfulness`
    and `ragas.metrics.answer_relevancy` metric objects,
    `ragas.llms.LlamaIndexLLMWrapper` around `models.get_judge_llm(cfg)`
    (`cfg.judge_model`, Anthropic), and
    `ragas.embeddings.LlamaIndexEmbeddingsWrapper` around
    `models.get_embed_model(cfg)` (`cfg.embed_model_name`, a HuggingFace/
    sentence-transformers embedding). `ragas` is imported here, not at
    module scope, so importing this module never pulls it in (see
    tests/test_import_purity.py).
    """
    import numpy as np
    from ragas import evaluate as ragas_evaluate
    from ragas.dataset_schema import EvaluationDataset
    from ragas.embeddings import LlamaIndexEmbeddingsWrapper
    from ragas.llms import LlamaIndexLLMWrapper
    from ragas.metrics import answer_relevancy, faithfulness

    from docuchat.models import get_embed_model, get_judge_llm

    rows = [
        {"user_input": r["question"], "response": r["answer"], "retrieved_contexts": r["contexts"]}
        for r in records
        if "error" not in r
    ]
    if not rows:
        return {"ragas_faithfulness": float("nan"), "ragas_answer_relevancy": float("nan")}

    dataset = EvaluationDataset.from_list(rows)
    llm = LlamaIndexLLMWrapper(get_judge_llm(cfg))
    embeddings = LlamaIndexEmbeddingsWrapper(get_embed_model(cfg))

    result = ragas_evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
        show_progress=False,
    )
    return {
        "ragas_faithfulness": float(np.nanmean(result["faithfulness"])),
        "ragas_answer_relevancy": float(np.nanmean(result["answer_relevancy"])),
    }


# --- Summary report ------------------------------------------------------------

_METRIC_NAMES = (
    "hit",
    "recall",
    "mrr",
    "candidate_recall",
    "correct_rate",
    "mean_correctness",
    "faithful_rate",
    "correct_refusal_rate",
    "false_refusal_rate",
    "llm_calls",
    "errors",
    "judge_errors",
)


def _fmt_stat(stat: dict) -> str:
    mean = stat.get("mean")
    if mean is None:
        return "n/a"
    ci = stat.get("ci")
    if ci is None:
        return f"{mean:.3f}"
    lo, hi = ci
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _recall_by_id(records: list[dict]) -> dict[str, float]:
    # Matches aggregate()'s retrieval stats, which exclude unanswerable
    # questions (they have no evidence, so recall is vacuous for them).
    return {
        r["id"]: r["retrieval"]["recall"]
        for r in records
        if "error" not in r and r["kind"] != "unanswerable"
    }


def _correctness_by_id(records: list[dict]) -> dict[str, float]:
    return {
        r["id"]: r["verdict"]["correctness"]
        for r in records
        if "error" not in r and "verdict" in r and "judge_error" not in r["verdict"]
    }


def render_summary(
    arm_records: dict[str, list[dict]],
    skipped: dict[str, str],
    unreachable: list[str],
    validation: dict | None,
) -> str:
    """Render the Markdown eval summary: per-arm aggregate metrics, a
    win/loss/tie table against each arm's COMPARE_TO base (when that base
    was also run), and skipped-arm / unreachable-evidence /
    judge-validation notes."""
    arm_names = [name for name in ARMS if name in arm_records]
    judged = any(any("verdict" in r for r in records) for records in arm_records.values())
    has_ragas = any(
        any("ragas_faithfulness" in r for r in records) for records in arm_records.values()
    )
    metric_names = _METRIC_NAMES + (("ragas_faithfulness", "ragas_answer_relevancy") if has_ragas else ())
    n_questions = len(next(iter(arm_records.values()))) if arm_records else 0

    lines = ["# Evaluation Summary", ""]
    lines.append(f"- Date: {date.today().isoformat()}")
    lines.append(f"- Commit: {_git_commit()}")
    lines.append(f"- Questions: {n_questions}")
    if judged:
        lines.append(f"- Judge model: {Settings.from_env().judge_model}")
    lines.append("")

    lines.append("## Arms")
    lines.append("")
    lines.append("| arm | " + " | ".join(metric_names) + " |")
    lines.append("|---|" + "---|" * len(metric_names))
    for name in arm_names:
        agg = aggregate(arm_records[name])
        row = " | ".join(_fmt_stat(agg.get(m, {"mean": None})) for m in metric_names)
        lines.append(f"| {name} | {row} |")
    lines.append("")

    lines.append("## vs previous arm")
    lines.append("")
    lines.append("| arm | metric | wins | losses | ties |")
    lines.append("|---|---|---|---|---|")
    for cur_name in arm_names:
        prev_name = COMPARE_TO.get(cur_name)
        if prev_name is None or prev_name not in arm_records:
            continue
        prev, cur = arm_records[prev_name], arm_records[cur_name]
        wins, losses, ties = win_loss(_recall_by_id(prev), _recall_by_id(cur))
        lines.append(f"| {cur_name} vs {prev_name} | recall | {wins} | {losses} | {ties} |")
        if judged:
            wins, losses, ties = win_loss(_correctness_by_id(prev), _correctness_by_id(cur))
            lines.append(f"| {cur_name} vs {prev_name} | correctness | {wins} | {losses} | {ties} |")
    lines.append("")

    lines.append("## Skipped arms")
    lines.append("")
    if skipped:
        for name, reason in skipped.items():
            lines.append(f"- {name}: {reason}")
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("## Unreachable evidence")
    lines.append("")
    lines.append(", ".join(unreachable) if unreachable else "(none)")
    lines.append("")

    if validation is not None:
        lines.append("## Judge validation")
        lines.append("")
        lines.append(f"Agreement: {validation['agreement']:.3f} over n={validation['n']}")
        lines.append("")

    return "\n".join(lines)


def judge_agreement(items: list[dict]) -> dict | None:
    """Share of judge-vs-human agreement on all three verdict fields, over
    items whose `human` fields are fully filled in. None if none are."""
    filled = [
        it for it in items if all(it["human"][k] is not None for k in ("correctness", "faithful", "refused"))
    ]
    if not filled:
        return None
    matches = sum(
        1
        for it in filled
        if all(it["human"][k] == it["judge"][k] for k in ("correctness", "faithful", "refused"))
    )
    return {"agreement": matches / len(filled), "n": len(filled)}


# --- CLI -------------------------------------------------------------------


def _cmd_ingest(args: argparse.Namespace) -> int:
    cfg = Settings.from_env(domain_profile=args.profile)
    try:
        llm = get_llm(cfg)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    pages = load_directory(CORPUS_DIR, cfg)
    pages = classify_pages(pages, llm, cfg)
    nodes = build_nodes(pages, get_encoder(cfg), cfg)

    NODES_DIR.mkdir(parents=True, exist_ok=True)
    write_snapshot(NODES_DIR / f"{args.profile}.jsonl", nodes, CORPUS_DIR, cfg)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    arm_names = args.arms.split(",") if args.arms else list(ARMS.keys())
    invalid = [n for n in arm_names if n not in ARMS]
    if invalid:
        print(f"unknown arm(s): {invalid}; valid arms: {list(ARMS.keys())}", file=sys.stderr)
        return 2

    if args.ragas:
        try:
            import ragas  # noqa: F401
        except ImportError:
            raise SystemExit("ragas is not installed; run `pip install -e .[eval-ragas]`")
        args.judge = True

    base_cfg = Settings.from_env()

    judge_llm = None
    if args.judge:
        try:
            judge_llm = get_judge_llm(base_cfg)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    questions = load_questions(QUESTIONS_PATH)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    skipped: dict[str, str] = {}
    arm_records: dict[str, list[dict]] = {}
    unreachable: set[str] = set()

    for name in arm_names:
        snapshot_profile, _overrides = ARMS[name]
        cfg = arm_settings(name, base_cfg)

        if args.judge or needs_llm(cfg):
            try:
                answer_llm = get_llm(cfg)
            except RuntimeError as exc:
                skipped[name] = str(exc)
                continue
        else:
            answer_llm = StubLLM()

        snapshot_path = NODES_DIR / f"{snapshot_profile}.jsonl"
        try:
            nodes = load_snapshot(snapshot_path, CORPUS_DIR, cfg)
        except StaleSnapshotError as exc:
            print(f"{name}: {exc}", file=sys.stderr)
            return 2

        unreachable.update(unreachable_evidence(questions, nodes))
        store = store_from_nodes(nodes, cfg)

        records = run_arm(name, questions, store, cfg, answer_llm, judge_llm=judge_llm)
        arm_records[name] = records

        arm_ragas = None
        if args.ragas:
            arm_ragas = ragas_scores(records, cfg)
            for r in records:
                if "error" not in r:
                    r.update(arm_ragas)

        with open(RESULTS_DIR / f"{name.replace('/', '_')}.json", "w") as f:
            json.dump({"records": records, "ragas": arm_ragas}, f, indent=2)

    validation = None
    validation_path = RESULTS_DIR / "judge_validation.yaml"
    if validation_path.exists():
        with open(validation_path) as f:
            items = yaml.safe_load(f) or []
        validation = judge_agreement(items)

    summary = render_summary(arm_records, skipped, sorted(unreachable), validation)
    (RESULTS_DIR / "summary.md").write_text(summary)
    print(summary)
    return 0


def _cmd_judge_sample(args: argparse.Namespace) -> int:
    validation_path = RESULTS_DIR / "judge_validation.yaml"
    if validation_path.exists() and not args.force:
        print(
            f"{validation_path} already exists and may hold hand-entered human labels; "
            "pass --force to overwrite it.",
            file=sys.stderr,
        )
        return 2

    questions_by_id = {q["id"]: q for q in load_questions(QUESTIONS_PATH)}

    groups: dict[tuple[str, str], list[dict]] = {}
    for name in ARMS:
        path = RESULTS_DIR / f"{name.replace('/', '_')}.json"
        if not path.exists():
            continue
        with open(path) as f:
            records = json.load(f)["records"]
        for r in records:
            if "verdict" in r and "judge_error" not in r["verdict"]:
                groups.setdefault((name, r["kind"]), []).append(r)

    rng = random.Random(0)
    for group in groups.values():
        rng.shuffle(group)

    keys = list(groups.keys())
    picked: list[tuple[str, dict]] = []
    i = 0
    while len(picked) < args.n and any(groups[k] for k in keys):
        key = keys[i % len(keys)]
        if groups[key]:
            picked.append((key[0], groups[key].pop()))
        i += 1

    items = []
    for arm_name, r in picked:
        reference_answer = questions_by_id.get(r["id"], {}).get("reference_answer", "")
        items.append(
            {
                "arm": arm_name,
                "id": r["id"],
                "question": r["question"],
                "reference_answer": reference_answer,
                "answer": r["answer"],
                "judge": {
                    "correctness": r["verdict"]["correctness"],
                    "faithful": r["verdict"]["faithful"],
                    "refused": r["verdict"]["refused"],
                },
                "human": {"correctness": None, "faithful": None, "refused": None},
            }
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(validation_path, "w") as f:
        yaml.safe_dump(items, f, sort_keys=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m docuchat.evaluate")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--profile", choices=["mortgage", "generic"], required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--arms", default=None)
    p_run.add_argument("--judge", action="store_true")
    p_run.add_argument("--ragas", action="store_true")

    p_sample = sub.add_parser("judge-sample")
    p_sample.add_argument("--n", type=int, default=10)
    p_sample.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "ingest":
        return _cmd_ingest(args)
    if args.command == "run":
        return _cmd_run(args)
    return _cmd_judge_sample(args)


if __name__ == "__main__":
    raise SystemExit(main())
