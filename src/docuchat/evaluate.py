"""Evaluation metrics and aggregation for docuchat runs.

A "match" is the pair (filename, page_number); page numbers are 1-based
physical PDF page indexes. Functions here are pure and read only plain
dicts/lists so they have no heavy imports.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from llama_index.core.schema import TextNode

from docuchat.config import Settings

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

    return {
        **retrieval_stats,
        **correctness_stats,
        **refusal_stats,
        "llm_calls": llm_calls_stat,
        **counts,
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
