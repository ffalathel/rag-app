"""Evaluation metrics and aggregation for docuchat runs.

A "match" is the pair (filename, page_number); page numbers are 1-based
physical PDF page indexes. Functions here are pure and read only plain
dicts/lists so they have no heavy imports.
"""

import numpy as np

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
