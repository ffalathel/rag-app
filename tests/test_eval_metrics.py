import warnings

import pytest

from docuchat.evaluate import aggregate, bootstrap_ci, retrieval_metrics, win_loss


def test_retrieval_metrics_basic():
    result = retrieval_metrics([("a", 1), ("b", 2)], [("a", 1), ("c", 3)], [("a", 1), ("b", 2)])
    assert result == {"hit": 1.0, "recall": 0.5, "mrr": 1.0, "candidate_recall": 1.0}


def test_retrieval_metrics_mrr_rank():
    result = retrieval_metrics([("a", 3)], [("x", 1), ("y", 1), ("a", 3)], [])
    assert result["mrr"] == pytest.approx(1 / 3)


def test_retrieval_metrics_duplicate_context_counts_once():
    result = retrieval_metrics([("a", 1), ("b", 2)], [("a", 1), ("a", 1)], [])
    assert result["recall"] == 0.5


def test_retrieval_metrics_no_hit():
    result = retrieval_metrics([("a", 1)], [("b", 2)], [("c", 3)])
    assert result == {"hit": 0.0, "recall": 0.0, "mrr": 0.0, "candidate_recall": 0.0}


def test_bootstrap_ci_constant_values():
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0)


def test_bootstrap_ci_deterministic():
    values = [0.0, 1.0, 0.5, 1.0, 0.0, 1.0]
    assert bootstrap_ci(values) == bootstrap_ci(values)


def test_bootstrap_ci_empty():
    assert bootstrap_ci([]) is None


def test_win_loss():
    assert win_loss({"q1": 1, "q2": 0, "q3": 1}, {"q1": 1, "q2": 1, "q3": 0}) == (1, 1, 1)


def test_win_loss_ignores_ids_missing_from_either():
    assert win_loss({"q1": 1, "q2": 0}, {"q1": 0, "q3": 1}) == (0, 1, 0)


def _fact_record(id_, hit=1.0, recall=1.0, mrr=1.0, candidate_recall=1.0, verdict=None):
    record = {
        "id": id_,
        "kind": "fact",
        "retrieval": {"hit": hit, "recall": recall, "mrr": mrr, "candidate_recall": candidate_recall},
        "llm_calls": 2,
    }
    if verdict is not None:
        record["verdict"] = verdict
    return record


def _unanswerable_record(id_, verdict=None):
    record = {
        "id": id_,
        "kind": "unanswerable",
        "retrieval": {"hit": 0.0, "recall": 0.0, "mrr": 0.0, "candidate_recall": 0.0},
        "llm_calls": 1,
    }
    if verdict is not None:
        record["verdict"] = verdict
    return record


def _error_record(id_):
    return {"id": id_, "kind": "fact", "question": "q", "error": "boom"}


def test_aggregate_excludes_unanswerable_from_retrieval_and_errors_from_everything():
    records = [
        _fact_record("q1", recall=1.0),
        _unanswerable_record("q2"),
        _error_record("q3"),
    ]
    result = aggregate(records)
    assert result["recall"]["n"] == 1
    assert result["recall"]["mean"] == 1.0
    assert result["errors"]["n"] == 1
    assert result["errors"]["mean"] == 1


def test_aggregate_only_errored_records_no_runtime_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = aggregate([_error_record("q1"), _error_record("q2")])
    assert result["recall"] == {"mean": None, "ci": None, "n": 0}
    assert result["errors"]["n"] == 2


def test_aggregate_only_unanswerable_records_recall_n_zero():
    result = aggregate([_unanswerable_record("q1"), _unanswerable_record("q2")])
    assert result["recall"]["n"] == 0


def test_aggregate_correctness_and_refusal_metrics():
    valid_verdict = {"refused": False, "correctness": 2, "faithful": True, "rationale": "ok"}
    refused_verdict = {"refused": True, "correctness": 0, "faithful": True, "rationale": "refused"}
    judge_error = {"judge_error": "oops"}
    records = [
        _fact_record("q1", verdict=valid_verdict),
        _fact_record("q2", verdict=refused_verdict),
        _unanswerable_record("q3", verdict=refused_verdict),
        _unanswerable_record("q4", verdict=judge_error),
        _error_record("q5"),
    ]
    result = aggregate(records)
    assert result["correct_rate"]["n"] == 3  # q1, q2, q3 have valid verdicts
    assert result["correct_rate"]["mean"] == pytest.approx(1 / 3)
    assert result["false_refusal_rate"]["n"] == 2  # q1, q2 answerable with valid verdict
    assert result["false_refusal_rate"]["mean"] == pytest.approx(0.5)
    assert result["correct_refusal_rate"]["n"] == 1  # q3 only
    assert result["correct_refusal_rate"]["mean"] == 1.0
    assert result["judge_errors"]["n"] == 1
    assert result["llm_calls"]["n"] == 4  # non-errored records
