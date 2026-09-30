"""Tests for arms, the eval runner, summary report, and the eval CLI."""

import json

import pytest

from docuchat.config import Settings
from docuchat.evaluate import (
    ARMS,
    COMPARE_TO,
    StubLLM,
    arm_settings,
    judge_agreement,
    main,
    needs_llm,
    ragas_scores,
    render_summary,
    run_arm,
)
from docuchat.judge import UNANSWERABLE_REFERENCE

QUESTIONS = [
    {
        "id": "q1",
        "kind": "fact",
        "question": "What is the loan amount?",
        "reference_answer": "$100,000",
        "evidence": [{"filename": "a.pdf", "page": 1}],
    },
    {
        "id": "q2",
        "kind": "unanswerable",
        "question": "What is the price of tea in China?",
        "reference_answer": UNANSWERABLE_REFERENCE,
        "evidence": [],
    },
]


@pytest.fixture(autouse=True)
def _no_real_models(monkeypatch, fake_models):
    # ask() would otherwise load the embedder and reranker from the HF cache (absent in CI).
    monkeypatch.setattr("docuchat.pipeline.get_encoder", lambda cfg: fake_models["encoder"])
    monkeypatch.setattr("docuchat.pipeline.get_cross_encoder", lambda cfg: fake_models["cross_encoder"])


def test_arm_settings_rerank():
    cfg = arm_settings("+rerank", Settings())
    assert cfg.use_rerank is True
    assert cfg.use_rewrite is False
    assert cfg.use_decomposition is False


def test_every_arm_override_key_is_a_real_settings_field():
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(Settings)}
    for name, (_profile, overrides) in ARMS.items():
        for key in overrides:
            assert key in field_names, f"{name}: {key!r} is not a Settings field"


def test_needs_llm():
    assert needs_llm(Settings(use_rewrite=True, use_decomposition=False)) is True
    assert needs_llm(Settings(use_rewrite=False, use_decomposition=False)) is False


def test_run_arm_produces_records_with_all_keys(fake_store):
    cfg = arm_settings("+rerank", Settings())
    llm = StubLLM()
    records = run_arm("+rerank", QUESTIONS, fake_store, cfg, llm)

    assert len(records) == 2
    for r in records:
        for key in ("id", "kind", "question", "answer", "sources", "candidates", "retrieval", "llm_calls", "latency_s"):
            assert key in r

    assert records[0]["retrieval"]["hit"] == 1.0
    assert records[0]["llm_calls"] == 1


def test_run_arm_with_judge_adds_verdict_and_uses_unanswerable_reference(fake_store, fake_llm_returning):
    cfg = arm_settings("+rerank", Settings())
    llm = StubLLM()
    judge_llm = fake_llm_returning('{"refused": true, "correctness": 2, "faithful": true, "rationale": "ok"}')
    records = run_arm("+rerank", QUESTIONS, fake_store, cfg, llm, judge_llm=judge_llm)

    assert all("verdict" in r for r in records)
    unanswerable_prompt = judge_llm.prompts[1]
    assert UNANSWERABLE_REFERENCE in unanswerable_prompt


def test_run_arm_records_error_and_continues(fake_store, monkeypatch):
    cfg = arm_settings("+rerank", Settings())
    llm = StubLLM()

    calls = {"n": 0}

    def fake_ask(query, store, cfg, llm=None, encoder=None, cross_encoder=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return {
            "answer": "ans",
            "sources": [],
            "debug": {"candidates": [], "contexts": []},
        }

    monkeypatch.setattr("docuchat.evaluate.ask", fake_ask)
    records = run_arm("+rerank", QUESTIONS, fake_store, cfg, llm)

    assert "error" in records[0]
    assert "answer" not in records[0]
    assert calls["n"] == 2
    assert "error" not in records[1]


def test_render_summary_contains_arms_na_winloss_and_skipped():
    # q1 flips from a recall hit to a total miss (+hybrid loses); q2 is
    # unanswerable with identical recall in both arms, which would count as
    # a tie if it weren't excluded -- it must not show up in the win/loss row.
    arm_records = {
        "baseline": [
            {"id": "q1", "kind": "fact", "retrieval": {"hit": 1.0, "recall": 1.0, "mrr": 1.0, "candidate_recall": 1.0}, "llm_calls": 1},
            {"id": "q2", "kind": "unanswerable", "retrieval": {"hit": 0.0, "recall": 0.0, "mrr": 0.0, "candidate_recall": 0.0}, "llm_calls": 1},
        ],
        "+hybrid": [
            {"id": "q1", "kind": "fact", "retrieval": {"hit": 1.0, "recall": 0.0, "mrr": 0.0, "candidate_recall": 1.0}, "llm_calls": 1},
            {"id": "q2", "kind": "unanswerable", "retrieval": {"hit": 0.0, "recall": 0.0, "mrr": 0.0, "candidate_recall": 0.0}, "llm_calls": 1},
        ],
    }
    skipped = {"full": "ANTHROPIC_API_KEY is not set"}
    out = render_summary(arm_records, skipped, unreachable=[], validation=None)

    assert "baseline" in out
    assert "+hybrid" in out
    assert "n/a" in out  # unjudged correctness
    assert "full" in out
    assert "ANTHROPIC_API_KEY is not set" in out
    # brief-required win/loss assertion: q1 is the only counted question
    # (unanswerable q2 is excluded), so it's 0 wins, 1 loss, 0 ties.
    assert "| +hybrid vs baseline | recall | 0 | 1 | 0 |" in out


def test_render_summary_uses_compare_to_not_arms_adjacency():
    # Running only full, full/generic, and full/embed-bge-base: both later
    # arms compare against "full" (their COMPARE_TO base, which was run),
    # and "full" itself gets no row since its base ("+rerank") was not run.
    assert COMPARE_TO["full/generic"] == "full"
    assert COMPARE_TO["full/embed-bge-base"] == "full"

    def _recs(recall):
        return [{"id": "q1", "kind": "fact", "retrieval": {"hit": 1.0, "recall": recall, "mrr": 1.0, "candidate_recall": 1.0}, "llm_calls": 1}]

    arm_records = {
        "full": _recs(0.5),
        "full/generic": _recs(1.0),
        "full/embed-bge-base": _recs(0.0),
    }
    out = render_summary(arm_records, skipped={}, unreachable=[], validation=None)

    assert "full/generic vs full" in out
    assert "full/embed-bge-base vs full" in out
    assert " vs full/generic" not in out
    assert " vs full/embed-bge-base" not in out
    # "full" has no run base (+rerank wasn't run), so it gets no comparison row.
    for line in out.splitlines():
        assert not line.startswith("| full vs")


def test_main_run_unknown_arm_returns_2(capsys):
    rc = main(["run", "--arms", "+rerenk"])
    assert rc == 2
    captured = capsys.readouterr()
    assert "+rerank" in captured.err


def test_main_run_judge_missing_key_returns_2_before_load_snapshot(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def boom(*args, **kwargs):
        raise AssertionError("load_snapshot should not be called")

    monkeypatch.setattr("docuchat.evaluate.load_snapshot", boom)
    rc = main(["run", "--judge"])
    assert rc == 2


def test_judge_agreement_mixed():
    items = [
        {"judge": {"correctness": 2, "faithful": True, "refused": False},
         "human": {"correctness": 2, "faithful": True, "refused": False}},
        {"judge": {"correctness": 2, "faithful": True, "refused": False},
         "human": {"correctness": 0, "faithful": True, "refused": False}},
        {"judge": {"correctness": 2, "faithful": True, "refused": False},
         "human": {"correctness": None, "faithful": None, "refused": None}},
    ]
    assert judge_agreement(items) == {"agreement": 0.5, "n": 2}


def test_judge_sample_refuses_to_overwrite_without_force(tmp_path, monkeypatch):
    monkeypatch.setattr("docuchat.evaluate.RESULTS_DIR", tmp_path)
    monkeypatch.setattr("docuchat.evaluate.QUESTIONS_PATH", tmp_path / "questions.yaml")
    existing = tmp_path / "judge_validation.yaml"
    existing.write_text("- human: {correctness: 2}\n")

    rc = main(["judge-sample"])

    assert rc == 2
    assert existing.read_text() == "- human: {correctness: 2}\n"


def test_render_summary_judge_model_from_env(monkeypatch):
    monkeypatch.setenv("DOCUCHAT_JUDGE_MODEL", "x-judge")
    arm_records = {
        "baseline": [
            {
                "id": "q1",
                "kind": "fact",
                "retrieval": {"hit": 1.0, "recall": 1.0, "mrr": 1.0, "candidate_recall": 1.0},
                "llm_calls": 1,
                "verdict": {"correctness": 2, "faithful": True, "refused": False},
            },
        ],
    }
    out = render_summary(arm_records, skipped={}, unreachable=[], validation=None)
    assert "x-judge" in out


def test_ragas_missing_gives_install_hint(monkeypatch):
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "ragas" or name.startswith("ragas."):
            raise ImportError("no ragas")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(SystemExit) as exc_info:
        main(["run", "--ragas"])
    assert "eval-ragas" in str(exc_info.value)


@pytest.mark.integration
def test_ragas_scores_on_two_records():
    records = [
        {
            "question": "What is the loan amount?",
            "answer": "The loan amount is $100,000.",
            "contexts": ["The borrower's loan amount is $100,000, per Section 2."],
        },
        {
            "question": "Who is the lender?",
            "answer": "The lender is Acme Bank.",
            "contexts": ["Acme Bank is the lender under this agreement."],
        },
    ]
    scores = ragas_scores(records, Settings())

    assert 0.0 <= scores["ragas_faithfulness"] <= 1.0
    assert 0.0 <= scores["ragas_answer_relevancy"] <= 1.0


def test_render_summary_shows_ragas_columns_when_records_carry_them():
    arm_records = {
        "baseline": [
            {
                "id": "q1",
                "kind": "fact",
                "retrieval": {"hit": 1.0, "recall": 1.0, "mrr": 1.0, "candidate_recall": 1.0},
                "llm_calls": 1,
                "ragas_faithfulness": 0.8,
                "ragas_answer_relevancy": 0.9,
            },
        ],
        "+hybrid": [
            {
                "id": "q1",
                "kind": "fact",
                "retrieval": {"hit": 1.0, "recall": 1.0, "mrr": 1.0, "candidate_recall": 1.0},
                "llm_calls": 1,
            },
        ],
    }
    out = render_summary(arm_records, skipped={}, unreachable=[], validation=None)

    assert "ragas_faithfulness" in out
    assert "ragas_answer_relevancy" in out
    assert "0.800" in out
    lines = [l for l in out.splitlines() if l.startswith("| +hybrid |")]
    assert len(lines) == 1
    assert "n/a" in lines[0]


def test_judge_agreement_none_when_nothing_filled():
    items = [
        {"judge": {"correctness": 2, "faithful": True, "refused": False},
         "human": {"correctness": None, "faithful": None, "refused": None}},
    ]
    assert judge_agreement(items) is None
