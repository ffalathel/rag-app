"""Tests for arms, the eval runner, summary report, and the eval CLI."""

import json

import pytest

from docuchat.config import Settings
from docuchat.evaluate import (
    ARMS,
    StubLLM,
    arm_settings,
    judge_agreement,
    main,
    needs_llm,
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
    arm_records = {
        "baseline": [
            {"id": "q1", "kind": "fact", "retrieval": {"hit": 1.0, "recall": 1.0, "mrr": 1.0, "candidate_recall": 1.0}, "llm_calls": 1},
        ],
        "+hybrid": [
            {"id": "q1", "kind": "fact", "retrieval": {"hit": 1.0, "recall": 0.0, "mrr": 0.0, "candidate_recall": 1.0}, "llm_calls": 1},
        ],
    }
    skipped = {"full": "ANTHROPIC_API_KEY is not set"}
    out = render_summary(arm_records, skipped, unreachable=[], validation=None)

    assert "baseline" in out
    assert "+hybrid" in out
    assert "n/a" in out  # unjudged correctness
    assert "full" in out
    assert "ANTHROPIC_API_KEY is not set" in out


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


def test_judge_agreement_none_when_nothing_filled():
    items = [
        {"judge": {"correctness": 2, "faithful": True, "refused": False},
         "human": {"correctness": None, "faithful": None, "refused": None}},
    ]
    assert judge_agreement(items) is None
