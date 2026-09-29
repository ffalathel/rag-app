"""End-to-end smoke test against the real pipeline: real Docling ingest,
real embedding/cross-encoder models, and a real Anthropic LLM call.

Skipped entirely without ANTHROPIC_API_KEY (the whole module, so the
real_store fixture -- which itself calls get_llm -- never runs keyless).
"""

import os

import pytest

from docuchat.classify import classify_pages
from docuchat.config import Settings
from docuchat.index import build_store
from docuchat.ingest import load_pdf
from docuchat.models import get_llm
from docuchat.pipeline import ask

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="needs ANTHROPIC_API_KEY for the real LLM",
    ),
]


@pytest.fixture(scope="module")
def real_store():
    cfg = Settings()
    pages = load_pdf("tests/fixtures/cfpb_closing_disclosure.pdf", cfg)
    pages = classify_pages(pages, get_llm(cfg), cfg)
    return build_store(pages, cfg)


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What is the loan amount?", "162,000"),
        ("What is the interest rate?", "3.875"),
        ("Is there a prepayment penalty?", "3,240"),
    ],
)
def test_end_to_end_answers_contain_the_correct_figure(real_store, question, expected):
    result = ask(question, real_store, Settings())
    assert expected in result["answer"]
    assert result["num_chunks_used"] > 0
    assert result["sources"][0]["page_number"] >= 1


def test_refuses_when_the_answer_is_absent(real_store):
    result = ask("What is the borrower's credit score?", real_store, Settings())
    assert "cannot find" in result["answer"].lower()
