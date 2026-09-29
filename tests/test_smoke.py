"""End-to-end smoke test against the real pipeline: real Docling ingest,
real embedding/cross-encoder models, and a real LLM call.

The provider comes from the environment (DOCUCHAT_LLM_PROVIDER, default
anthropic). Every test skips when get_llm reports that provider's key or
model file is missing, before the slow ingest runs.
"""

from pathlib import Path

import pytest

from docuchat.classify import classify_pages
from docuchat.config import Settings
from docuchat.index import build_store
from docuchat.ingest import load_pdf
from docuchat.models import get_llm
from docuchat.pipeline import ask

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def cfg():
    return Settings.from_env()


@pytest.fixture(scope="module")
def real_store(cfg):
    try:
        llm = get_llm(cfg)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    pages = load_pdf(Path(__file__).parent / "fixtures" / "cfpb_closing_disclosure.pdf", cfg)
    pages = classify_pages(pages, llm, cfg)
    return build_store(pages, cfg)


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What is the loan amount?", "162,000"),
        ("What is the interest rate?", "3.875"),
        ("Is there a prepayment penalty?", "3,240"),
    ],
)
def test_end_to_end_answers_contain_the_correct_figure(real_store, cfg, question, expected):
    result = ask(question, real_store, cfg)
    assert expected in result["answer"]
    assert result["num_chunks_used"] > 0
    assert result["sources"][0]["page_number"] >= 1


def test_refuses_when_the_answer_is_absent(real_store, cfg):
    result = ask("What is the borrower's credit score?", real_store, cfg)
    assert "cannot find" in result["answer"].lower()
