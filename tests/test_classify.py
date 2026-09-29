"""Tests for page classification: heuristic first, LLM fallback."""

import pytest

from docuchat.classify import (
    classify_doc_type_heuristic,
    classify_page_with_llm,
    classify_pages,
)
from docuchat.config import Settings


@pytest.mark.parametrize(
    "text,expected",
    [
        ("This CLOSING DISCLOSURE describes...", "Closing Disclosure"),
        ("PROMISSORY NOTE dated...", "Promissory Note"),
        ("Nothing recognisable here at all", "Unknown"),
    ],
)
def test_heuristic_classification(text, expected):
    assert classify_doc_type_heuristic(text, Settings()) == expected


# Acceptance criterion 4: the profile switch is real, not decorative
def test_generic_profile_has_no_heuristic_hits(fake_llm):
    text = "This CLOSING DISCLOSURE describes..."
    assert classify_doc_type_heuristic(text, Settings()) == "Closing Disclosure"
    assert classify_doc_type_heuristic(text, Settings(domain_profile="generic")) == "Unknown"


def test_generic_profile_routes_everything_to_the_llm(fake_llm):
    pages = [{"text": "This CLOSING DISCLOSURE describes..." + "x" * 80,
              "page_number": 1, "filename": "a.pdf"}]
    classify_pages(pages, fake_llm, Settings(domain_profile="generic"))
    assert fake_llm.calls == 1


def test_llm_used_only_when_heuristic_returns_unknown(fake_llm):
    pages = [
        {"text": "CLOSING DISCLOSURE ...", "page_number": 1, "filename": "a.pdf"},
        {"text": "x" * 100, "page_number": 2, "filename": "a.pdf"},
    ]
    classify_pages(pages, fake_llm, Settings())
    assert fake_llm.calls == 1


def test_llm_classification_skipped_when_disabled(fake_llm):
    pages = [{"text": "x" * 100, "page_number": 1, "filename": "a.pdf"}]
    classify_pages(pages, fake_llm, Settings(use_llm_classification=False))
    assert fake_llm.calls == 0
    assert pages[0]["doc_type"] == "Unknown"


def test_overlong_llm_label_falls_back_to_other(fake_llm_returning):
    # Ruling C10: the notebook's classify_page_with_llm short-circuits to
    # "Unknown" for any snippet under 20 chars, before the LLM is ever
    # called. "some page text here" is exactly 19 chars, so the original
    # brief text would hit that guard, not the LLM. Lengthened well past
    # the floor so the test actually exercises the LLM's overlong-label
    # fallback, not the length guard.
    llm = fake_llm_returning("x" * 80)
    text = "some page text here, and quite a bit more of it besides"
    assert classify_page_with_llm(text, llm, Settings()) == "Other"
