"""Tests for docuchat.ui: construction and answer formatting only."""

import gradio as gr

from docuchat.ui import build, format_answer

SOURCE = {"filename": "cd.pdf", "page_number": 2, "doc_type": "Closing Disclosure",
          "section_title": "", "score": 0.9, "preview": "Loan Amount $162,000..."}


def test_build_returns_blocks(fake_service):
    assert isinstance(build(fake_service), gr.Blocks)


def test_answer_lists_sources():
    text = format_answer({"answer": "It is $162,000.", "sources": [SOURCE], "expired": False})
    assert text.startswith("It is $162,000.")
    assert "cd.pdf, page 2: Loan Amount $162,000..." in text


def test_expired_session_is_announced():
    text = format_answer({"answer": "x", "sources": [], "expired": True})
    assert "session expired" in text.lower()
