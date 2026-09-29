"""Ingestion: Docling-backed PDF -> page-dict extraction.

`_pages_from_texts` is tested directly (no Docling, no network) so the
blank-page filter is covered without paying for a conversion. The Docling
wiring itself is only exercised by the integration test, which downloads
models on first run.
"""

import pytest

from docuchat.config import Settings
from docuchat.ingest import PAGE_KEYS, _pages_from_texts, load_directory, load_pdf


def test_page_dict_shape_is_stable():
    assert PAGE_KEYS == {"filename", "page_number", "text", "source_type", "doc_type"}


def test_blank_pages_below_min_text_length_are_dropped():
    pages = _pages_from_texts("a.pdf", ["", "   ", "x" * 100], Settings(min_text_length=50))
    assert [p["page_number"] for p in pages] == [3]


def test_pages_from_texts_shape():
    pages = _pages_from_texts("a.pdf", ["x" * 100], Settings(min_text_length=50))
    assert len(pages) == 1
    page = pages[0]
    assert set(page) == PAGE_KEYS
    assert page["filename"] == "a.pdf"
    assert page["page_number"] == 1
    assert page["text"] == "x" * 100
    assert page["source_type"] == "docling"
    assert page["doc_type"] == "unknown"


def test_load_directory_returns_empty_for_no_pdfs(tmp_path):
    assert load_directory(tmp_path, Settings()) == []


@pytest.mark.integration
def test_load_pdf_extracts_text_and_pages():
    pages = load_pdf("tests/fixtures/cfpb_closing_disclosure.pdf", Settings())
    assert len(pages) >= 1
    assert all(set(p) == PAGE_KEYS for p in pages)
    assert any("Closing Disclosure" in p["text"] for p in pages)
