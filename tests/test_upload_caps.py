"""Upload validation runs before Docling ever sees a file."""

from pathlib import Path

import pytest

from docuchat.config import Settings
from docuchat.service import UploadRejected, check_upload, safe_name

FIXTURE = Path(__file__).parent / "fixtures" / "cfpb_closing_disclosure.pdf"  # 6 pages


def pdf(name="cd.pdf"):
    return (name, FIXTURE.read_bytes())


def status_of(files, **overrides):
    with pytest.raises(UploadRejected) as exc:
        check_upload(files, Settings(**overrides))
    return exc.value.status


def test_accepts_a_pdf_within_caps():
    check_upload([pdf()], Settings())


def test_rejects_an_empty_upload():
    assert status_of([]) == 422


def test_rejects_a_non_pdf():
    assert status_of([("notes.txt", b"hello")]) == 415


# Review Focus 4
def test_rejects_a_corrupt_pdf():
    assert status_of([("bad.pdf", b"%PDF-1.7 this is not really a pdf")]) == 415


def test_size_cap_is_checked_before_parsing():
    # not parseable: a 413 proves the size check ran first
    huge = ("big.pdf", b"%PDF" + b"0" * (1024 * 1024))
    assert status_of([huge], max_upload_mb=1) == 413


def test_page_cap_counts_pages_across_files():
    check_upload([pdf()], Settings(max_upload_pages=6))
    assert status_of([pdf()], max_upload_pages=5) == 413
    assert status_of([pdf("a.pdf"), pdf("b.pdf")], max_upload_pages=11) == 413


# Review Focus 3
def test_safe_name_strips_paths_adds_suffix_and_dedupes():
    assert safe_name("../../etc/x.pdf", 0, set()) == "x.pdf"
    assert safe_name("scan", 1, set()) == "scan.pdf"
    assert safe_name("", 2, set()) == "upload_2.pdf"
    assert safe_name("x.pdf", 3, {"x.pdf"}) == "3_x.pdf"


def test_safe_name_never_collides():
    taken: set[str] = set()
    for i, n in enumerate(["x.pdf", "x.pdf", "1_x.pdf"]):
        taken.add(safe_name(n, i, taken))
    assert len(taken) == 3


def test_safe_name_strips_nul_and_truncates():
    assert "\x00" not in safe_name("a\x00b.pdf", 0, set())
    long = safe_name("a" * 300 + ".pdf", 0, set())
    assert len(long) <= 110 and long.endswith(".pdf")
