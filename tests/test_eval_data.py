"""Tests for ground-truth loading, validation, and node snapshots."""

import re
import shutil
from pathlib import Path

import pytest
from llama_index.core.schema import TextNode

from docuchat.config import Settings
from docuchat.evaluate import (
    CORPUS_DIR,
    INGEST_FIELDS,
    NODES_DIR,
    QUESTIONS_PATH,
    StaleSnapshotError,
    corpus_hashes,
    load_questions,
    load_snapshot,
    unreachable_evidence,
    validate_questions,
    write_snapshot,
)

FIXTURE_PDF = "tests/fixtures/cfpb_closing_disclosure.pdf"


@pytest.fixture
def corpus_dir(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    shutil.copy(FIXTURE_PDF, corpus / "cfpb_closing_disclosure.pdf")
    return corpus


def valid_question(**overrides):
    q = {
        "id": "q1",
        "question": "What is the loan amount?",
        "reference_answer": "$100,000",
        "evidence": [{"filename": "cfpb_closing_disclosure.pdf", "page": 1}],
        "kind": "fact",
    }
    q.update(overrides)
    return q


def test_valid_questions_return_no_errors(corpus_dir):
    assert validate_questions([valid_question()], corpus_dir) == []


def test_missing_key(corpus_dir):
    q = valid_question()
    del q["reference_answer"]
    errors = validate_questions([q], corpus_dir)
    assert len(errors) == 1
    assert "q1" in errors[0]


def test_bad_kind(corpus_dir):
    q = valid_question(kind="bogus")
    errors = validate_questions([q], corpus_dir)
    assert len(errors) == 1
    assert "q1" in errors[0]


def test_duplicate_id(corpus_dir):
    q1 = valid_question(id="dup")
    q2 = valid_question(id="dup")
    errors = validate_questions([q1, q2], corpus_dir)
    assert len(errors) == 1
    assert "dup" in errors[0]


def test_unanswerable_with_evidence(corpus_dir):
    q = valid_question(kind="unanswerable")
    errors = validate_questions([q], corpus_dir)
    assert len(errors) == 1
    assert "q1" in errors[0]


def test_answerable_without_evidence(corpus_dir):
    q = valid_question(evidence=[])
    errors = validate_questions([q], corpus_dir)
    assert len(errors) == 1
    assert "q1" in errors[0]


def test_unknown_filename(corpus_dir):
    q = valid_question(evidence=[{"filename": "nope.pdf", "page": 1}])
    errors = validate_questions([q], corpus_dir)
    assert len(errors) == 1
    assert "q1" in errors[0]


def test_page_zero(corpus_dir):
    q = valid_question(evidence=[{"filename": "cfpb_closing_disclosure.pdf", "page": 0}])
    errors = validate_questions([q], corpus_dir)
    assert len(errors) == 1
    assert "q1" in errors[0]


def test_page_out_of_range(corpus_dir):
    q = valid_question(evidence=[{"filename": "cfpb_closing_disclosure.pdf", "page": 7}])
    errors = validate_questions([q], corpus_dir)
    assert len(errors) == 1
    assert "q1" in errors[0]


def test_snapshot_round_trip(tmp_path, corpus_dir, make_node):
    nodes = [make_node(text="hello world", chunk_id=0), make_node(text="second node", chunk_id=1)]
    cfg = Settings()
    path = tmp_path / "snapshot.jsonl"
    write_snapshot(path, nodes, corpus_dir, cfg)
    loaded = load_snapshot(path, corpus_dir, cfg)
    assert [n.get_content() for n in loaded] == [n.get_content() for n in nodes]
    assert [n.metadata for n in loaded] == [n.metadata for n in nodes]


def test_stale_snapshot_on_corpus_change(tmp_path, corpus_dir, make_node):
    nodes = [make_node()]
    cfg = Settings()
    path = tmp_path / "snapshot.jsonl"
    write_snapshot(path, nodes, corpus_dir, cfg)

    with open(corpus_dir / "cfpb_closing_disclosure.pdf", "ab") as f:
        f.write(b"\x00")

    with pytest.raises(StaleSnapshotError) as exc_info:
        load_snapshot(path, corpus_dir, cfg)
    assert "ingest" in str(exc_info.value)


def test_stale_snapshot_on_settings_change(tmp_path, corpus_dir, make_node):
    nodes = [make_node()]
    cfg = Settings()
    path = tmp_path / "snapshot.jsonl"
    write_snapshot(path, nodes, corpus_dir, cfg)

    with pytest.raises(StaleSnapshotError) as exc_info:
        load_snapshot(path, corpus_dir, Settings(chunk_max_tokens=256))
    assert "ingest" in str(exc_info.value)


def test_not_stale_on_embed_model_change(tmp_path, corpus_dir, make_node):
    nodes = [make_node()]
    cfg = Settings()
    path = tmp_path / "snapshot.jsonl"
    write_snapshot(path, nodes, corpus_dir, cfg)

    loaded = load_snapshot(path, corpus_dir, Settings(embed_model_name="BAAI/bge-base-en-v1.5"))
    assert len(loaded) == 1


def test_unreachable_evidence_flags_missing_node(make_node):
    questions = [
        valid_question(id="q1", evidence=[{"filename": "cfpb_closing_disclosure.pdf", "page": 1}]),
        valid_question(id="q2", evidence=[{"filename": "cfpb_closing_disclosure.pdf", "page": 99}]),
        valid_question(id="q3", kind="unanswerable", evidence=[]),
    ]
    nodes = [make_node(filename="cfpb_closing_disclosure.pdf", page_number=1)]
    result = unreachable_evidence(questions, nodes)
    assert result == ["q2"]


def test_ingest_fields_excludes_embed_model_name():
    assert "embed_model_name" not in INGEST_FIELDS


def test_corpus_hashes_sorted_by_filename(corpus_dir):
    hashes = corpus_hashes(corpus_dir)
    assert list(hashes.keys()) == sorted(hashes.keys())
    assert set(hashes.keys()) == {"cfpb_closing_disclosure.pdf"}


def _parse_sources_md(path: Path) -> list[dict[str, str]]:
    """Parse the `file | origin URL | sha256 | document type | pages | scanned`
    table in SOURCES.md into a list of row dicts."""
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0] in ("file", "") or set(cells[0]) <= {"-"}:
            continue
        rows.append(
            {
                "file": cells[0],
                "origin_url": cells[1],
                "sha256": cells[2],
                "document_type": cells[3],
                "pages": cells[4],
                "scanned": cells[5],
            }
        )
    return rows


def test_sources_md_matches_corpus():
    rows = _parse_sources_md(CORPUS_DIR / "SOURCES.md")

    corpus_filenames = {p.name for p in CORPUS_DIR.glob("*.pdf")}
    assert {row["file"] for row in rows} == corpus_filenames

    actual_hashes = corpus_hashes(CORPUS_DIR)
    for row in rows:
        assert re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), row["file"]
        assert row["sha256"] == actual_hashes[row["file"]], row["file"]

    assert any(row["scanned"].lower().startswith("yes") for row in rows)


# --- Real ground truth and snapshots -----------------------------------------

TARGET_MIX = {"fact": 15, "table": 10, "multi_hop": 10, "unanswerable": 5}


def test_questions_yaml_is_valid():
    assert validate_questions(load_questions(QUESTIONS_PATH), CORPUS_DIR) == []


def test_question_mix():
    questions = load_questions(QUESTIONS_PATH)
    assert len(questions) >= 35
    for kind, target in TARGET_MIX.items():
        count = sum(q["kind"] == kind for q in questions)
        assert abs(count - target) <= 3, (kind, count)


def test_all_evidence_reachable():
    questions = load_questions(QUESTIONS_PATH)
    nodes = load_snapshot(NODES_DIR / "mortgage.jsonl", CORPUS_DIR, Settings())
    assert unreachable_evidence(questions, nodes) == []
