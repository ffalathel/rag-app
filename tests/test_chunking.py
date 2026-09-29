import pytest

from docuchat.chunking import recursive_chunk, semantic_merge
from docuchat.config import Settings


def test_splits_on_numbered_section_headers():
    text = (
        "1. PAYMENTS\n"
        "Borrower shall pay the monthly installment amount promptly and in full each month.\n\n"
        "2. DEFAULT\n"
        "Lender may accelerate the entire loan balance immediately upon any default event."
    )
    chunks = recursive_chunk(text, Settings())
    assert len(chunks) == 2
    assert "PAYMENTS" in chunks[0][1] and "DEFAULT" in chunks[1][1]


def test_no_chunk_exceeds_max_tokens():
    text = "\n\n".join("word " * 200 for _ in range(10))
    cfg = Settings(chunk_max_tokens=512)
    for chunk_text, _ in recursive_chunk(text, cfg):
        assert len(chunk_text.split()) * 1.3 <= cfg.chunk_max_tokens


def test_drops_chunks_below_min_chunk_words():
    # "1. AA" and "2. BB" both match the numbered-header pattern, so this
    # splits into a 3-word section and a 52-word one; only the long one
    # survives the min_chunk_words filter.
    chunks = recursive_chunk("1. AA\nshort\n\n2. BB\n" + "word " * 50, Settings())
    assert len(chunks) == 1
    assert all(len(t.split()) >= Settings().min_chunk_words for t, _ in chunks)


def test_unheadered_text_becomes_one_full_page_chunk():
    chunks = recursive_chunk("word " * 50, Settings())
    assert len(chunks) == 1 and chunks[0][1] == "Full Page"


def test_profile_selects_the_section_patterns():
    # "(a)" is a mortgage sub-clause pattern; the generic profile omits it
    text = "(a) " + "word " * 40 + "\n\n(b) " + "word " * 40
    mortgage = recursive_chunk(text, Settings(domain_profile="mortgage"))
    generic = recursive_chunk(text, Settings(domain_profile="generic"))
    assert len(mortgage) != len(generic)


# Review Focus 4
@pytest.mark.parametrize("text", ["", "   ", "\n\n\t\n"])
def test_empty_or_whitespace_text_returns_no_chunks(text):
    assert recursive_chunk(text, Settings()) == []


def test_semantic_merge_combines_similar_adjacent_chunks(stub_encoder):
    # stub_encoder returns identical vectors → cosine similarity 1.0
    merged = semantic_merge([("a " * 20, "S1"), ("b " * 20, "S1")], stub_encoder, Settings())
    assert len(merged) == 1


def test_semantic_merge_respects_merge_max_tokens(stub_encoder):
    big = "word " * 400
    merged = semantic_merge([(big, "S1"), (big, "S1")], stub_encoder, Settings())
    assert len(merged) == 2
