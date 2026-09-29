"""Chunking: section-aware recursive splitting plus semantic merge.

Ported from complete_mortgage_rag_pipeline.py (lines 318-404), with all
tuning constants read from Settings (never module-level defaults) and
section patterns sourced from the active domain profile. `semantic_merge`
takes its encoder as a parameter instead of a global model, and cosine
similarity is a plain numpy dot product of the already-normalised vectors.
"""

import re

import numpy as np

from docuchat.config import Settings


def find_section_breaks(text: str, patterns: tuple[str, ...]) -> list[tuple[int, str]]:
    breaks = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            start = match.start()
            header_start = start + 1 if text[start:start + 1] == "\n" else start
            newline_idx = text.find("\n", header_start)
            header_end = newline_idx if newline_idx != -1 else len(text)
            title = text[header_start:header_end].strip()
            breaks.append((start, title))
    breaks.sort(key=lambda x: x[0])
    return breaks


def recursive_chunk(text: str, cfg: Settings) -> list[tuple[str, str]]:
    """Split by section headers -> paragraphs -> sentences."""
    if not text or not text.strip():
        return []

    max_tokens = cfg.chunk_max_tokens
    min_tokens = cfg.chunk_min_tokens

    breaks = find_section_breaks(text, cfg.profile.section_patterns)
    if breaks:
        sections = []
        if breaks[0][0] > 0:
            preamble = text[:breaks[0][0]].strip()
            if len(preamble) > min_tokens:
                sections.append((preamble, "Preamble"))
        for i, (pos, header) in enumerate(breaks):
            end = breaks[i + 1][0] if i + 1 < len(breaks) else len(text)
            section_text = text[pos:end].strip()
            if section_text:
                sections.append((section_text, header))
    else:
        sections = [(text, "Full Page")]

    chunks = []
    for section_text, section_title in sections:
        if len(section_text.split()) * 1.3 <= max_tokens:
            chunks.append((section_text, section_title))
        else:
            paragraphs = re.split(r"\n\s*\n", section_text)
            current_chunk = ""
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                test_chunk = current_chunk + "\n\n" + para if current_chunk else para
                if len(test_chunk.split()) * 1.3 <= max_tokens:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append((current_chunk, section_title))
                    current_chunk = para
            if current_chunk:
                chunks.append((current_chunk, section_title))

    return [(t, s) for t, s in chunks if len(t.split()) >= cfg.min_chunk_words]


def semantic_merge(
    chunks: list[tuple[str, str]], encoder, cfg: Settings
) -> list[tuple[str, str]]:
    """Merge adjacent chunks if semantically similar."""
    if len(chunks) <= 1:
        return chunks

    threshold = cfg.merge_threshold
    max_tokens = cfg.merge_max_tokens

    texts = [c[0] for c in chunks]
    embeddings = encoder.encode(texts, normalize_embeddings=True)
    merged = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_text, prev_title = merged[-1]
        curr_text, curr_title = chunks[i]
        sim = float(np.dot(embeddings[i - 1], embeddings[i]))
        combined = prev_text + "\n\n" + curr_text
        if sim >= threshold and len(combined.split()) * 1.3 <= max_tokens:
            merged[-1] = (combined, prev_title)
        else:
            merged.append((curr_text, curr_title))
    return merged


def chunk_page(text: str, encoder, cfg: Settings) -> list[tuple[str, str]]:
    """Recursive split -> semantic merge."""
    raw = recursive_chunk(text, cfg)
    return semantic_merge(raw, encoder, cfg)


def detect_table_content(text: str) -> bool:
    """Heuristic: does this chunk contain tabular data?"""
    indicators = 0
    if len(re.findall(r"^.*\d+[.,]\d+.*\d+[.,]\d+.*$", text, re.MULTILINE)) >= 3:
        indicators += 1
    if len(re.findall(r"^.+\s{3,}.+$", text, re.MULTILINE)) >= 3:
        indicators += 1
    if "|" in text or "\t" in text:
        indicators += 1
    if len(re.findall(r"\$[\d,]+", text)) >= 3:
        indicators += 1
    return indicators >= 2
