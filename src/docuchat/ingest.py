"""Ingestion: Docling-backed PDF -> page dicts.

Replaces the notebook's hand-rolled OCR chain (preprocess_image,
ocr_with_paddle, ocr_with_tesseract, ingest_pdf, ingest_all_pdfs) with
Docling's layout/OCR/table-structure pipeline. Docling (and its torch
dependency) is imported inside `load_pdf`, never at module scope, so
importing this module stays cheap -- see test_import_purity.py.
"""

from pathlib import Path

from docuchat.config import Settings

PAGE_KEYS: frozenset[str] = frozenset(
    {"filename", "page_number", "text", "source_type", "doc_type"}
)


def _pages_from_texts(filename: str, texts: list[str], cfg: Settings) -> list[dict]:
    """Build page dicts from per-page text, dropping pages below min_text_length."""
    pages = []
    for i, text in enumerate(texts, start=1):
        if len(text.strip()) < cfg.min_text_length:
            continue
        pages.append(
            {
                "filename": filename,
                "page_number": i,
                "text": text,
                "source_type": "docling",
                "doc_type": "Unknown",
            }
        )
    return pages


def _make_converter(cfg: Settings):
    """Build a Docling DocumentConverter configured from cfg."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = cfg.do_ocr
    pipeline_options.do_table_structure = cfg.do_table_structure

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def load_pdf(path: str | Path, cfg: Settings, converter=None) -> list[dict]:
    """Convert a single PDF into page dicts via Docling.

    Builds its own converter when none is given; load_directory shares one
    converter across every PDF instead.
    """
    path = Path(path)
    if converter is None:
        converter = _make_converter(cfg)
    result = converter.convert(path)
    doc = result.document

    texts = [doc.export_to_markdown(page_no=n) for n in range(1, doc.num_pages() + 1)]
    return _pages_from_texts(path.name, texts, cfg)


def load_directory(folder: str | Path, cfg: Settings) -> list[dict]:
    """Convert every PDF in a directory (case-insensitive glob, sorted)."""
    folder = Path(folder)
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".pdf")
    if not paths:
        return []
    converter = _make_converter(cfg)
    pages = []
    for path in paths:
        pages.extend(load_pdf(path, cfg, converter=converter))
    return pages
