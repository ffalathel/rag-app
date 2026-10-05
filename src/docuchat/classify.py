"""Page classification: keyword heuristic first, LLM as fallback.

Ported from complete_mortgage_rag_pipeline.py (lines 408-475), with the
keyword list and LLM category list read from the active domain profile
instead of hardcoded here, and the LLM snippet length read from Settings.
The keyword list is iterated in order -- ordering is load-bearing, since
"disclosure" must not shadow "closing disclosure".

Classification is per document, not per page: only the first page (lowest
page_number) of each filename is classified, and every page in that
document inherits its label.
"""

from docuchat.config import Settings


def classify_doc_type_heuristic(text: str, cfg: Settings) -> str:
    """Quick keyword-based classification."""
    lower = text.lower()
    for keyword, label in cfg.profile.doc_type_keywords:
        if keyword in lower:
            return label
    return "Unknown"


def classify_page_with_llm(text: str, llm, cfg: Settings) -> str:
    """LLM classifies a page into one of the profile's document types."""
    snippet = text[: cfg.classify_snippet_chars].strip()
    if len(snippet) < 20:
        return "Unknown"

    categories = "\n".join(f"- {c}" for c in cfg.profile.llm_categories)
    prompt = f"""Classify this document page into ONE category:
{categories}

Respond with ONLY the category name.

Page content:
\"\"\"
{snippet}
\"\"\"

Category:"""
    try:
        response = llm.complete(prompt)
        label = str(response).strip().split("\n")[0].strip().strip("\"'.").strip()
        return label if len(label) < 50 else "Other"
    except Exception:
        return "Unknown"


def classify_pages(pages: list[dict], llm, cfg: Settings) -> list[dict]:
    """Classify each document once: heuristic first, LLM fallback.

    Pages are grouped by filename, preserving order. The page with the
    lowest page_number in each group is classified, and every page in the
    group inherits that label.
    # ponytail: a PDF that bundles several document types gets one label;
    # split per section if the corpus ever contains loan packages.
    """
    groups: dict[str, list[dict]] = {}
    for page in pages:
        groups.setdefault(page["filename"], []).append(page)

    for group in groups.values():
        first_page = min(group, key=lambda p: p["page_number"])
        label = classify_doc_type_heuristic(first_page["text"], cfg)
        if (
            label == "Unknown"
            and cfg.use_llm_classification
            and len(first_page["text"].strip()) > 20
        ):
            label = classify_page_with_llm(first_page["text"], llm, cfg)
        for page in group:
            page["doc_type"] = label
    return pages
