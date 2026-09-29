"""Page classification: keyword heuristic first, LLM as fallback.

Ported from complete_mortgage_rag_pipeline.py (lines 408-475), with the
keyword list and LLM category list read from the active domain profile
instead of hardcoded here, and the LLM snippet length read from Settings.
The keyword list is iterated in order -- ordering is load-bearing, since
"disclosure" must not shadow "closing disclosure".
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
    """Classify every page: heuristic first, LLM fallback."""
    for page in pages:
        label = classify_doc_type_heuristic(page["text"], cfg)
        if label == "Unknown" and cfg.use_llm_classification and len(page["text"].strip()) > 20:
            label = classify_page_with_llm(page["text"], llm, cfg)
        page["doc_type"] = label
    return pages
