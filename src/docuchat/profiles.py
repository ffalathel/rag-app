"""Domain profiles: doc-type taxonomy, section patterns, and answer prompts.

Every mortgage-specific value ported from the notebook lives here, and only
here. Settings.profile resolves the active DomainProfile through PROFILES.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainProfile:
    name: str
    doc_type_keywords: tuple[tuple[str, str], ...]
    llm_categories: tuple[str, ...]
    section_patterns: tuple[str, ...]
    answer_system_prompt: str


# Ported verbatim from complete_mortgage_rag_pipeline.py:
#   classify_doc_type_heuristic (line 408) — order is load-bearing.
_MORTGAGE_DOC_TYPE_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("closing disclosure", "Closing Disclosure"),
    ("loan estimate", "Loan Estimate"),
    ("promissory note", "Promissory Note"),
    ("deed of trust", "Deed of Trust"),
    ("mortgage agreement", "Mortgage Agreement"),
    ("loan agreement", "Loan Agreement"),
    ("appraisal", "Appraisal"),
    ("title insurance", "Title Insurance"),
    ("homeowners insurance", "Homeowners Insurance"),
    ("pay stub", "Pay Stub"),
    ("paystub", "Pay Stub"),
    ("bank statement", "Bank Statement"),
    ("w-2", "Tax Document (W-2)"),
    ("1099", "Tax Document (1099)"),
    ("tax return", "Tax Return"),
    ("escrow", "Escrow Statement"),
    ("agreement", "Contract/Agreement"),
    ("contract", "Contract/Agreement"),
    ("disclosure", "Disclosure"),
)

# classify_page_with_llm (line 428) prompt category list.
_MORTGAGE_LLM_CATEGORIES: tuple[str, ...] = (
    "Loan Agreement",
    "Closing Disclosure",
    "Promissory Note",
    "Deed of Trust",
    "Loan Estimate",
    "Appraisal",
    "Title Insurance",
    "Homeowners Insurance",
    "Pay Stub",
    "Bank Statement",
    "Tax Document",
    "Escrow Statement",
    "Cover Letter / Correspondence",
    "Other",
)

# SECTION_PATTERNS (line 320).
_MORTGAGE_SECTION_PATTERNS: tuple[str, ...] = (
    r"(?:^|\n)(?:SECTION|Section|ARTICLE|Article)\s+\d+[.:]?\s+[A-Z]",
    r"(?:^|\n)\d{1,2}\.\s+[A-Z][A-Za-z]",
    r"(?:^|\n)([A-Z][A-Z\s]{5,})(?:\n|$)",
    r"(?:^|\n)\([a-z]\)\s",
    r"(?:^|\n)\([ivx]+\)\s",
)

# ask_v3 prompt preamble (line 763).
_MORTGAGE_ANSWER_PROMPT = (
    'You are a mortgage document analysis assistant. Answer ONLY from the '
    'provided context.\n'
    'If the answer is not in the context, say "I cannot find this '
    'information in the provided documents."\n'
    "Be precise with financial figures, dates, and legal terms. Cite "
    "sources with filename and page number."
)

_GENERIC_LLM_CATEGORIES: tuple[str, ...] = (
    "Contract",
    "Invoice",
    "Report",
    "Statement",
    "Form",
    "Correspondence",
    "Academic Paper",
    "Other",
)

# Structural heading patterns only: numbered headings and all-caps lines.
# Drops the mortgage set's SECTION/ARTICLE heading and legal (a)/(iv)
# sub-clause patterns, which are domain vocabulary, not structure.
_GENERIC_SECTION_PATTERNS: tuple[str, ...] = (
    r"(?:^|\n)\d{1,2}\.\s+[A-Z][A-Za-z]",
    r"(?:^|\n)([A-Z][A-Z\s]{5,})(?:\n|$)",
)

_GENERIC_ANSWER_PROMPT = (
    "You are a document analysis assistant. Answer ONLY from the provided "
    "context.\n"
    'If the answer is not in the context, say "I cannot find this '
    'information in the provided documents."\n'
    "Be precise with figures, dates, and names. Cite sources with filename "
    "and page number."
)

PROFILES: dict[str, DomainProfile] = {
    "mortgage": DomainProfile(
        name="mortgage",
        doc_type_keywords=_MORTGAGE_DOC_TYPE_KEYWORDS,
        llm_categories=_MORTGAGE_LLM_CATEGORIES,
        section_patterns=_MORTGAGE_SECTION_PATTERNS,
        answer_system_prompt=_MORTGAGE_ANSWER_PROMPT,
    ),
    "generic": DomainProfile(
        name="generic",
        doc_type_keywords=(),
        llm_categories=_GENERIC_LLM_CATEGORIES,
        section_patterns=_GENERIC_SECTION_PATTERNS,
        answer_system_prompt=_GENERIC_ANSWER_PROMPT,
    ),
}
