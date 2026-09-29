"""LLM judge: grades a pipeline answer against a reference answer and
retrieved context, returning a structured verdict.

Parses the judge LLM's response by slicing from the first "{" to the last
"}" (tolerant of a fenced ```json block or a leading sentence) and loading
it as JSON. A malformed response is retried once; two failures in a row
give a judge_error result instead of raising.
"""

import json

UNANSWERABLE_REFERENCE = "UNANSWERABLE: the documents do not contain this information."

_PROMPT_TEMPLATE = """You are grading an answer produced by a document question-answering system.

Question: {question}

Reference answer: {reference_answer}

Retrieved context given to the system:
{contexts}

System answer: {answer}

Grade the system answer. Respond with ONLY a JSON object with these keys:
- "refused": true if the answer declines to answer or says the information is not available, else false.
- "correctness": 2 if it matches the reference answer's facts, 1 if partially correct or incomplete, 0 if wrong. If the reference is UNANSWERABLE, a refusal scores 2 and any substantive answer scores 0.
- "faithful": true if every factual claim in the answer is supported by the retrieved context (a refusal is faithful), else false.
- "rationale": one sentence explaining the grade."""


def _parse_verdict(response: str) -> dict | None:
    """Extract and validate the JSON verdict, or None if malformed."""
    text = str(response)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None
    refused = data.get("refused")
    correctness = data.get("correctness")
    faithful = data.get("faithful")
    rationale = data.get("rationale")
    if not isinstance(refused, bool):
        return None
    if not isinstance(correctness, int) or isinstance(correctness, bool) or correctness not in (0, 1, 2):
        return None
    if not isinstance(faithful, bool):
        return None
    if not isinstance(rationale, str):
        return None
    return {
        "refused": refused,
        "correctness": correctness,
        "faithful": faithful,
        "rationale": rationale,
    }


def judge_answer(
    question: str,
    reference_answer: str,
    contexts: list[str],
    answer: str,
    llm,
) -> dict:
    """Grade `answer` against `reference_answer` and `contexts` using `llm`.

    Retries once on a malformed response. Returns a judge_error dict if both
    attempts fail.
    """
    prompt = _PROMPT_TEMPLATE.format(
        question=question,
        reference_answer=reference_answer,
        contexts="\n\n---\n\n".join(contexts),
        answer=answer,
    )

    for attempt in range(2):
        try:
            response = str(llm.complete(prompt))
        except Exception as exc:
            if attempt == 1:
                return {"judge_error": str(exc)}
            continue
        verdict = _parse_verdict(response)
        if verdict is not None:
            return verdict

    return {"judge_error": "judge response was not valid JSON after retry"}
