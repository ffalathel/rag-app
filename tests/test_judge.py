from docuchat.judge import UNANSWERABLE_REFERENCE, judge_answer

VALID_JSON = '{"refused": false, "correctness": 2, "faithful": true, "rationale": "ok"}'


class _TwoResponseLLM:
    """Fake LLM returning successive responses from a list."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.prompts = []

    def complete(self, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        return self._responses[self.calls - 1]


def test_parses_bare_json(fake_llm_returning):
    llm = fake_llm_returning(VALID_JSON)
    result = judge_answer("q", "ref", ["ctx"], "answer", llm)
    assert result == {
        "refused": False,
        "correctness": 2,
        "faithful": True,
        "rationale": "ok",
    }
    assert llm.calls == 1


def test_parses_fenced_json_without_retry(fake_llm_returning):
    fenced = f"Here is my grading.\n```json\n{VALID_JSON}\n```"
    llm = fake_llm_returning(fenced)
    result = judge_answer("q", "ref", ["ctx"], "answer", llm)
    assert result == {
        "refused": False,
        "correctness": 2,
        "faithful": True,
        "rationale": "ok",
    }
    assert llm.calls == 1


def test_retries_once_then_succeeds():
    llm = _TwoResponseLLM(["not json", VALID_JSON])
    result = judge_answer("q", "ref", ["ctx"], "answer", llm)
    assert result == {
        "refused": False,
        "correctness": 2,
        "faithful": True,
        "rationale": "ok",
    }
    assert llm.calls == 2


def test_two_failures_give_judge_error():
    llm = _TwoResponseLLM(["not json", "not json"])
    result = judge_answer("q", "ref", ["ctx"], "answer", llm)
    assert "judge_error" in result
    assert llm.calls == 2


def test_out_of_range_correctness_is_a_failure():
    bad = '{"refused": false, "correctness": 3, "faithful": true, "rationale": "ok"}'
    llm = _TwoResponseLLM([bad, VALID_JSON])
    result = judge_answer("q", "ref", ["ctx"], "answer", llm)
    assert result == {
        "refused": False,
        "correctness": 2,
        "faithful": True,
        "rationale": "ok",
    }
    assert llm.calls == 2


def test_prompt_contains_all_inputs(fake_llm_returning):
    llm = fake_llm_returning(VALID_JSON)
    judge_answer(
        "What is the rate?",
        "The rate is 5%.",
        ["context one", "context two"],
        "The rate is five percent.",
        llm,
    )
    prompt = llm.prompts[0]
    assert "What is the rate?" in prompt
    assert "The rate is 5%." in prompt
    assert "context one" in prompt
    assert "context two" in prompt
    assert "The rate is five percent." in prompt


def test_unanswerable_reference_constant():
    assert UNANSWERABLE_REFERENCE == "UNANSWERABLE: the documents do not contain this information."
