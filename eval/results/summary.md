# Evaluation Summary

- Date: 2026-09-30
- Commit: dfe5fce
- Questions: 43

## Arms

| arm | hit | recall | mrr | candidate_recall | correct_rate | mean_correctness | faithful_rate | correct_refusal_rate | false_refusal_rate | llm_calls | errors | judge_errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.447 [0.289, 0.605] | 0.417 [0.263, 0.566] | 0.322 [0.200, 0.456] | 0.561 [0.421, 0.706] | n/a | n/a | n/a | n/a | n/a | 1.000 [1.000, 1.000] | 0.000 | 0.000 |
| +hybrid | 0.474 [0.316, 0.605] | 0.443 [0.289, 0.583] | 0.296 [0.174, 0.429] | 0.557 [0.399, 0.697] | n/a | n/a | n/a | n/a | n/a | 1.000 [1.000, 1.000] | 0.000 | 0.000 |
| +rerank | 0.553 [0.395, 0.685] | 0.509 [0.355, 0.658] | 0.461 [0.320, 0.592] | 0.557 [0.399, 0.697] | n/a | n/a | n/a | n/a | n/a | 1.000 [1.000, 1.000] | 0.000 | 0.000 |

## vs previous arm

| arm | metric | wins | losses | ties |
|---|---|---|---|---|
| +hybrid vs baseline | recall | 2 | 1 | 35 |
| +rerank vs +hybrid | recall | 4 | 1 | 33 |

## Skipped arms

(none)

## Unreachable evidence

(none)
