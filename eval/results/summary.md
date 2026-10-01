# Evaluation Summary

- Date: 2026-10-01
- Commit: 84c415f
- Questions: 43

## Arms

| arm | hit | recall | mrr | candidate_recall | correct_rate | mean_correctness | faithful_rate | correct_refusal_rate | false_refusal_rate | llm_calls | errors | judge_errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.632 [0.474, 0.789] | 0.575 [0.430, 0.719] | 0.355 [0.234, 0.483] | 0.706 [0.579, 0.820] | n/a | n/a | n/a | n/a | n/a | 1.000 [1.000, 1.000] | 0.000 | 0.000 |
| +hybrid | 0.605 [0.474, 0.737] | 0.583 [0.439, 0.724] | 0.315 [0.203, 0.448] | 0.829 [0.724, 0.921] | n/a | n/a | n/a | n/a | n/a | 1.000 [1.000, 1.000] | 0.000 | 0.000 |
| +rerank | 0.763 [0.632, 0.895] | 0.715 [0.583, 0.847] | 0.512 [0.379, 0.634] | 0.829 [0.724, 0.921] | n/a | n/a | n/a | n/a | n/a | 1.000 [1.000, 1.000] | 0.000 | 0.000 |

## vs previous arm

| arm | metric | wins | losses | ties |
|---|---|---|---|---|
| +hybrid vs baseline | recall | 3 | 3 | 32 |
| +rerank vs +hybrid | recall | 8 | 2 | 28 |

## Skipped arms

(none)

## Unreachable evidence

(none)
