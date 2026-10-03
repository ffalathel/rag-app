# Evaluation Summary (judged)

- Date: 2026-10-03
- Commit: ae1aaa2
- Questions: 43
- Answer model: Qwen3 8B (Q4_K_M GGUF, llama.cpp) · Judge model: Qwen3 14B (Q4_K_M GGUF, llama.cpp)
- Run on a Kaggle T4 GPU via `eval/kaggle_eval.ipynb`. Per-record JSON was not kept, so this judge has
  not been validated against human labels (`judge-sample`), and RAGAS was not run.

## Arms

| arm | hit | recall | mrr | candidate_recall | correct_rate | mean_correctness | faithful_rate | correct_refusal_rate | false_refusal_rate | llm_calls | errors | judge_errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.632 [0.474, 0.789] | 0.575 [0.430, 0.719] | 0.355 [0.234, 0.483] | 0.706 [0.579, 0.820] | 0.698 [0.558, 0.814] | 1.651 [1.488, 1.814] | 0.744 [0.628, 0.860] | 1.000 [1.000, 1.000] | 0.211 [0.079, 0.342] | 1.000 [1.000, 1.000] | 0.000 | 0.000 |
| +hybrid | 0.605 [0.474, 0.737] | 0.583 [0.439, 0.724] | 0.315 [0.203, 0.448] | 0.829 [0.724, 0.921] | 0.698 [0.558, 0.814] | 1.558 [1.326, 1.744] | 0.744 [0.605, 0.860] | 1.000 [1.000, 1.000] | 0.158 [0.053, 0.263] | 1.000 [1.000, 1.000] | 0.000 | 0.000 |
| +rerank | 0.763 [0.632, 0.895] | 0.715 [0.583, 0.847] | 0.512 [0.379, 0.634] | 0.829 [0.724, 0.921] | 0.674 [0.535, 0.814] | 1.581 [1.372, 1.767] | 0.698 [0.558, 0.837] | 1.000 [1.000, 1.000] | 0.105 [0.026, 0.211] | 1.000 [1.000, 1.000] | 0.000 | 0.000 |
| full | 0.789 [0.658, 0.921] | 0.741 [0.610, 0.873] | 0.538 [0.405, 0.660] | 0.961 [0.895, 1.000] | 0.698 [0.558, 0.837] | 1.628 [1.441, 1.791] | 0.721 [0.581, 0.860] | 1.000 [1.000, 1.000] | 0.079 [0.000, 0.184] | 3.000 [3.000, 3.000] | 0.000 | 0.000 |
| full/generic | 0.816 [0.684, 0.921] | 0.754 [0.627, 0.873] | 0.560 [0.436, 0.680] | 0.934 [0.868, 0.987] | 0.721 [0.581, 0.860] | 1.628 [1.419, 1.814] | 0.721 [0.581, 0.860] | 1.000 [1.000, 1.000] | 0.105 [0.026, 0.211] | 3.000 [3.000, 3.000] | 0.000 | 0.000 |
| full/embed-bge-base | 0.816 [0.684, 0.921] | 0.754 [0.627, 0.873] | 0.547 [0.427, 0.666] | 0.987 [0.961, 1.000] | 0.767 [0.628, 0.884] | 1.744 [1.581, 1.884] | 0.814 [0.674, 0.930] | 1.000 [1.000, 1.000] | 0.132 [0.026, 0.263] | 3.000 [3.000, 3.000] | 0.000 | 0.000 |

## vs previous arm

| arm | metric | wins | losses | ties |
|---|---|---|---|---|
| +hybrid vs baseline | recall | 3 | 3 | 32 |
| +hybrid vs baseline | correctness | 4 | 5 | 34 |
| +rerank vs +hybrid | recall | 8 | 2 | 28 |
| +rerank vs +hybrid | correctness | 5 | 5 | 33 |
| full vs +rerank | recall | 4 | 2 | 32 |
| full vs +rerank | correctness | 5 | 3 | 35 |
| full/generic vs full | recall | 1 | 1 | 36 |
| full/generic vs full | correctness | 3 | 3 | 37 |
| full/embed-bge-base vs full | recall | 1 | 1 | 36 |
| full/embed-bge-base vs full | correctness | 5 | 1 | 37 |
