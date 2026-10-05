# Latency

- URL: https://redesigned-sniffle-7jvqgj6qp7vcx97g-7860.app.github.dev
- Date: 2026-10-05
- Commit: cc9cd70
- Arm: +rerank
- Requests: 1 cold + 3 warm, sequential; 4 failed and excluded
- Host: a GitHub Codespace (4-core CPU, 16 GB), a temporary demo host, reached through its port-forwarding proxy. LLM: gemini-3.7-flash (free tier).
- Reranker overridden for CPU: `DOCUCHAT_CROSS_ENCODER_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2`, `DOCUCHAT_CROSS_ENCODER_MAX_LENGTH=512`. The default bge-reranker-v2-m3 took 125-130 s per question on this CPU. The evaluated answer quality is for the default reranker.
- Failed requests are Gemini free-tier errors (busy or out of daily quota), not app errors.
- `client` is end-to-end from the benchmark machine, network included; every other row is server-side.

| Stage | p50 (ms) | p95 (ms) |
|---|---|---|
| retrieve | 35 | 36 |
| rerank | 3280 | 3309 |
| clean_context | 3090 | 3119 |
| llm | 12402 | 19130 |
| total | 18863 | 25535 |
| client | 18945 | 25705 |

First request (cold): 14233 ms client-side, 14068 ms server-side.
