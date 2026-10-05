"""Latency benchmark: p50/p95 per pipeline stage against a running docuchat.

    python -m docuchat.bench --url https://<space>.hf.space --key KEY

An HTTP client, because free Spaces have no shell. Runs every question in
eval/questions.yaml sequentially in one session. The first request is
reported separately as cold and excluded from the percentiles. Each run
spends len(questions) of the server's daily query cap.
"""

import argparse
import datetime
import math
import time
from pathlib import Path

import httpx

from docuchat.evaluate import QUESTIONS_PATH, RESULTS_DIR, _git_commit, load_questions
from docuchat.service import SERVED_ARM

STAGES = ("rewrite", "decompose", "retrieve", "rerank", "clean_context", "llm", "total", "client")


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile; defined for any non-empty list."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def summarize(samples: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    warm = samples[1:]
    summary = {}
    for stage in STAGES:
        values = [s[stage] for s in warm if stage in s]
        if values:
            summary[stage] = (percentile(values, 50), percentile(values, 95))
    return summary


def run(url: str, key: str | None, questions: list[dict], transport=None) -> list[dict[str, float]]:
    # The key rides on every request (the gate accepts it on POSTs), so the
    # bench never depends on the Secure cookie, which plain-http localhost drops.
    params = {"key": key} if key else None
    samples = []
    with httpx.Client(base_url=url, params=params, timeout=300, transport=transport) as client:
        session_id = client.post("/api/sessions").raise_for_status().json()["session_id"]
        for q in questions:
            start = time.perf_counter()
            response = client.post("/api/ask", json={"session_id": session_id, "query": q["question"]})
            response.raise_for_status()
            sample = dict(response.json()["timings"])
            sample["client"] = (time.perf_counter() - start) * 1000
            samples.append(sample)
    return samples


def render(url: str, samples: list[dict[str, float]], commit: str, date: str) -> str:
    lines = [
        "# Latency",
        "",
        f"- URL: {url}",
        f"- Date: {date}",
        f"- Commit: {commit}",
        f"- Arm: {SERVED_ARM}",
        f"- Requests: 1 cold + {len(samples) - 1} warm, sequential",
        "- `client` is end-to-end from the benchmark machine, network included; "
        "every other row is server-side.",
        "",
        "| Stage | p50 (ms) | p95 (ms) |",
        "|---|---|---|",
    ]
    lines += [f"| {stage} | {p50:.0f} | {p95:.0f} |" for stage, (p50, p95) in summarize(samples).items()]
    cold = samples[0]
    lines += ["", f"First request (cold): {cold['client']:.0f} ms client-side, "
                  f"{cold.get('total', 0):.0f} ms server-side."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m docuchat.bench")
    parser.add_argument("--url", required=True)
    parser.add_argument("--key")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "latency.md")
    args = parser.parse_args(argv)
    samples = run(args.url, args.key, load_questions(QUESTIONS_PATH))
    report = render(args.url, samples, _git_commit(), datetime.date.today().isoformat())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
