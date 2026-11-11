"""Score trackdub-gpu-docs retrieval against tools/docs-eval/questions.json.

Run from tools/mcp-trackdub-gpu-docs so the package resolves:
    uv run python ../docs-eval/eval_gpu.py [--min-hit3 0.8]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from trackdub_gpu_docs_mcp.search import search_chunks

QUESTIONS = Path(__file__).resolve().with_name("questions.json")


def distinct_sources(hits: list[dict]) -> list[str]:
    seen: list[str] = []
    for hit in hits:
        if hit["sourceId"] not in seen:
            seen.append(hit["sourceId"])
    return seen


def score(questions: list[dict], limit: int = 8) -> dict:
    rows = []
    for item in questions:
        expect = item.get("gpu_expect")
        if not expect:
            continue
        ranked = distinct_sources(search_chunks(item["q"], limit=limit))
        rank = next((i + 1 for i, source in enumerate(ranked) if source in expect), None)
        rows.append({"q": item["q"], "rank": rank, "top": ranked[:3]})
    total = len(rows)

    def hit_at(k: int) -> float:
        return sum(1 for r in rows if r["rank"] is not None and r["rank"] <= k) / total if total else 0.0

    mrr = sum(1 / r["rank"] for r in rows if r["rank"]) / total if total else 0.0
    return {"total": total, "hit1": hit_at(1), "hit3": hit_at(3), "mrr": mrr, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-hit3", type=float, default=0.0, help="Exit nonzero when hit@3 is below this")
    parser.add_argument("--json", action="store_true", help="Print the full result as JSON")
    args = parser.parse_args()
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    result = score(questions)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for row in result["rows"]:
            mark = "ok  " if row["rank"] and row["rank"] <= 3 else "MISS"
            print(f"{mark} rank={row['rank']} {row['q'][:70]}  top={row['top']}")
        print(f"n={result['total']} hit@1={result['hit1']:.2f} hit@3={result['hit3']:.2f} mrr={result['mrr']:.3f}")
    return 0 if result["hit3"] >= args.min_hit3 else 1


if __name__ == "__main__":
    sys.exit(main())
