"""Query the /retrieve API and pretty-print results.

Usage:
    uv run python scripts/query.py "Battle of Thermopylae" "Roman Empire"
    uv run python scripts/query.py "black holes" --k 3 --page-limit 5
    uv run python scripts/query.py "gravity" --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

import httpx


def print_hit(i: int, hit: dict) -> None:
    score_line = f"score={hit['score']:.3f}  bm25={hit['bm25']:.3f}  dense={hit['dense']:.3f}"
    print(f"  [{i}] {hit['title']} / {hit['section_title']}")
    print(f"       {score_line}")
    print(f"       {hit['url']}")
    chunk = textwrap.fill(hit["best_chunk"], width=100, initial_indent="       ", subsequent_indent="       ")
    print(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("queries", nargs="+", help="One or more search queries")
    parser.add_argument("--url", default=os.getenv("RETRIEVER_URL", "https://wiki.sgcore.dev"))
    parser.add_argument("--token", default=os.getenv("RETRIEVER_API_TOKEN"), help="Bearer token")
    parser.add_argument("--k", type=int, default=5, help="Results per query (default: 5)")
    parser.add_argument("--page-limit", type=int, default=7, help="Wikipedia pages to search (default: 7)")
    parser.add_argument("--section-limit", type=int, default=15, help="Sections per page (default: 15)")
    args = parser.parse_args()

    if not args.token:
        sys.exit("error: provide --token or set RETRIEVER_API_TOKEN")

    payload = {
        "queries": args.queries,
        "k": args.k,
        "page_limit": args.page_limit,
        "section_limit_per_page": args.section_limit,
    }

    with httpx.Client(timeout=300.0) as client:
        resp = client.post(
            f"{args.url.rstrip('/')}/retrieve",
            json=payload,
            headers={"Authorization": f"Bearer {args.token}"},
        )

    if resp.status_code != 200:
        sys.exit(f"error {resp.status_code}: {resp.text}")

    results = resp.json()["results"]
    for query, hits in zip(args.queries, results):
        print(f"\n{'='*60}")
        print(f"Query: {query!r}  ({len(hits)} results)")
        print(f"{'='*60}")
        for i, hit in enumerate(hits, 1):
            print_hit(i, hit)
            if i < len(hits):
                print()


if __name__ == "__main__":
    main()
