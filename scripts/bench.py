import argparse
import asyncio
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.retriever import WikipediaHybridSectionRetriever  # noqa: E402


def _percentiles(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "mean": float(np.mean(samples)),
        "p50": float(np.percentile(samples, 50)),
        "p95": float(np.percentile(samples, 95)),
        "p99": float(np.percentile(samples, 99)),
    }


def _fmt(stats: dict[str, float]) -> str:
    return (
        f"mean={stats['mean']:.2f}s  "
        f"p50={stats['p50']:.2f}s  "
        f"p95={stats['p95']:.2f}s  "
        f"p99={stats['p99']:.2f}s"
    )


async def bench_single(retriever: WikipediaHybridSectionRetriever, queries: list[str]) -> list[float]:
    times: list[float] = []
    for q in queries:
        t = time.perf_counter()
        await retriever.retrieve([q])
        times.append(time.perf_counter() - t)
    return times


async def bench_batch(retriever: WikipediaHybridSectionRetriever, queries: list[str]) -> float:
    t = time.perf_counter()
    await retriever.retrieve(queries)
    return time.perf_counter() - t


async def run(args: argparse.Namespace) -> None:
    queries_path = ROOT / "queries.txt"
    all_queries = [q.strip() for q in queries_path.read_text().splitlines() if q.strip()]
    rng = random.Random(args.seed)
    sample = rng.sample(all_queries, min(args.sample, len(all_queries)))
    print(f"loaded {len(all_queries)} queries; sampled {len(sample)} (seed={args.seed})", flush=True)

    t0 = time.perf_counter()
    retriever = WikipediaHybridSectionRetriever()
    print(f"model load: {time.perf_counter() - t0:.2f}s", flush=True)

    # Warm-up
    warmup = sample[:2]
    print("warmup...", flush=True)
    for q in warmup:
        await retriever.retrieve([q])
    print("warmup done", flush=True)

    bench_queries = sample[2:] if len(sample) > 2 else sample

    try:
        if args.mode in ("single", "both"):
            print(f"\n== SINGLE (n={len(bench_queries)}, sequential awaits) ==", flush=True)
            t_wall = time.perf_counter()
            times = await bench_single(retriever, bench_queries)
            total = time.perf_counter() - t_wall
            stats = _percentiles(times)
            print(f"  per-query: {_fmt(stats)}")
            print(f"  total wall-clock: {total:.2f}s")
            print(f"  throughput: {len(bench_queries) / total:.2f} q/s")

        if args.mode in ("batch", "both"):
            print(f"\n== BATCH (n={len(bench_queries)}, one call) ==", flush=True)
            # Run the batch 3 times; report per-query amortised
            batch_totals: list[float] = []
            for i in range(args.batch_runs):
                total = await bench_batch(retriever, bench_queries)
                batch_totals.append(total)
                print(f"  run {i + 1}/{args.batch_runs}: total={total:.2f}s  ({total/len(bench_queries):.2f}s/q)")
            avg = statistics.mean(batch_totals)
            print(f"  average batch total: {avg:.2f}s  ({avg/len(bench_queries):.2f}s/q amortised)")
    finally:
        await retriever.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("single", "batch", "both"), default="both")
    parser.add_argument("--batch-runs", type=int, default=1)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
