from __future__ import annotations

import time

from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests processed by the retriever service.",
    labelnames=("endpoint", "method", "status"),
)

# Buckets tuned against measured retriever latency:
#   /health          ~0.2 ms          → 0.05 bucket captures it
#   POST /retrieve   single-query     → p50=6 s, p95=8.5 s  (async, 7 parallel fetches)
#   POST /retrieve   batch of 8       → ~48 s               (CPU embed dominates)
#   POST /retrieve   batch of 50 max  → projected ~3 min worst case
# Sub-second buckets for health/errors, 5-15 s band for typical single-query traffic,
# 30-120 s tail for large batches.
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds.",
    labelnames=("endpoint", "method"),
    buckets=(0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 7.5, 10.0, 15.0, 30.0, 60.0, 120.0),
)

RETRIEVER_QUERIES_TOTAL = Counter(
    "retriever_queries_total",
    "Total queries processed (one increment per query in a batch).",
)

RETRIEVER_BATCHES_TOTAL = Counter(
    "retriever_batches_total",
    "Total /retrieve batches processed.",
)

RETRIEVER_PAGES_FETCHED_TOTAL = Counter(
    "retriever_pages_fetched_total",
    "Total Wikipedia page fetches (after dedup).",
)

RETRIEVER_CACHE_HITS_TOTAL = Counter(
    "retriever_cache_hits_total",
    "Page-fetch requests satisfied by the batch-local dedup cache.",
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Records request count and latency for every route except `/metrics`."""

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        if path.startswith("/metrics"):
            return await call_next(request)

        method = request.method
        start = time.perf_counter()
        status = "500"
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            elapsed = time.perf_counter() - start
            HTTP_REQUESTS_TOTAL.labels(endpoint=path, method=method, status=status).inc()
            HTTP_REQUEST_DURATION_SECONDS.labels(endpoint=path, method=method).observe(elapsed)
