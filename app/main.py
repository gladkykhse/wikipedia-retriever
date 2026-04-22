from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from ulid import ULID

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.core.logging import configure_logging, request_id_var
from app.core.metrics import (
    RETRIEVER_CACHE_HITS_TOTAL,
    RETRIEVER_PAGES_FETCHED_TOTAL,
    PrometheusMiddleware,
)
from app.retriever import WikipediaHybridSectionRetriever

log = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(level=settings.log_level, log_file=settings.log_file)
    log.info("startup_begin", extra={"embedding_model": settings.embedding_model})

    headers = {
        "User-Agent": f"{settings.user_agent} (gladkykh.sviatoslav@gmail.com; AI Research Bot)",
    }
    client = httpx.AsyncClient(
        timeout=20.0,
        http2=True,
        limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
        headers=headers,
    )
    retriever = WikipediaHybridSectionRetriever(
        emb_model_name=settings.embedding_model,
        user_agent=settings.user_agent,
        lang=settings.lang,
        client=client,
        http_concurrency=settings.http_concurrency,
        on_page_fetched=RETRIEVER_PAGES_FETCHED_TOTAL.inc,
        on_cache_hit=RETRIEVER_CACHE_HITS_TOTAL.inc,
    )
    app.state.client = client
    app.state.retriever = retriever

    log.info("startup_complete")
    try:
        yield
    finally:
        log.info("shutdown_begin")
        await client.aclose()
        log.info("shutdown_complete")


app = FastAPI(title="wikipedia-retriever", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(ULID())
    token = request_id_var.set(rid)
    request.state.request_id = rid
    start = time.perf_counter()
    try:
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled_exception", extra={"path": request.url.path})
            raise
        latency_ms = (time.perf_counter() - start) * 1000.0
        log.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round(latency_ms, 2),
            },
        )
        response.headers["x-request-id"] = rid
        return response
    finally:
        request_id_var.reset(token)


app.add_middleware(PrometheusMiddleware)


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


app.include_router(api_router)
