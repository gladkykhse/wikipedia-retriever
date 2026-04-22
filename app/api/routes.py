from __future__ import annotations

import logging
from dataclasses import asdict

from fastapi import APIRouter, Depends

from app.api.deps import get_retriever, require_bearer_token
from app.core.metrics import RETRIEVER_BATCHES_TOTAL, RETRIEVER_QUERIES_TOTAL
from app.retriever import WikipediaHybridSectionRetriever
from app.schemas import RetrieveRequest, RetrieveResponse, SectionHitOut

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    dependencies=[Depends(require_bearer_token)],
)
async def retrieve(
    body: RetrieveRequest,
    retriever: WikipediaHybridSectionRetriever = Depends(get_retriever),
) -> RetrieveResponse:
    RETRIEVER_BATCHES_TOTAL.inc()
    RETRIEVER_QUERIES_TOTAL.inc(len(body.queries))
    log.info(
        "retrieve_request",
        extra={
            "batch_size": len(body.queries),
            "k": body.k,
            "page_limit": body.page_limit,
            "section_limit_per_page": body.section_limit_per_page,
        },
    )

    results = await retriever.retrieve(
        queries=body.queries,
        k=body.k,
        page_limit=body.page_limit,
        section_limit_per_page=body.section_limit_per_page,
    )
    return RetrieveResponse(
        results=[[SectionHitOut(**asdict(h)) for h in hits] for hits in results]
    )
