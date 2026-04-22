from __future__ import annotations

from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    queries: list[str] = Field(..., min_length=1, max_length=50)
    k: int = Field(5, ge=1, le=50)
    page_limit: int = Field(7, ge=1, le=20)
    section_limit_per_page: int = Field(15, ge=1, le=50)


class SectionHitOut(BaseModel):
    title: str
    pageid: int
    url: str
    section_title: str
    text: str
    bm25: float
    dense: float
    score: float
    best_chunk: str


class RetrieveResponse(BaseModel):
    results: list[list[SectionHitOut]]
