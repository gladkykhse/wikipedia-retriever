import logging
import os

import httpx
from langchain_core.tools import tool

log = logging.getLogger(__name__)

_URL = os.getenv("RETRIEVER_URL", "http://127.0.0.1:8000").rstrip("/") + "/retrieve"
_TOKEN = os.getenv("RETRIEVER_API_TOKEN", "")

_client = httpx.Client(
    headers={"Authorization": f"Bearer {_TOKEN}"},
    timeout=120.0,
    transport=httpx.HTTPTransport(retries=3),
)


def _retrieve(queries: list[str], k: int, page_limit: int, section_limit_per_page: int) -> list[list[dict]]:
    if not _TOKEN:
        raise RuntimeError("RETRIEVER_API_TOKEN env var is not set")
    resp = _client.post(
        _URL,
        json={"queries": queries, "k": k, "page_limit": page_limit, "section_limit_per_page": section_limit_per_page},
    )
    resp.raise_for_status()
    return resp.json()["results"]


@tool
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for physics concepts, laws, constants, or named theorems.

    Args:
        query: Concise entity-based search term (2-6 words).

    Returns:
        Relevant text passages from matching Wikipedia sections.
    """
    try:
        hits = _retrieve([query], k=3, page_limit=5, section_limit_per_page=10)[0]
        output = [
            f"--- Article: {h['title']} | Section: {h['section_title']} ---\n{h['text']}"
            for h in hits
        ]
        res = "\n\n".join(output)
        log.info("[WIKI_SEARCH] Query: %r | Found %d sections.", query, len(hits))
        return res
    except Exception as e:
        log.error("[WIKI_SEARCH] Error: %s", e)
        return f"Search failed due to an error: {e}"


@tool
def wikipedia_multi_search(queries: list[str]) -> str:
    """
    Performs multiple Wikipedia searches simultaneously.

    Args:
        queries: A list of concise entity-based search terms.

    Returns:
        A structured report with unique sections grouped by query.
    """
    seen: set[tuple[int, str]] = set()
    output = ["# Wikipedia Search Results\n"]
    try:
        results = _retrieve(queries, k=3, page_limit=5, section_limit_per_page=10)
        for query, hits in zip(queries, results):
            fresh = []
            for h in hits:
                key = (h["pageid"], h["section_title"])
                if key not in seen:
                    seen.add(key)
                    fresh.append(h)
            if not fresh:
                continue
            output.append(f"## Query: {query}")
            for i, h in enumerate(fresh, 1):
                output.append(f"### Section {i}: {h['title']} - {h['section_title']}")
                output.append(f"{h['text']}\n")
                output.append("---\n")
        if not seen:
            return "No relevant Wikipedia sections found for the provided queries."
        res = "\n".join(output)
        log.info("[MULTI_WIKI_SEARCH] Queries: %s | Found %d unique sections.", queries, len(seen))
        return res
    except Exception as e:
        log.error("[MULTI_WIKI_SEARCH] Error: %s", e)
        return f"Search failed due to an error: {e}"
