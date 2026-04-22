# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for dependency management (Python 3.14).

```bash
# Install dependencies
uv sync

# Run the retriever interactively
uv run python -c "
from src.wikipedia import WikipediaHybridSectionRetriever
r = WikipediaHybridSectionRetriever()
hits = r.retrieve('your query here', k=5)
for h in hits:
    print(h.title, '/', h.section_title)
    print(h.best_chunk[:200])
"

# Add a dependency
uv add <package>
```

There is no test suite, linter config, or build step currently.

## Architecture

`src/wikipedia.py` is the entire implementation. It exposes one public class and one dataclass:

**`WikipediaHybridSectionRetriever`** — takes a query string and returns ranked `SectionHit` dataclasses from Wikipedia.

**`SectionHit`** — result dataclass with fields: `title`, `pageid`, `url`, `section_title`, `text` (full section), `best_chunk` (highest-scoring excerpt), `bm25`, `dense`, `score`.

### Pipeline (called via `.retrieve(query, k, ...)`)

1. **Search** — `_search_pages` calls the Wikipedia search API to get candidate page titles and IDs.
2. **Fetch** — `_fetch_page_html` downloads each page's full HTML via the Wikipedia parse API, with exponential backoff on 429s.
3. **Split** — `_split_html_into_sections` splits the page HTML into `(title, body_html)` pairs using `<h2>`–`<h6>` heading tags. The heading tag itself is excluded from each section's body (only the content after the heading is kept). The text before the first heading is returned as the `"Lead"` section.
4. **Clean** — `_html_to_text` strips navboxes, infoboxes, captions, references, and edit-section artifacts; converts `<math>` tags to inline LaTeX; normalises whitespace.
5. **Filter** — sections shorter than 50 characters or matching a blacklist (References, See Also, External Links, etc.) are dropped.
6. **Chunk** — `_chunk_text` calls `_split_by_tokens` on the full section text as a single sliding window. Sections that fit within the token limit are returned as a single chunk; longer sections are split with a configurable overlap.
7. **Score** — `_score_sections` embeds all chunks with the dense model and scores them with BM25. Per-section scores are the max over all its chunks. Both scores are min-max normalised and fused with configurable weights. The highest-scoring chunk per section is stored as `best_chunk`.
8. **Return** — sections sorted by fused score, top-k returned.

### Key defaults

| Parameter | Default | Constructor arg |
|---|---|---|
| Embedding model | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` | `emb_model_name` |
| BM25 / dense fusion weights | 0.3 / 0.7 | `bm25_weight`, `dense_weight` |
| Chunk overlap | 16 tokens | `chunk_overlap` |
| Pages searched per query | 7 | `page_limit` in `.retrieve()` |
| Sections per page | 15 (first N by position) | `section_limit_per_page` in `.retrieve()` |
| Results returned | 5 | `k` in `.retrieve()` |
| Wikipedia language | `en` | `lang` |

### Known limitations

- Pages are fetched sequentially; latency scales linearly with `page_limit`.
- `section_limit_per_page` always takes the first N sections by document order — relevant sections late in a long article may be missed.
- No deduplication: multiple results from the same article can appear in top-k.
