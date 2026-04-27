# Wikipedia Retriever

Fast Wikipedia retrieval for LLM agent tools.

## Description / Overview

Wikipedia Retriever is a FastAPI service that searches Wikipedia, fetches matching pages, extracts useful sections, and ranks them with a hybrid BM25 + dense embedding scorer. It is built for LLM agents and tool-calling workflows that need concise, relevant Wikipedia context over HTTP or through the included Python tool wrapper.

## Installation

Requirements:

- [git](https://git-scm.com/) - Version control system
- [uv](https://docs.astral.sh/uv/) - Python package and project manager

Set up the project locally:

```bash
git clone <repo-url>
cd wikipedia-retriever
uv sync
cp .env.example .env
```

The default `.env.example` is ready for localhost development. Before exposing the service beyond your machine, replace `RETRIEVER_API_TOKEN` with a strong token:

```bash
openssl rand -hex 32
```

Optional Docker setup:

```bash
docker build -t wikipedia-retriever .
docker run --env-file .env -p 8000:8000 wikipedia-retriever
```

## Usage

Start the API locally:

```bash
uv run uvicorn app.main:app --reload
```

The service listens on `http://127.0.0.1:8000`.

Check health:

```bash
curl http://127.0.0.1:8000/health
```

Query the API with the helper script:

```bash
uv run python scripts/query.py "black holes" --token change-me-local-token
```

Send a raw HTTP request:

```bash
curl -X POST http://127.0.0.1:8000/retrieve \
  -H "Authorization: Bearer change-me-local-token" \
  -H "Content-Type: application/json" \
  -d '{"queries":["black holes"],"k":3,"page_limit":5,"section_limit_per_page":10}'
```

Use the optional LLM-agent tools by installing the tools dependency group:

```bash
uv sync --group tools
```

Set the tool environment in the process that runs your agent:

```bash
export RETRIEVER_URL=http://127.0.0.1:8000
export RETRIEVER_API_TOKEN=change-me-local-token
```

Then import `wikipedia_search` or `wikipedia_multi_search` from `tools.wikipedia` and register them with your LLM agent framework.

## Features

- HTTP API with `/retrieve`, `/health`, and `/metrics` endpoints.
- Bearer-token protection for retrieval requests.
- Concurrent Wikipedia search and page fetching with retry/backoff.
- Section extraction and cleanup from Wikipedia HTML.
- Hybrid ranking with BM25 and SentenceTransformer dense embeddings.
- Batch query support with per-request page deduplication.
- JSON logging, request IDs, and Prometheus metrics.
- Local script client and optional LLM-agent tool wrappers.

## Tech Stack / Built With

- Python 3.14
- FastAPI and Uvicorn
- HTTPX with HTTP/2 support
- Pydantic Settings
- BeautifulSoup
- rank-bm25
- SentenceTransformers, PyTorch, NumPy
- NLTK
- Prometheus client
- uv
- Docker
