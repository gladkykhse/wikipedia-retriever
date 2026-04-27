# syntax=docker/dockerfile:1.9

ARG PYTHON_IMAGE=python:3.14-slim

# ---------------- build stage ----------------
FROM ${PYTHON_IMAGE} AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_PYTHON_PREFERENCE=only-system \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /usr/local/bin/

WORKDIR /app

# Install deps first (separate layer for better caching on code-only changes).
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-group tools --no-install-project

COPY app ./app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-group tools

# ---------------- model-fetch stage ----------------
# Pre-downloads the embedding model and NLTK corpora so the runtime container
# needs no outbound internet access.
FROM builder AS model-fetch

ENV HF_HOME=/models \
    NLTK_DATA=/nltk_data \
    TRANSFORMERS_OFFLINE=0 \
    HF_HUB_OFFLINE=0

RUN mkdir -p /models /nltk_data && \
    /app/.venv/bin/python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1'); \
import nltk; nltk.download('stopwords', download_dir='/nltk_data', quiet=True)"

# ---------------- runtime stage ----------------
FROM ${PYTHON_IMAGE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH=/app/.venv/bin:$PATH \
    HF_HOME=/models \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    NLTK_DATA=/nltk_data \
    TOKENIZERS_PARALLELISM=true \
    PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus

RUN useradd --system --uid 10001 --home /home/app --create-home app && \
    mkdir -p /app && \
    chown -R app:app /app /home/app

WORKDIR /app

COPY --from=builder  --chown=app:app /app/.venv /app/.venv
COPY --from=model-fetch --chown=app:app /models /models
COPY --from=model-fetch --chown=app:app /nltk_data /nltk_data
COPY --chown=app:app app ./app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3).read(); sys.exit(0)" || exit 1

CMD ["sh", "-c", "mkdir -p ${PROMETHEUS_MULTIPROC_DIR} && uvicorn app.main:app \
     --host 0.0.0.0 --port 8000 \
     --loop uvloop --http httptools \
     --timeout-graceful-shutdown 20 \
     --workers ${RETRIEVER_WORKERS:-1}"]
