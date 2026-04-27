from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RETRIEVER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_token: SecretStr = Field(
        ..., description="Bearer token required on /retrieve. Generate via `openssl rand -hex 32`."
    )
    log_level: str = "INFO"
    log_file: Optional[Path] = None

    embedding_model: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    user_agent: str = "WikipediaHybridSectionRetriever/0.1.0"
    http_concurrency: int = 8
    lang: str = "en"

    max_batch_size: int = 50
    max_query_length: int = 500


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
