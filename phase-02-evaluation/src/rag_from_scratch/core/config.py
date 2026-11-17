"""Settings read from environment variables or a .env file (pydantic-settings)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    chat_model: str = "gpt-4o-mini"
    # the judge grades the answers in the evaluation; a stronger model judges better
    judge_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    opensearch_url: str = "http://localhost:9200"
    index_name: str = "rag-chunks"

    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # values come from the environment
