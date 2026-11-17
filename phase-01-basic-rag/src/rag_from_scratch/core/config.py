"""Settings read from environment variables or a .env file (pydantic-settings)."""

from functools import lru_cache
from typing import Literal, Self

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # which LLM provider to use; only the key of that provider is required
    llm_provider: Literal["gemini", "openai"] = "gemini"
    gemini_api_key: str | None = None
    openai_api_key: str | None = None

    # defaults are for Gemini; .env.example shows the OpenAI values
    chat_model: str = "gemini-3.5-flash-lite"
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 1536

    opensearch_url: str = "http://localhost:9200"
    index_name: str = "rag-chunks"

    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 5

    @model_validator(mode="after")
    def check_api_key(self) -> Self:
        key = self.gemini_api_key if self.llm_provider == "gemini" else self.openai_api_key
        if not key:
            raise ValueError(f"{self.llm_provider.upper()}_API_KEY is required")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
