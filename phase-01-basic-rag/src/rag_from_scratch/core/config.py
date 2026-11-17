"""Settings read from environment variables or a .env file (pydantic-settings)."""

from functools import lru_cache
from typing import Literal, Self

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["gemini", "openai", "ollama"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # provider for embeddings and answers
    llm_provider: ProviderName = "gemini"

    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    ollama_url: str = "http://localhost:11434"

    # defaults are for Gemini; .env.example shows the values for the other providers
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
        keys = {"gemini": self.gemini_api_key, "openai": self.openai_api_key}
        if self.llm_provider in keys and not keys[self.llm_provider]:
            raise ValueError(f"{self.llm_provider.upper()}_API_KEY is required")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
