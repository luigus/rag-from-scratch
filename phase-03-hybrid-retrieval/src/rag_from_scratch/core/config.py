"""Settings read from environment variables or a .env file (pydantic-settings)."""

from functools import lru_cache
from typing import Literal, Self

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["gemini", "openai", "ollama"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # provider for the RAG itself (embeddings and answers)
    llm_provider: ProviderName = "gemini"
    # provider for the evaluation judge; empty = same as llm_provider.
    # The judge makes as many calls as the RAG, so running it locally
    # (ollama) avoids the rate limits of free API tiers.
    judge_provider: ProviderName | None = None

    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    ollama_url: str = "http://localhost:11434"

    # defaults are for Gemini; .env.example shows the values for the other providers
    chat_model: str = "gemini-3.5-flash-lite"
    judge_model: str = "gemini-3.8-flash"
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 1536

    opensearch_url: str = "http://localhost:9200"
    index_name: str = "rag-chunks"

    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k: int = 3

    # retrieval (phase 3)
    retrieval_mode: Literal["vector", "keyword", "hybrid"] = "hybrid"
    retrieval_candidates: int = 20  # chunks per search before fusion/reranking
    rrf_k: int = 60
    rerank: bool = False
    rerank_model: str | None = None  # empty = same as chat_model
    rerank_candidates: int = 10  # how many fused chunks the reranker reads

    @property
    def effective_judge_provider(self) -> ProviderName:
        return self.judge_provider or self.llm_provider

    @model_validator(mode="after")
    def check_api_keys(self) -> Self:
        keys = {"gemini": self.gemini_api_key, "openai": self.openai_api_key}
        for provider in {self.llm_provider, self.effective_judge_provider}:
            if provider in keys and not keys[provider]:
                raise ValueError(f"{provider.upper()}_API_KEY is required")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
