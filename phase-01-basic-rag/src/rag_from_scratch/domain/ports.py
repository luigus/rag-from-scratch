"""Interfaces (Protocols) for the external dependencies.

RagService depends on these interfaces, not on OpenAI or OpenSearch directly:
- tests use fake implementations that are fast and deterministic;
- switching providers (Azure OpenAI, Bedrock, Ollama) only needs a new class.
"""

from typing import Protocol

from rag_from_scratch.domain.models import Chunk, RetrievedChunk


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class ChatModel(Protocol):
    async def complete(self, system: str, user: str) -> str: ...


class VectorStore(Protocol):
    async def ensure_index(self) -> None: ...

    async def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    async def search(self, vector: list[float], k: int) -> list[RetrievedChunk]: ...
