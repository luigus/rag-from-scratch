"""Interfaces (Protocols) for the external dependencies.

RagService depends on these interfaces, not on Gemini, OpenAI or OpenSearch
directly:
- tests use fake implementations that are fast and deterministic;
- switching providers only needs a new class (see infrastructure/).
"""

from typing import Protocol

from rag_from_scratch.domain.models import Chunk, RetrievedChunk


class Embedder(Protocol):
    """Documents and queries are embedded separately: some models (like Gemini)
    produce better retrieval when they know which side of the search a text is on."""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class ChatModel(Protocol):
    async def complete(self, system: str, user: str) -> str: ...


class SearchStore(Protocol):
    """Stores the chunks and finds them in two ways:
    by meaning (vector search) and by words (keyword search, BM25)."""

    async def ensure_index(self) -> None: ...

    async def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    async def vector_search(self, vector: list[float], k: int) -> list[RetrievedChunk]: ...

    async def keyword_search(self, query: str, k: int) -> list[RetrievedChunk]: ...


class Reranker(Protocol):
    """Reorders a short list of candidates by how well each one answers the question."""

    async def rerank(
        self, question: str, candidates: list[RetrievedChunk], k: int
    ) -> list[RetrievedChunk]: ...
