"""Interfaces (Protocols) das dependências externas.

O RagService depende destas interfaces e não da OpenAI ou do OpenSearch
diretamente. Assim:
- os testes usam implementações falsas, rápidas e determinísticas;
- trocar de fornecedor (Azure OpenAI, Bedrock, Ollama) é só uma nova classe.
"""

from typing import Protocol

from rag_from_scratch.models import Chunk, RetrievedChunk


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class ChatModel(Protocol):
    async def complete(self, system: str, user: str) -> str: ...


class VectorStore(Protocol):
    async def ensure_index(self) -> None: ...

    async def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    async def search(self, vector: list[float], k: int) -> list[RetrievedChunk]: ...
