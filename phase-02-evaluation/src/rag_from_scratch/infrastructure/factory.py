"""Builds the real clients and the RagService from the settings.

Used by both the API (main.py) and the evaluation CLI, so they always run the
exact same pipeline.
"""

from dataclasses import dataclass

from opensearchpy import AsyncOpenSearch

from rag_from_scratch.core.config import Settings
from rag_from_scratch.infrastructure.opensearch_store import OpenSearchStore
from rag_from_scratch.infrastructure.providers import LLMProvider, build_provider
from rag_from_scratch.services.rag import RagService


@dataclass
class Container:
    provider: LLMProvider
    opensearch: AsyncOpenSearch
    store: OpenSearchStore
    rag: RagService

    async def close(self) -> None:
        await self.opensearch.close()
        await self.provider.close()


def build_container(settings: Settings, index_name: str | None = None) -> Container:
    provider = build_provider(settings)
    os_client = AsyncOpenSearch(hosts=[settings.opensearch_url])
    store = OpenSearchStore(os_client, index_name or settings.index_name, settings.embedding_dim)
    rag = RagService(
        embedder=provider.embedder,
        store=store,
        chat=provider.make_chat(settings.chat_model),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
    )
    return Container(provider=provider, opensearch=os_client, store=store, rag=rag)
