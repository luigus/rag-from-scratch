"""Builds the real clients and the RagService from the settings.

Used by both the API (main.py) and the evaluation CLI, so they always run the
exact same pipeline.
"""

from dataclasses import dataclass, field

from opensearchpy import AsyncOpenSearch

from rag_from_scratch.core.config import Settings
from rag_from_scratch.domain.ports import ChatModel
from rag_from_scratch.infrastructure.opensearch_store import OpenSearchStore
from rag_from_scratch.infrastructure.providers import LLMProvider, build_provider
from rag_from_scratch.services.rag import RagService
from rag_from_scratch.services.reranking import LLMReranker
from rag_from_scratch.services.retrieval import Retriever


@dataclass
class Container:
    settings: Settings
    provider: LLMProvider
    opensearch: AsyncOpenSearch
    store: OpenSearchStore
    rag: RagService
    _extra_providers: list[LLMProvider] = field(default_factory=list)

    def judge_chat(self) -> ChatModel:
        """The judge's chat model, which may come from another provider than the RAG."""
        name = self.settings.effective_judge_provider
        provider = self.provider
        if name != provider.name:
            provider = build_provider(name, self.settings)
            self._extra_providers.append(provider)
        return provider.make_chat(self.settings.judge_model)

    async def close(self) -> None:
        await self.opensearch.close()
        for provider in [self.provider, *self._extra_providers]:
            await provider.close()


def build_container(settings: Settings, index_name: str | None = None) -> Container:
    provider = build_provider(settings.llm_provider, settings)
    os_client = AsyncOpenSearch(hosts=[settings.opensearch_url])
    store = OpenSearchStore(os_client, index_name or settings.index_name, settings.embedding_dim)
    chat = provider.make_chat(settings.chat_model)
    reranker = (
        LLMReranker(provider.make_chat(settings.rerank_model or settings.chat_model))
        if settings.rerank
        else None
    )
    retriever = Retriever(
        provider.embedder,
        store,
        mode=settings.retrieval_mode,
        candidates=settings.retrieval_candidates,
        rrf_k=settings.rrf_k,
        reranker=reranker,
        rerank_candidates=settings.rerank_candidates,
    )
    rag = RagService(
        embedder=provider.embedder,
        store=store,
        chat=chat,
        retriever=retriever,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
    )
    return Container(
        settings=settings, provider=provider, opensearch=os_client, store=store, rag=rag
    )
