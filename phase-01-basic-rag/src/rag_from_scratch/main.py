"""Application entry point: creates the FastAPI app and wires the dependencies."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from opensearchpy import AsyncOpenSearch

from rag_from_scratch.api.routes import ask, documents, health
from rag_from_scratch.core.config import get_settings
from rag_from_scratch.infrastructure.opensearch_store import OpenSearchStore
from rag_from_scratch.infrastructure.providers import build_provider
from rag_from_scratch.services.rag import RagService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create the clients once at startup and close them at shutdown."""
    s = get_settings()
    provider = build_provider(s.llm_provider, s)  # Gemini, OpenAI or Ollama
    os_client = AsyncOpenSearch(hosts=[s.opensearch_url])
    store = OpenSearchStore(os_client, s.index_name, s.embedding_dim)
    await store.ensure_index()

    app.state.rag = RagService(
        embedder=provider.embedder,
        store=store,
        chat=provider.make_chat(s.chat_model),
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        top_k=s.top_k,
    )
    yield
    await os_client.close()
    await provider.close()


app = FastAPI(title="rag-from-scratch", version="0.1.0", lifespan=lifespan)
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(ask.router)
