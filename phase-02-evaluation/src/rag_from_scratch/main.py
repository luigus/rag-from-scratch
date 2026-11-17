"""Application entry point: creates the FastAPI app and wires the dependencies."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag_from_scratch.api.routes import ask, documents, health
from rag_from_scratch.core.config import get_settings
from rag_from_scratch.infrastructure.factory import build_container


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create the clients once at startup and close them at shutdown."""
    container = build_container(get_settings())
    await container.store.ensure_index()
    app.state.rag = container.rag
    yield
    await container.close()


app = FastAPI(title="rag-from-scratch", version="0.2.0", lifespan=lifespan)
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(ask.router)
