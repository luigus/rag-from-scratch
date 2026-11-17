"""FastAPI dependencies shared by the routes."""

from fastapi import Request

from rag_from_scratch.services.rag import RagService


def get_rag(request: Request) -> RagService:
    """Return the RagService created once at startup (see main.lifespan)."""
    return request.app.state.rag  # type: ignore[no-any-return]
