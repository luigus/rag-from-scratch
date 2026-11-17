"""Request and response schemas for the /ask endpoint."""

from pydantic import BaseModel, Field

from rag_from_scratch.services.retrieval import RetrievalMode


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    mode: RetrievalMode | None = None  # default: RETRIEVAL_MODE from the settings


class Source(BaseModel):
    source: str
    position: int
    score: float
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
