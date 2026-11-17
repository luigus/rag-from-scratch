"""Modelos Pydantic: o "contrato" entre as camadas e a API."""

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Um pedaço de documento pronto a indexar."""

    chunk_id: str
    doc_id: str
    source: str  # nome do ficheiro original
    position: int  # ordem do chunk dentro do documento
    text: str


class RetrievedChunk(BaseModel):
    """Um chunk devolvido pela pesquisa, com o score de semelhança."""

    chunk: Chunk
    score: float


class IngestResponse(BaseModel):
    doc_id: str
    source: str
    chunks_indexed: int


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    source: str
    position: int
    score: float
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
