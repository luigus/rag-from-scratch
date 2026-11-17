"""Internal domain models shared by the services and the infrastructure."""

from pydantic import BaseModel


class Chunk(BaseModel):
    """A piece of a document, ready to be indexed."""

    chunk_id: str
    doc_id: str
    source: str  # original file name
    position: int  # order of the chunk inside the document
    text: str


class RetrievedChunk(BaseModel):
    """A chunk returned by the search, with its similarity score."""

    chunk: Chunk
    score: float
