"""Request and response schemas for the /documents endpoints."""

from pydantic import BaseModel


class IngestResponse(BaseModel):
    doc_id: str
    source: str
    chunks_indexed: int
