"""RAG logic: ingestion (text → chunks → vectors → index) and question answering."""

import uuid

from rag_from_scratch.domain.models import Chunk
from rag_from_scratch.domain.ports import ChatModel, Embedder, VectorStore
from rag_from_scratch.schemas.ask import AskResponse, Source
from rag_from_scratch.schemas.documents import IngestResponse
from rag_from_scratch.services.chunking import split_text
from rag_from_scratch.services.prompts import SYSTEM_PROMPT, build_user_prompt

NO_CONTEXT_ANSWER = "I couldn't find information about that in the indexed documents."


class RagService:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        chat: ChatModel,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        top_k: int = 5,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.chat = chat
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

    async def ingest(self, source: str, text: str) -> IngestResponse:
        doc_id = uuid.uuid4().hex
        pieces = split_text(text, self.chunk_size, self.chunk_overlap)
        chunks = [
            Chunk(chunk_id=f"{doc_id}-{i}", doc_id=doc_id, source=source, position=i, text=t)
            for i, t in enumerate(pieces)
        ]
        if chunks:
            vectors = await self.embedder.embed_documents([c.text for c in chunks])
            await self.store.add(chunks, vectors)
        return IngestResponse(doc_id=doc_id, source=source, chunks_indexed=len(chunks))

    async def ask(self, question: str, top_k: int | None = None) -> AskResponse:
        query_vector = await self.embedder.embed_query(question)
        retrieved = await self.store.search(query_vector, k=top_k or self.top_k)
        if not retrieved:
            # no context means no LLM call: avoids made-up answers and saves cost
            return AskResponse(answer=NO_CONTEXT_ANSWER, sources=[])

        answer = await self.chat.complete(SYSTEM_PROMPT, build_user_prompt(question, retrieved))
        sources = [
            Source(
                source=r.chunk.source,
                position=r.chunk.position,
                score=round(r.score, 4),
                excerpt=r.chunk.text[:200],
            )
            for r in retrieved
        ]
        return AskResponse(answer=answer, sources=sources)
