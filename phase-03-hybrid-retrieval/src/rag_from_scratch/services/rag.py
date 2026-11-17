"""RAG logic: ingestion (text → chunks → vectors → index) and question answering.

Answering is split into `retrieve` and `generate` so the evaluation can score
each half separately: was the right context found, and was it used well?
"""

import uuid

from rag_from_scratch.domain.models import Chunk, RetrievedChunk
from rag_from_scratch.domain.ports import ChatModel, Embedder, SearchStore
from rag_from_scratch.schemas.ask import AskResponse, Source
from rag_from_scratch.schemas.documents import IngestResponse
from rag_from_scratch.services.chunking import split_text
from rag_from_scratch.services.prompts import SYSTEM_PROMPT, build_user_prompt
from rag_from_scratch.services.retrieval import RetrievalMode, Retriever

NO_CONTEXT_ANSWER = "I couldn't find information about that in the indexed documents."


class RagService:
    def __init__(
        self,
        embedder: Embedder,
        store: SearchStore,
        chat: ChatModel,
        *,
        retriever: Retriever | None = None,
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
        # without an explicit retriever, plain vector search (the behaviour of phases 1-2)
        self.retriever = retriever or Retriever(embedder, store, mode="vector")

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

    async def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        *,
        mode: RetrievalMode | None = None,
        rerank: bool | None = None,
    ) -> list[RetrievedChunk]:
        return await self.retriever.retrieve(
            question, top_k or self.top_k, mode=mode, rerank=rerank
        )

    async def generate(self, question: str, retrieved: list[RetrievedChunk]) -> AskResponse:
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

    async def ask(
        self, question: str, top_k: int | None = None, mode: RetrievalMode | None = None
    ) -> AskResponse:
        retrieved = await self.retrieve(question, top_k, mode=mode)
        return await self.generate(question, retrieved)
