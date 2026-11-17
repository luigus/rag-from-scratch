"""Lógica do RAG: ingestão (texto → chunks → vetores → índice) e pergunta."""

import uuid

from rag_from_scratch.chunking import split_text
from rag_from_scratch.models import AskResponse, Chunk, IngestResponse, Source
from rag_from_scratch.ports import ChatModel, Embedder, VectorStore
from rag_from_scratch.prompts import SYSTEM_PROMPT, build_user_prompt

NO_CONTEXT_ANSWER = "Não encontrei informação sobre isso nos documentos indexados."


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
            vectors = await self.embedder.embed([c.text for c in chunks])
            await self.store.add(chunks, vectors)
        return IngestResponse(doc_id=doc_id, source=source, chunks_indexed=len(chunks))

    async def ask(self, question: str, top_k: int | None = None) -> AskResponse:
        [query_vector] = await self.embedder.embed([question])
        retrieved = await self.store.search(query_vector, k=top_k or self.top_k)
        if not retrieved:
            # sem contexto não chamamos o LLM: evita respostas inventadas e custo
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
