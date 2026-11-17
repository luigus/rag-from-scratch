"""Retrieval strategies: vector, keyword (BM25), hybrid, and optional reranking.

- vector:  finds chunks with a similar *meaning*. Good with paraphrases
           ("how fast can it go?" ↔ "maximum speed"), weak with exact codes.
- keyword: finds chunks that contain the *same words* (BM25). Good with codes
           and names ("BRK-4417"), weak with paraphrases.
- hybrid:  runs both and merges the two rankings with Reciprocal Rank Fusion.

Reranking is an optional last step: a model reads the question and the top
candidates together and puts them in a better order.
"""

import asyncio
from typing import Literal, get_args

from rag_from_scratch.domain.models import RetrievedChunk
from rag_from_scratch.domain.ports import Embedder, Reranker, SearchStore

RetrievalMode = Literal["vector", "keyword", "hybrid"]
RETRIEVAL_MODES: tuple[RetrievalMode, ...] = get_args(RetrievalMode)


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]], *, k: int = 60
) -> list[RetrievedChunk]:
    """Merge several rankings into one.

    Each chunk gets 1 / (k + rank) from every ranking it appears in, and the
    points are added up. Only the *positions* matter, not the original scores,
    which is the point: a BM25 score (e.g. 12.7) and a cosine similarity
    (e.g. 0.83) are on different scales and can't be added directly.

    k = 60 is the value from the original RRF paper (Cormack et al., 2009). It
    softens the gap between the first positions, so a chunk that is 2nd in both
    lists can beat one that is 1st in only one of them.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            chunk_id = item.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)
            chunks.setdefault(chunk_id, item)

    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [RetrievedChunk(chunk=chunks[cid].chunk, score=scores[cid]) for cid in ordered]


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: SearchStore,
        *,
        mode: RetrievalMode = "hybrid",
        candidates: int = 20,
        rrf_k: int = 60,
        reranker: Reranker | None = None,
        rerank_candidates: int = 10,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.mode = mode
        self.candidates = candidates
        self.rrf_k = rrf_k
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates

    async def _vector(self, question: str, n: int) -> list[RetrievedChunk]:
        vector = await self.embedder.embed_query(question)
        return await self.store.vector_search(vector, n)

    async def retrieve(
        self,
        question: str,
        k: int,
        *,
        mode: RetrievalMode | None = None,
        rerank: bool | None = None,
    ) -> list[RetrievedChunk]:
        mode = mode or self.mode
        use_reranker = self.reranker is not None and (rerank if rerank is not None else True)

        # fusion and reranking need a bigger pool than the final k to choose from
        pool = max(k, self.candidates) if mode == "hybrid" or use_reranker else k

        if mode == "vector":
            results = await self._vector(question, pool)
        elif mode == "keyword":
            results = await self.store.keyword_search(question, pool)
        else:
            # both searches are independent, so they run at the same time
            vector_hits, keyword_hits = await asyncio.gather(
                self._vector(question, pool), self.store.keyword_search(question, pool)
            )
            results = reciprocal_rank_fusion([vector_hits, keyword_hits], k=self.rrf_k)

        if use_reranker and self.reranker and results:
            results = await self.reranker.rerank(
                question, results[: self.rerank_candidates], k
            )
        return results[:k]
