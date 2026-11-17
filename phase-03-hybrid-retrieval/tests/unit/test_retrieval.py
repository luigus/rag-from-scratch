import asyncio

import pytest

from rag_from_scratch.domain.models import Chunk, RetrievedChunk
from rag_from_scratch.services.rag import RagService
from rag_from_scratch.services.retrieval import Retriever, reciprocal_rank_fusion
from tests.fakes import FakeChat, FakeEmbedder, InMemoryStore


def hit(chunk_id: str, score: float = 0.0) -> RetrievedChunk:
    chunk = Chunk(chunk_id=chunk_id, doc_id="d", source="s", position=0, text=chunk_id)
    return RetrievedChunk(chunk=chunk, score=score)


def ids(results: list[RetrievedChunk]) -> list[str]:
    return [r.chunk.chunk_id for r in results]


# --- Reciprocal Rank Fusion -------------------------------------------------------


def test_rrf_ignores_the_original_score_scales():
    # BM25 scores around 10, cosine around 0.8: only the positions should matter
    keyword = [hit("a", 12.7), hit("b", 9.1)]
    vector = [hit("b", 0.83), hit("a", 0.80)]
    fused = reciprocal_rank_fusion([keyword, vector])
    assert fused[0].score == pytest.approx(fused[1].score)  # a and b: 1st once, 2nd once


def test_rrf_rewards_chunks_found_by_both_searches():
    keyword = [hit("only-keyword"), hit("both")]
    vector = [hit("only-vector"), hit("both")]
    fused = reciprocal_rank_fusion([keyword, vector])
    # 2nd in both lists beats 1st in only one list
    assert ids(fused)[0] == "both"


def test_rrf_formula():
    fused = reciprocal_rank_fusion([[hit("a"), hit("b")]], k=60)
    assert fused[0].score == pytest.approx(1 / 61)
    assert fused[1].score == pytest.approx(1 / 62)


# --- Retriever modes on a small corpus -------------------------------------------

DOC = """\
Brake pads BRK-4471 are replaced when the pad thickness is below 7 mm.

Brake pads BRK-4417 are replaced once they are thinner than 9 mm.

The quiet car is car 5, where phone calls are not allowed.
"""


def make_service(**retriever_kwargs) -> RagService:
    embedder, store = FakeEmbedder(), InMemoryStore()
    retriever = Retriever(embedder, store, **retriever_kwargs)
    service = RagService(
        embedder, store, FakeChat(), retriever=retriever, chunk_size=80, chunk_overlap=0
    )
    asyncio.run(service.ingest("brakes.txt", DOC))
    return service


@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid"])
def test_every_mode_returns_at_most_k(mode):
    service = make_service(mode=mode)
    results = asyncio.run(service.retrieve("brake pads BRK-4417", top_k=2))
    assert 0 < len(results) <= 2


def test_keyword_mode_finds_the_exact_code():
    service = make_service(mode="keyword")
    results = asyncio.run(service.retrieve("BRK-4417", top_k=1))
    assert "BRK-4417" in results[0].chunk.text


def test_mode_can_be_overridden_per_call():
    service = make_service(mode="vector")
    results = asyncio.run(service.retrieve("quiet car", top_k=1, mode="keyword"))
    assert "quiet car" in results[0].chunk.text


class ReverseReranker:
    def __init__(self) -> None:
        self.calls = 0

    async def rerank(self, question, candidates, k):
        self.calls += 1
        return list(reversed(candidates))[:k]


def test_reranker_runs_only_when_enabled():
    reranker = ReverseReranker()
    service = make_service(mode="hybrid", reranker=reranker)

    asyncio.run(service.retrieve("brake pads", top_k=2, rerank=False))
    assert reranker.calls == 0

    asyncio.run(service.retrieve("brake pads", top_k=2, rerank=True))
    assert reranker.calls == 1
