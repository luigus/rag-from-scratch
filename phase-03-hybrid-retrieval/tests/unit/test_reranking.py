import asyncio

from rag_from_scratch.domain.models import Chunk, RetrievedChunk
from rag_from_scratch.services.reranking import LLMReranker, parse_ranking
from tests.fakes import FakeChat


def candidates(*texts: str) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=Chunk(chunk_id=t, doc_id="d", source="s", position=i, text=t), score=0.5
        )
        for i, t in enumerate(texts)
    ]


def test_parse_valid_ranking():
    assert parse_ranking('{"ranking": [3, 1, 2]}', 3) == [2, 0, 1]


def test_forgotten_passages_are_appended_in_original_order():
    assert parse_ranking('{"ranking": [2]}', 3) == [1, 0, 2]


def test_invalid_numbers_and_duplicates_are_ignored():
    assert parse_ranking('{"ranking": [9, 2, 2, "x", 1]}', 3) == [1, 0, 2]


def test_unparseable_output_keeps_the_original_order():
    assert parse_ranking("I think passage two is best", 3) == [0, 1, 2]


def test_thinking_text_is_ignored():
    assert parse_ranking('<think>{"ranking": [1]}</think>{"ranking": [2, 1]}', 2) == [1, 0]


def test_reranker_reorders_and_cuts_to_k():
    chat = FakeChat('{"ranking": [3, 1, 2]}')
    result = asyncio.run(LLMReranker(chat).rerank("q", candidates("a", "b", "c"), k=2))
    assert [r.chunk.text for r in result] == ["c", "a"]
    assert [r.score for r in result] == [1.0, 0.5]
