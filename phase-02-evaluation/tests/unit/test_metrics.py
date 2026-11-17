import pytest

from rag_from_scratch.evaluation.metrics import contains, score_retrieval


def test_contains_ignores_case_and_whitespace():
    assert contains("The fleet  has\n24 units.", "the fleet has 24 units")


def test_relevant_chunk_first():
    s = score_retrieval(["the answer is here", "noise", "more noise"], ["answer is here"])
    assert (s.hit, s.recall, s.reciprocal_rank, s.first_relevant_rank) == (1.0, 1.0, 1.0, 1)
    assert s.precision == pytest.approx(1 / 3)


def test_relevant_chunk_third():
    s = score_retrieval(["noise", "noise", "the answer"], ["the answer"])
    assert s.reciprocal_rank == pytest.approx(1 / 3)
    assert s.first_relevant_rank == 3


def test_nothing_relevant():
    s = score_retrieval(["noise", "noise"], ["the answer"])
    assert (s.hit, s.recall, s.precision, s.reciprocal_rank) == (0.0, 0.0, 0.0, 0.0)
    assert s.first_relevant_rank is None


def test_partial_recall_for_multi_fact_questions():
    s = score_retrieval(["fact A is here", "noise"], ["fact A", "fact B"])
    assert s.hit == 1.0
    assert s.recall == 0.5


def test_needs_evidence():
    with pytest.raises(ValueError):
        score_retrieval(["text"], [])
