import asyncio

from rag_from_scratch.evaluation.dataset import EvalCase
from rag_from_scratch.evaluation.judge import LLMJudge
from rag_from_scratch.evaluation.report import build_report, to_markdown
from rag_from_scratch.evaluation.runner import run_evaluation
from rag_from_scratch.services.rag import RagService
from tests.fakes import FakeChat, FakeEmbedder, InMemoryStore

DOC = """\
The Velaro train has a maximum speed of 320 km/h.

Preventive maintenance of the bogies is done every 1.2 million km.

The signalling system used on the line is ETCS level 2.
"""

CASES = [
    EvalCase(
        id="q1",
        question="What is the maximum speed of the Velaro train?",
        reference_answer="320 km/h",
        evidence=["maximum speed of 320 km/h"],
        category="factual",
    ),
    EvalCase(
        id="q2",
        question="Which signalling system is used on the line?",
        reference_answer="ETCS level 2",
        evidence=["ETCS level 2"],
        category="paraphrase",
    ),
    EvalCase(
        id="q3",
        question="Who built the trains?",
        reference_answer="Not in the documents.",
        category="unanswerable",
    ),
]


def run(judge_reply: str = '{"faithful": true, "correct": true, "reason": "ok"}'):
    rag = RagService(FakeEmbedder(), InMemoryStore(), FakeChat(), chunk_size=80, chunk_overlap=0)
    asyncio.run(rag.ingest("velaro.txt", DOC))
    judge = LLMJudge(FakeChat(judge_reply))
    return asyncio.run(run_evaluation(rag, judge, CASES, k=2, concurrency=2))


def test_results_keep_dataset_order_and_score_retrieval():
    results = run()
    assert [r.id for r in results] == ["q1", "q2", "q3"]
    assert results[0].retrieval is not None and results[0].retrieval.hit == 1.0
    assert results[2].retrieval is None  # unanswerable: no retrieval metrics


def test_report_aggregates_and_renders():
    results = run('{"faithful": true, "correct": false, "reason": "missed it"}')
    report = build_report(results, config={"top_k": 2})

    assert report.overall.cases == 3
    assert report.overall.hit_at_k == 1.0
    assert report.overall.correctness == 0.0
    assert set(report.by_category) == {"factual", "paraphrase", "unanswerable"}
    assert report.by_category["unanswerable"].hit_at_k is None

    md = to_markdown(report)
    assert "Hit@2" in md
    assert "missed it" in md
