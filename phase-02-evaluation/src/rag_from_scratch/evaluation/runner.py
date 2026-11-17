"""Runs the RAG pipeline over the golden dataset and scores every case."""

import asyncio
import time

from pydantic import BaseModel

from rag_from_scratch.evaluation.dataset import Category, EvalCase
from rag_from_scratch.evaluation.judge import JudgeVerdict, LLMJudge
from rag_from_scratch.evaluation.metrics import RetrievalScores, score_retrieval
from rag_from_scratch.services.rag import RagService


class CaseResult(BaseModel):
    id: str
    category: Category
    question: str
    answer: str
    retrieved_sources: list[str]  # "file#position", in ranking order
    retrieval: RetrievalScores | None  # None for unanswerable questions
    verdict: JudgeVerdict | None  # None when the judge is disabled
    latency_ms: float  # retrieval + generation, without the judge


async def evaluate_case(
    rag: RagService, judge: LLMJudge | None, case: EvalCase, k: int
) -> CaseResult:
    start = time.perf_counter()
    retrieved = await rag.retrieve(case.question, top_k=k)
    response = await rag.generate(case.question, retrieved)
    latency_ms = (time.perf_counter() - start) * 1000

    retrieval = (
        score_retrieval([r.chunk.text for r in retrieved], case.evidence)
        if case.answerable
        else None
    )
    verdict = await judge.evaluate(case, response.answer, retrieved) if judge else None

    return CaseResult(
        id=case.id,
        category=case.category,
        question=case.question,
        answer=response.answer,
        retrieved_sources=[f"{r.chunk.source}#{r.chunk.position}" for r in retrieved],
        retrieval=retrieval,
        verdict=verdict,
        latency_ms=round(latency_ms, 1),
    )


async def run_evaluation(
    rag: RagService,
    judge: LLMJudge | None,
    cases: list[EvalCase],
    *,
    k: int,
    concurrency: int = 5,
) -> list[CaseResult]:
    """Evaluate the cases in parallel, with a limit to respect the API rate limits."""
    semaphore = asyncio.Semaphore(concurrency)

    async def limited(case: EvalCase) -> CaseResult:
        async with semaphore:
            return await evaluate_case(rag, judge, case, k)

    # gather keeps the input order, so the report follows the dataset order
    return list(await asyncio.gather(*(limited(c) for c in cases)))
