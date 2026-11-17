"""Runs the RAG pipeline over the golden dataset and scores every case."""

import asyncio
import time
from collections.abc import Callable

from pydantic import BaseModel

from rag_from_scratch.evaluation.dataset import Category, EvalCase
from rag_from_scratch.evaluation.judge import JudgeVerdict, LLMJudge
from rag_from_scratch.evaluation.metrics import RetrievalScores, score_retrieval
from rag_from_scratch.evaluation.variants import Variant
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
    error: str | None = None  # set when the case failed even after the retries


async def evaluate_case(
    rag: RagService,
    judge: LLMJudge | None,
    case: EvalCase,
    k: int,
    variant: Variant | None = None,
) -> CaseResult:
    start = time.perf_counter()
    retrieved = await rag.retrieve(
        case.question,
        top_k=k,
        mode=variant.mode if variant else None,
        rerank=variant.rerank if variant else None,
    )
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


def failed_case(case: EvalCase, error: Exception) -> CaseResult:
    return CaseResult(
        id=case.id,
        category=case.category,
        question=case.question,
        answer="",
        retrieved_sources=[],
        retrieval=None,
        verdict=None,
        latency_ms=0.0,
        error=f"{type(error).__name__}: {error}",
    )


async def run_evaluation(
    rag: RagService,
    judge: LLMJudge | None,
    cases: list[EvalCase],
    *,
    k: int,
    concurrency: int = 5,
    variant: Variant | None = None,
    on_progress: Callable[[CaseResult, int, int], None] | None = None,
) -> list[CaseResult]:
    """Evaluate the cases in parallel, with a limit to respect the API rate limits.

    `on_progress(result, done, total)` is called as each case finishes.
    """
    semaphore = asyncio.Semaphore(concurrency)
    done = 0

    async def limited(case: EvalCase) -> CaseResult:
        nonlocal done
        async with semaphore:
            try:
                result = await evaluate_case(rag, judge, case, k, variant)
            except Exception as e:
                # one failed question must not throw away the results of all the others
                result = failed_case(case, e)
        done += 1
        if on_progress:
            on_progress(result, done, len(cases))
        return result

    # gather keeps the input order, so the report follows the dataset order
    return list(await asyncio.gather(*(limited(c) for c in cases)))
