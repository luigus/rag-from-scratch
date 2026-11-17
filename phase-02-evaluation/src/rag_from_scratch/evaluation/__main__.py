"""Evaluation CLI.

    python -m rag_from_scratch.evaluation
    python -m rag_from_scratch.evaluation --k 3 --no-judge

Steps: reset a dedicated index, ingest the corpus, run every question of the
golden dataset, grade the answers and save a JSON and a Markdown report.
"""

import argparse
import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path

from rag_from_scratch.core.config import get_settings
from rag_from_scratch.evaluation.dataset import load_dataset
from rag_from_scratch.evaluation.judge import JUDGE_PROMPT_VERSION, LLMJudge
from rag_from_scratch.evaluation.report import build_report, to_markdown
from rag_from_scratch.evaluation.runner import CaseResult, run_evaluation
from rag_from_scratch.infrastructure.factory import build_container
from rag_from_scratch.services.prompts import PROMPT_VERSION


def print_progress(result: CaseResult, done: int, total: int) -> None:
    found = "–" if result.retrieval is None else ("found" if result.retrieval.hit else "missed")
    verdict = "" if result.verdict is None else ("correct" if result.verdict.correct else "WRONG")
    print(f"[{done:>2}/{total}] {result.id} {result.latency_ms / 1000:5.1f}s  {found:6} {verdict}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the RAG pipeline on the golden dataset")
    p.add_argument("--dataset", type=Path, default=Path("data/eval/golden.jsonl"))
    p.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    p.add_argument("--k", type=int, default=None, help="top-k chunks (default: TOP_K)")
    p.add_argument("--index", default="rag-eval", help="index used only by the evaluation")
    p.add_argument(
        "--concurrency", type=int, default=2, help="parallel questions (keep low on free tiers)"
    )
    p.add_argument("--no-judge", action="store_true", help="only retrieval metrics (cheaper)")
    p.add_argument("--out", type=Path, default=Path("eval-results"))
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="  ! %(message)s")
    settings = get_settings()
    k = args.k or settings.top_k
    container = build_container(settings, index_name=args.index)

    try:
        # a fresh index on every run: results don't depend on leftovers of older runs
        await container.store.reset_index()
        files = sorted(args.corpus.glob("*.md")) + sorted(args.corpus.glob("*.txt"))
        for f in files:
            await container.rag.ingest(f.name, f.read_text(encoding="utf-8"))
        print(f"Ingested {len(files)} documents into '{args.index}'")

        judge = (
            None if args.no_judge else LLMJudge(container.provider.make_chat(settings.judge_model))
        )
        cases = load_dataset(args.dataset)
        print(f"Running {len(cases)} questions ({args.concurrency} at a time)...")
        results = await run_evaluation(
            container.rag,
            judge,
            cases,
            k=k,
            concurrency=args.concurrency,
            on_progress=print_progress,
        )
    finally:
        await container.close()

    report = build_report(
        results,
        config={
            "provider": settings.llm_provider,
            "chat_model": settings.chat_model,
            "embedding_model": settings.embedding_model,
            "judge_model": "disabled" if args.no_judge else settings.judge_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": k,
            "prompt_version": PROMPT_VERSION,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "dataset": args.dataset.name,
        },
    )

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    (args.out / f"{stamp}.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    markdown = to_markdown(report)
    (args.out / f"{stamp}.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Saved to {args.out}/{stamp}.json and .md")


if __name__ == "__main__":
    asyncio.run(main())
