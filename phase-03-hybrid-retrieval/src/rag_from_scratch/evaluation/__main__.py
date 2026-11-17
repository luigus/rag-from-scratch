"""Evaluation CLI.

    python -m rag_from_scratch.evaluation                          # vector, keyword, hybrid
    python -m rag_from_scratch.evaluation --modes hybrid,hybrid+rerank
    python -m rag_from_scratch.evaluation --no-judge               # retrieval metrics only

Steps: reset a dedicated index, ingest the corpus once, then for every
retrieval variant run all the questions of the golden dataset, grade the
answers, and save one report per variant plus a comparison table.
"""

import argparse
import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path

from rag_from_scratch.core.config import Settings, get_settings
from rag_from_scratch.evaluation.dataset import load_dataset
from rag_from_scratch.evaluation.judge import JUDGE_PROMPT_VERSION, LLMJudge
from rag_from_scratch.evaluation.report import (
    EvalReport,
    build_report,
    comparison_markdown,
    to_markdown,
)
from rag_from_scratch.evaluation.runner import CaseResult, run_evaluation
from rag_from_scratch.evaluation.variants import Variant, parse_variants
from rag_from_scratch.infrastructure.factory import build_container
from rag_from_scratch.services.prompts import PROMPT_VERSION
from rag_from_scratch.services.reranking import LLMReranker


def print_progress(result: CaseResult, done: int, total: int) -> None:
    if result.error:
        print(f"[{done:>2}/{total}] {result.id} FAILED  {result.error[:100]}")
        return
    found = "–" if result.retrieval is None else ("found" if result.retrieval.hit else "missed")
    verdict = "" if result.verdict is None else ("correct" if result.verdict.correct else "WRONG")
    print(f"[{done:>2}/{total}] {result.id} {result.latency_ms / 1000:5.1f}s  {found:6} {verdict}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the RAG pipeline on the golden dataset")
    p.add_argument("--dataset", type=Path, default=Path("data/eval/golden.jsonl"))
    p.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    p.add_argument("--k", type=int, default=None, help="top-k chunks (default: TOP_K)")
    p.add_argument(
        "--modes",
        default="vector,keyword,hybrid",
        help="comma-separated variants: vector, keyword, hybrid, <mode>+rerank",
    )
    p.add_argument("--index", default="rag-eval", help="index used only by the evaluation")
    p.add_argument(
        "--concurrency", type=int, default=2, help="parallel questions (keep low on free tiers)"
    )
    p.add_argument("--no-judge", action="store_true", help="only retrieval metrics (cheaper)")
    p.add_argument("--out", type=Path, default=Path("eval-results"))
    return p.parse_args()


def run_config(
    settings: Settings, args: argparse.Namespace, k: int, variant: Variant
) -> dict[str, str | int]:
    return {
        "variant": variant.name,
        "provider": settings.llm_provider,
        "judge_provider": "disabled" if args.no_judge else settings.effective_judge_provider,
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "judge_model": "disabled" if args.no_judge else settings.judge_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": k,
        "retrieval_candidates": settings.retrieval_candidates,
        "rrf_k": settings.rrf_k,
        "prompt_version": PROMPT_VERSION,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "dataset": args.dataset.name,
    }


async def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="  ! %(message)s")
    settings = get_settings()
    k = args.k or settings.top_k
    variants = parse_variants(args.modes)
    container = build_container(settings, index_name=args.index)
    reports: dict[str, EvalReport] = {}

    try:
        # a fresh index on every run: results don't depend on leftovers of older runs
        await container.store.reset_index()
        files = sorted(args.corpus.glob("*.md")) + sorted(args.corpus.glob("*.txt"))
        for f in files:
            await container.rag.ingest(f.name, f.read_text(encoding="utf-8"))
        print(f"Ingested {len(files)} documents into '{args.index}'")

        retriever = container.rag.retriever
        if any(v.rerank for v in variants) and retriever.reranker is None:
            model = settings.rerank_model or settings.chat_model
            retriever.reranker = LLMReranker(container.provider.make_chat(model))

        judge = None if args.no_judge else LLMJudge(container.judge_chat())
        cases = load_dataset(args.dataset)

        for variant in variants:
            print(f"\n== {variant.name}: {len(cases)} questions ({args.concurrency} at a time)")
            results = await run_evaluation(
                container.rag,
                judge,
                cases,
                k=k,
                concurrency=args.concurrency,
                variant=variant,
                on_progress=print_progress,
            )
            reports[variant.name] = build_report(
                results, config=run_config(settings, args, k, variant)
            )
    finally:
        await container.close()

    out = args.out / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    for name, report in reports.items():
        safe = name.replace("+", "-")
        (out / f"{safe}.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        (out / f"{safe}.md").write_text(to_markdown(report), encoding="utf-8")
    comparison = comparison_markdown(reports)
    (out / "comparison.md").write_text(comparison, encoding="utf-8")

    print()
    print(comparison)
    print(f"Saved one report per variant and comparison.md in {out}/")


if __name__ == "__main__":
    asyncio.run(main())
