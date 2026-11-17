"""Aggregates the case results into a report (JSON for machines, Markdown for people)."""

from collections.abc import Sequence
from datetime import UTC, datetime
from statistics import mean, quantiles

from pydantic import BaseModel

from rag_from_scratch.evaluation.runner import CaseResult


class Summary(BaseModel):
    cases: int
    hit_at_k: float | None = None
    recall_at_k: float | None = None
    precision_at_k: float | None = None
    mrr: float | None = None
    faithfulness: float | None = None
    correctness: float | None = None
    judge_errors: int = 0


class EvalReport(BaseModel):
    created_at: str
    config: dict[str, str | int]  # everything needed to reproduce the run
    overall: Summary
    by_category: dict[str, Summary]
    latency_p50_ms: float
    latency_p95_ms: float
    results: list[CaseResult]


def _avg(values: Sequence[float]) -> float | None:
    return round(mean(values), 3) if values else None


def summarize(results: Sequence[CaseResult]) -> Summary:
    scored = [r.retrieval for r in results if r.retrieval]
    verdicts = [r.verdict for r in results if r.verdict]
    return Summary(
        cases=len(results),
        hit_at_k=_avg([s.hit for s in scored]),
        recall_at_k=_avg([s.recall for s in scored]),
        precision_at_k=_avg([s.precision for s in scored]),
        mrr=_avg([s.reciprocal_rank for s in scored]),
        faithfulness=_avg([float(v.faithful) for v in verdicts]),
        correctness=_avg([float(v.correct) for v in verdicts]),
        judge_errors=sum(v.error for v in verdicts),
    )


def build_report(results: list[CaseResult], config: dict[str, str | int]) -> EvalReport:
    categories = sorted({r.category for r in results})
    latencies = [r.latency_ms for r in results]
    p50, p95 = (
        (quantiles(latencies, n=100)[49], quantiles(latencies, n=100)[94])
        if len(latencies) > 1
        else (latencies[0], latencies[0])
    )
    return EvalReport(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        config=config,
        overall=summarize(results),
        by_category={c: summarize([r for r in results if r.category == c]) for c in categories},
        latency_p50_ms=round(p50, 1),
        latency_p95_ms=round(p95, 1),
        results=results,
    )


def _fmt(value: float | None) -> str:
    return "–" if value is None else f"{value:.2f}"


def to_markdown(report: EvalReport) -> str:
    k = report.config.get("top_k", "k")
    header = (
        f"| Group | Cases | Hit@{k} | Recall@{k} | Precision@{k} | MRR | Faithful | Correct |\n"
        "|---|---|---|---|---|---|---|---|"
    )

    def row(name: str, s: Summary) -> str:
        values = [s.hit_at_k, s.recall_at_k, s.precision_at_k, s.mrr, s.faithfulness, s.correctness]
        return f"| {name} | {s.cases} | " + " | ".join(_fmt(v) for v in values) + " |"

    rows = [row("**overall**", report.overall)]
    rows += [row(name, s) for name, s in report.by_category.items()]

    failures = [
        f"- `{r.id}` ({r.category}) {r.question} → {r.verdict.reason}"
        for r in report.results
        if r.verdict and not r.verdict.correct
    ]
    config = ", ".join(f"{key}={value}" for key, value in report.config.items())

    return "\n".join(
        [
            f"# Evaluation report — {report.created_at}",
            "",
            f"Config: {config}",
            "",
            header,
            *rows,
            "",
            f"Latency (retrieval + generation): p50 {report.latency_p50_ms:.0f} ms, "
            f"p95 {report.latency_p95_ms:.0f} ms. Judge errors: {report.overall.judge_errors}.",
            "",
            "## Incorrect answers",
            "",
            *(failures or ["None 🎉"]),
            "",
        ]
    )
