"""Retrieval metrics. Pure functions: easy to test and to reason about.

A retrieved chunk is *relevant* when it contains at least one evidence phrase.

- hit@k:       is there at least one relevant chunk in the top k? (0 or 1)
- recall@k:    fraction of the evidence phrases found in the top k
               (matters for questions that need several facts)
- precision@k: fraction of the retrieved chunks that are relevant
- MRR:         1 / rank of the first relevant chunk (1.0 = it came first)
"""

import re

from pydantic import BaseModel


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def contains(chunk_text: str, phrase: str) -> bool:
    return _normalize(phrase) in _normalize(chunk_text)


class RetrievalScores(BaseModel):
    hit: float
    recall: float
    precision: float
    reciprocal_rank: float
    first_relevant_rank: int | None  # 1-based; None if nothing relevant was found


def score_retrieval(retrieved_texts: list[str], evidence: list[str]) -> RetrievalScores:
    if not evidence:
        raise ValueError("retrieval metrics need at least one evidence phrase")

    relevant = [any(contains(t, e) for e in evidence) for t in retrieved_texts]
    found = [e for e in evidence if any(contains(t, e) for t in retrieved_texts)]
    first = next((i + 1 for i, r in enumerate(relevant) if r), None)

    return RetrievalScores(
        hit=1.0 if first else 0.0,
        recall=len(found) / len(evidence),
        precision=sum(relevant) / len(retrieved_texts) if retrieved_texts else 0.0,
        reciprocal_rank=1 / first if first else 0.0,
        first_relevant_rank=first,
    )
