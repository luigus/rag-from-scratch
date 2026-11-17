"""Reranking with an LLM (listwise): the model sees the question and all the
candidates at once and returns them in order of usefulness.

Production systems usually use a dedicated cross-encoder (e.g. bge-reranker,
Cohere Rerank), which is faster and cheaper. An LLM reranker needs no extra
model or service, which keeps this project runnable with just Ollama or an API
key, and it is a good way to see what reranking does before optimizing it.
"""

import json
import re

from rag_from_scratch.domain.models import RetrievedChunk
from rag_from_scratch.domain.ports import ChatModel

RERANK_SYSTEM_PROMPT = """\
You rank passages for a question answering system.
Given a question and numbered passages, order the passages from the most to the
least useful for answering the question. A passage is useful only if it
contains facts that answer the question; passages about a similar topic but a
different train, part number or code are NOT useful.
Return ONLY a JSON object: {"ranking": [passage numbers, most useful first]}
"""


def build_rerank_prompt(question: str, candidates: list[RetrievedChunk]) -> str:
    passages = "\n\n".join(f"[{i}] {c.chunk.text}" for i, c in enumerate(candidates, 1))
    return f"Question: {question}\n\nPassages:\n{passages}"


def parse_ranking(raw: str, n: int) -> list[int]:
    """Return 0-based indexes. Unknown or repeated numbers are ignored, and passages
    the model forgot are appended in their original order, so nothing is lost."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    order: list[int] = []
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        values = json.loads(match.group())["ranking"] if match else []
    except (ValueError, KeyError, TypeError):
        values = []
    for value in values if isinstance(values, list) else []:
        if isinstance(value, int) and 1 <= value <= n and value - 1 not in order:
            order.append(value - 1)
    return order + [i for i in range(n) if i not in order]


class LLMReranker:
    def __init__(self, chat: ChatModel) -> None:
        self.chat = chat

    async def rerank(
        self, question: str, candidates: list[RetrievedChunk], k: int
    ) -> list[RetrievedChunk]:
        if len(candidates) <= 1:
            return candidates[:k]
        raw = await self.chat.complete(
            RERANK_SYSTEM_PROMPT, build_rerank_prompt(question, candidates)
        )
        order = parse_ranking(raw, len(candidates))
        # the new score is just the new position: 1.0, 0.5, 0.33, ...
        return [
            RetrievedChunk(chunk=candidates[i].chunk, score=round(1 / (pos + 1), 4))
            for pos, i in enumerate(order[:k])
        ]
