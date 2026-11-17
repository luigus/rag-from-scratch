"""Fake implementations of the protocols: fast tests, no network, no cost."""

import math
import re
import zlib

from rag_from_scratch.domain.models import Chunk, RetrievedChunk


class FakeEmbedder:
    """Bag-of-words 'embedding': texts that share words end up close together."""

    dim = 64

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for t in texts:
            v = [0.0] * self.dim
            for word in re.findall(r"\w+", t.lower()):
                # crc32 instead of hash(): hash() changes between runs and would make tests flaky
                v[zlib.crc32(word.encode()) % self.dim] += 1.0
            vectors.append(v)
        return vectors


def _cosine(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return 0.0 if na == 0 or nb == 0 else sum(x * y for x, y in zip(a, b)) / (na * nb)


class InMemoryStore:
    def __init__(self) -> None:
        self.items: list[tuple[Chunk, list[float]]] = []

    async def ensure_index(self) -> None:
        pass

    async def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self.items.extend(zip(chunks, vectors, strict=True))

    async def search(self, vector: list[float], k: int) -> list[RetrievedChunk]:
        scored = [RetrievedChunk(chunk=c, score=_cosine(vector, v)) for c, v in self.items]
        scored.sort(key=lambda r: r.score, reverse=True)
        return [r for r in scored[:k] if r.score > 0]


class FakeChat:
    def __init__(self, reply: str = "answer [1]") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply
