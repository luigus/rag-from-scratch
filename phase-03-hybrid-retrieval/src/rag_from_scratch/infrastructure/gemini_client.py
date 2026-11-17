"""Gemini implementations of the Embedder and ChatModel protocols (google-genai SDK)."""

from google import genai
from google.genai import types

from rag_from_scratch.infrastructure.retry import with_retry
from rag_from_scratch.infrastructure.vectors import l2_normalize


class GeminiEmbedder:
    def __init__(self, client: genai.Client, model: str, dim: int, batch_size: int = 100) -> None:
        self.client = client
        self.model = model
        self.dim = dim
        self.batch_size = batch_size

    async def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim)
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = await with_retry(
                lambda batch=batch: self.client.aio.models.embed_content(
                    model=self.model, contents=batch, config=config
                )
            )
            vectors.extend(l2_normalize(list(e.values or [])) for e in resp.embeddings or [])
        return vectors

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts, "RETRIEVAL_DOCUMENT")

    async def embed_query(self, text: str) -> list[float]:
        [vector] = await self._embed([text], "RETRIEVAL_QUERY")
        return vector


class GeminiChat:
    def __init__(self, client: genai.Client, model: str) -> None:
        self.client = client
        self.model = model

    async def complete(self, system: str, user: str) -> str:
        resp = await with_retry(
            lambda: self.client.aio.models.generate_content(
                model=self.model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0,  # as deterministic as possible: makes evaluation easier
                ),
            )
        )
        return resp.text or ""
