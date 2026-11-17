"""OpenAI implementations of the Embedder and ChatModel protocols."""

from openai import AsyncOpenAI

from rag_from_scratch.infrastructure.retry import with_retry


class OpenAIEmbedder:
    def __init__(self, client: AsyncOpenAI, model: str, batch_size: int = 100) -> None:
        self.client = client
        self.model = model
        self.batch_size = batch_size

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        # the API accepts lists, but we send batches to stay within its limits
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = await with_retry(
                lambda batch=batch: self.client.embeddings.create(model=self.model, input=batch)
            )
            vectors.extend(item.embedding for item in resp.data)
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        # OpenAI embeddings are symmetric: queries and documents use the same call
        [vector] = await self.embed_documents([text])
        return vector


class OpenAIChat:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self.client = client
        self.model = model

    async def complete(self, system: str, user: str) -> str:
        resp = await with_retry(
            lambda: self.client.chat.completions.create(
                model=self.model,
                temperature=0,  # as deterministic as possible: makes evaluation easier
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        )
        return resp.choices[0].message.content or ""
