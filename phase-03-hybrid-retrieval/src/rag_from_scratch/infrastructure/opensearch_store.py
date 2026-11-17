"""SearchStore implementation with OpenSearch.

The same index answers both kinds of search:
- vector search: k-NN with HNSW on the `embedding` field;
- keyword search: BM25 on the `text` field, OpenSearch's default text scoring.
"""

from opensearchpy import AsyncOpenSearch
from opensearchpy.helpers import async_bulk

from rag_from_scratch.domain.models import Chunk, RetrievedChunk


class OpenSearchStore:
    def __init__(self, client: AsyncOpenSearch, index: str, dim: int) -> None:
        self.client = client
        self.index = index
        self.dim = dim

    async def ensure_index(self) -> None:
        if await self.client.indices.exists(index=self.index):
            return
        await self.client.indices.create(
            index=self.index,
            body={
                "settings": {"index": {"knn": True}},
                "mappings": {
                    "properties": {
                        "chunk_id": {"type": "keyword"},
                        "doc_id": {"type": "keyword"},
                        "source": {"type": "keyword"},
                        "position": {"type": "integer"},
                        "text": {"type": "text"},
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": self.dim,
                            "method": {
                                "name": "hnsw",
                                "engine": "lucene",
                                "space_type": "cosinesimil",
                            },
                        },
                    }
                },
            },
        )

    async def reset_index(self) -> None:
        """Drop and recreate the index (used by the evaluation for clean runs)."""
        if await self.client.indices.exists(index=self.index):
            await self.client.indices.delete(index=self.index)
        await self.ensure_index()

    async def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        actions = [
            {
                "_index": self.index,
                "_id": c.chunk_id,
                "_source": {**c.model_dump(), "embedding": v},
            }
            for c, v in zip(chunks, vectors, strict=True)
        ]
        # refresh=True makes the chunks searchable right away (fine for a study
        # project; usually avoided in production because of the indexing cost)
        await async_bulk(self.client, actions, refresh=True)

    async def _search(self, query: dict[str, object], k: int) -> list[RetrievedChunk]:
        resp = await self.client.search(
            index=self.index,
            body={"size": k, "query": query, "_source": {"excludes": ["embedding"]}},
        )
        return [
            RetrievedChunk(chunk=Chunk(**hit["_source"]), score=hit["_score"])
            for hit in resp["hits"]["hits"]
        ]

    async def vector_search(self, vector: list[float], k: int) -> list[RetrievedChunk]:
        return await self._search({"knn": {"embedding": {"vector": vector, "k": k}}}, k)

    async def keyword_search(self, query: str, k: int) -> list[RetrievedChunk]:
        # the standard analyzer splits "BRK-4417" into "brk" and "4417", so an exact
        # part number matches even when it's written inside a sentence
        return await self._search({"match": {"text": query}}, k)
