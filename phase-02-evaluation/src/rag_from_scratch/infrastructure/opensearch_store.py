"""VectorStore implementation with OpenSearch (k-NN with HNSW).

The `text` field is also indexed as full text: this enables lexical BM25 search
and hybrid search in phase 3 without reindexing.
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

    async def search(self, vector: list[float], k: int) -> list[RetrievedChunk]:
        resp = await self.client.search(
            index=self.index,
            body={
                "size": k,
                "query": {"knn": {"embedding": {"vector": vector, "k": k}}},
                "_source": {"excludes": ["embedding"]},
            },
        )
        return [
            RetrievedChunk(chunk=Chunk(**hit["_source"]), score=hit["_score"])
            for hit in resp["hits"]["hits"]
        ]
