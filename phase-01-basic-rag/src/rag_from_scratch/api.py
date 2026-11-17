"""API FastAPI: POST /documents (upload) e POST /ask (pergunta)."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from openai import AsyncOpenAI
from opensearchpy import AsyncOpenSearch

from rag_from_scratch.config import get_settings
from rag_from_scratch.loaders import UnsupportedFileError, extract_text
from rag_from_scratch.models import AskRequest, AskResponse, IngestResponse
from rag_from_scratch.openai_client import OpenAIChat, OpenAIEmbedder
from rag_from_scratch.opensearch_store import OpenSearchStore
from rag_from_scratch.rag import RagService

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    s = get_settings()
    openai_client = AsyncOpenAI(api_key=s.openai_api_key)
    os_client = AsyncOpenSearch(hosts=[s.opensearch_url])
    store = OpenSearchStore(os_client, s.index_name, s.embedding_dim)
    await store.ensure_index()

    app.state.rag = RagService(
        embedder=OpenAIEmbedder(openai_client, s.embedding_model),
        store=store,
        chat=OpenAIChat(openai_client, s.chat_model),
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        top_k=s.top_k,
    )
    yield
    await os_client.close()
    await openai_client.close()


app = FastAPI(title="rag-from-scratch", version="0.1.0", lifespan=lifespan)


def get_rag(request: Request) -> RagService:
    return request.app.state.rag  # type: ignore[no-any-return]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents", response_model=IngestResponse)
async def upload_document(file: UploadFile, rag: RagService = Depends(get_rag)) -> IngestResponse:
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Ficheiro demasiado grande")
    try:
        text = extract_text(file.filename or "", data)
    except UnsupportedFileError as e:
        raise HTTPException(415, str(e)) from e
    if not text.strip():
        raise HTTPException(422, "Não foi possível extrair texto (PDF digitalizado?)")
    return await rag.ingest(file.filename or "sem-nome", text)


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, rag: RagService = Depends(get_rag)) -> AskResponse:
    return await rag.ask(body.question, body.top_k)
