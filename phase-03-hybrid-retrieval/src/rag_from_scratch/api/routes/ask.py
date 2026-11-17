from fastapi import APIRouter, Depends

from rag_from_scratch.api.deps import get_rag
from rag_from_scratch.schemas.ask import AskRequest, AskResponse
from rag_from_scratch.services.rag import RagService

router = APIRouter(prefix="/ask", tags=["ask"])


@router.post("", response_model=AskResponse)
async def ask(body: AskRequest, rag: RagService = Depends(get_rag)) -> AskResponse:
    return await rag.ask(body.question, body.top_k, body.mode)
