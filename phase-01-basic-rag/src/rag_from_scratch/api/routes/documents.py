from fastapi import APIRouter, Depends, HTTPException, UploadFile

from rag_from_scratch.api.deps import get_rag
from rag_from_scratch.schemas.documents import IngestResponse
from rag_from_scratch.services.loaders import UnsupportedFileError, extract_text
from rag_from_scratch.services.rag import RagService

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=IngestResponse)
async def upload_document(file: UploadFile, rag: RagService = Depends(get_rag)) -> IngestResponse:
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large")
    try:
        text = extract_text(file.filename or "", data)
    except UnsupportedFileError as e:
        raise HTTPException(415, str(e)) from e
    if not text.strip():
        raise HTTPException(422, "Could not extract any text (scanned PDF?)")
    return await rag.ingest(file.filename or "unnamed", text)
