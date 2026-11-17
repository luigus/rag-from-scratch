"""Extração de texto dos ficheiros enviados (PDF, TXT, MD)."""

import io
from pathlib import Path

SUPPORTED = {".pdf", ".txt", ".md"}


class UnsupportedFileError(ValueError):
    pass


def extract_text(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise UnsupportedFileError(f"Tipo não suportado: {suffix or '(sem extensão)'}")

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        # um parágrafo vazio entre páginas ajuda o chunking a respeitar a estrutura
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    return data.decode("utf-8", errors="replace")
