"""Text extraction from uploaded files (PDF, TXT, MD)."""

import io
from pathlib import Path

SUPPORTED = {".pdf", ".txt", ".md"}


class UnsupportedFileError(ValueError):
    pass


def extract_text(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise UnsupportedFileError(f"Unsupported file type: {suffix or '(no extension)'}")

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        # an empty line between pages helps the chunker respect the structure
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    return data.decode("utf-8", errors="replace")
