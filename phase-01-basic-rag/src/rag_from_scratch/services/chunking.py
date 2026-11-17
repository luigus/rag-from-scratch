"""Split text into chunks with overlap.

Strategy: join paragraphs until the chunk gets close to `chunk_size`. A
paragraph larger than `chunk_size` is split into sentences and, as a last
resort, by characters. Each new chunk starts with the end of the previous one
(`overlap`), so an idea cut in the middle is not lost during search.
"""

import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split_long(text: str, size: int) -> list[str]:
    """Split a large block into sentences; if still too large, by characters."""
    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        if len(sentence) <= size:
            pieces.append(sentence)
        else:
            pieces.extend(sentence[i : i + size] for i in range(0, len(sentence), size))
    return pieces


def split_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for p in paragraphs:
        units.extend(_split_long(p, chunk_size) if len(p) > chunk_size else [p])

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        chunks.append(current)
        tail = current[-overlap:] if overlap else ""
        # the overlap must never push a chunk over the maximum size
        current = f"{tail} {unit}".strip() if len(tail) + 1 + len(unit) <= chunk_size else unit
    if current:
        chunks.append(current)
    return chunks
