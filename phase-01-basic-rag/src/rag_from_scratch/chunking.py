"""Divisão de texto em chunks com sobreposição (overlap).

Estratégia: juntar parágrafos até chegar perto de `chunk_size`. Um parágrafo
maior do que `chunk_size` é cortado em frases e, em último caso, por caracteres.
Cada novo chunk começa com o final do anterior (`overlap`), para que uma ideia
cortada a meio não se perca na pesquisa.
"""

import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split_long(text: str, size: int) -> list[str]:
    """Corta um bloco grande em frases; se ainda for grande, por caracteres."""
    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        if len(sentence) <= size:
            pieces.append(sentence)
        else:
            pieces.extend(sentence[i : i + size] for i in range(0, len(sentence), size))
    return pieces


def split_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size tem de ser > 0")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap tem de estar entre 0 e chunk_size - 1")

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
        # o overlap nunca pode fazer o chunk passar do tamanho máximo
        current = f"{tail} {unit}".strip() if len(tail) + 1 + len(unit) <= chunk_size else unit
    if current:
        chunks.append(current)
    return chunks
