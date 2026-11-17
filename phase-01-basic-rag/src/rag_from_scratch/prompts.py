"""Prompts versionados como código: mudar aqui = mudança revista em PR."""

from rag_from_scratch.models import RetrievedChunk

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """\
You are an assistant that answers questions using ONLY the provided context.
Rules:
- If the context does not contain the answer, say you don't know.
- Cite the sources you used as [n], where n is the number of the context block.
- Answer in the same language as the question.
"""


def build_user_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    blocks = "\n\n".join(
        f"[{i}] (source: {r.chunk.source}, part {r.chunk.position})\n{r.chunk.text}"
        for i, r in enumerate(retrieved, start=1)
    )
    return f"Context:\n{blocks}\n\nQuestion: {question}"
