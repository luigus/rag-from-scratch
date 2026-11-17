"""The golden dataset: questions with a reference answer and the evidence needed.

Relevance is defined by *evidence phrases* (exact sentences from the corpus)
instead of chunk ids. Chunk ids change every time the chunking changes, while
the evidence stays the same, so the dataset survives the next phases.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["factual", "paraphrase", "exact-code", "multi-hop", "unanswerable"]


class EvalCase(BaseModel):
    id: str
    question: str
    reference_answer: str
    evidence: list[str] = Field(default_factory=list)  # empty = the corpus has no answer
    category: Category

    @property
    def answerable(self) -> bool:
        return bool(self.evidence)


def load_dataset(path: Path) -> list[EvalCase]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [EvalCase.model_validate_json(line) for line in lines if line.strip()]
