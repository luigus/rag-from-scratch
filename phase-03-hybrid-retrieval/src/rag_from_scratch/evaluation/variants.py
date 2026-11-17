"""A retrieval variant is one way of retrieving to compare: "vector", "keyword",
"hybrid" or "hybrid+rerank". The evaluation runs the same questions once per
variant, on the same index, so the only difference between the results is the
retrieval strategy."""

from dataclasses import dataclass

from rag_from_scratch.services.retrieval import RETRIEVAL_MODES, RetrievalMode


@dataclass(frozen=True)
class Variant:
    mode: RetrievalMode
    rerank: bool = False

    @property
    def name(self) -> str:
        return f"{self.mode}+rerank" if self.rerank else self.mode

    @classmethod
    def parse(cls, text: str) -> "Variant":
        mode, plus, extra = text.strip().partition("+")
        if mode not in RETRIEVAL_MODES or (plus and extra != "rerank"):
            valid = ", ".join([*RETRIEVAL_MODES, "<mode>+rerank"])
            raise ValueError(f"unknown variant {text!r}; valid: {valid}")
        return cls(mode=mode, rerank=extra == "rerank")  # type: ignore[arg-type]


def parse_variants(text: str) -> list[Variant]:
    return [Variant.parse(part) for part in text.split(",") if part.strip()]
