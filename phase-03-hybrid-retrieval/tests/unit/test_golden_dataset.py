"""Checks on the dataset itself: a broken golden dataset gives misleading metrics."""

from pathlib import Path

from rag_from_scratch.evaluation.dataset import load_dataset
from rag_from_scratch.evaluation.metrics import contains

ROOT = Path(__file__).resolve().parents[2]
CASES = load_dataset(ROOT / "data/eval/golden.jsonl")
CORPUS = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "data/corpus").glob("*.md"))


def test_ids_are_unique():
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids))


def test_every_evidence_phrase_exists_in_the_corpus():
    missing = [(c.id, e) for c in CASES for e in c.evidence if not contains(CORPUS, e)]
    assert missing == []


def test_unanswerable_cases_have_no_evidence():
    assert all(not c.evidence for c in CASES if c.category == "unanswerable")
    assert all(c.evidence for c in CASES if c.category != "unanswerable")


def test_every_category_is_covered():
    assert {c.category for c in CASES} == {
        "factual",
        "paraphrase",
        "exact-code",
        "multi-hop",
        "unanswerable",
    }


def test_every_evidence_phrase_appears_exactly_once():
    # if a phrase appeared twice, "found it" would not tell which of the two was found
    norm = " ".join(CORPUS.split()).lower()
    repeated = [(c.id, e) for c in CASES for e in c.evidence if norm.count(" ".join(e.split()).lower()) != 1]
    assert repeated == []
