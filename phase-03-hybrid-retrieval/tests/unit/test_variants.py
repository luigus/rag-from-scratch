import pytest

from rag_from_scratch.evaluation.variants import Variant, parse_variants


def test_parse_variants():
    variants = parse_variants("vector, keyword,hybrid+rerank")
    assert [v.name for v in variants] == ["vector", "keyword", "hybrid+rerank"]
    assert variants[2] == Variant(mode="hybrid", rerank=True)


@pytest.mark.parametrize("bad", ["bm25", "hybrid+magic", "vector+"])
def test_invalid_variants(bad):
    with pytest.raises(ValueError):
        Variant.parse(bad)
