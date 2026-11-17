import pytest

from rag_from_scratch.chunking import split_text


def test_short_text_is_one_chunk():
    assert split_text("Olá mundo.", chunk_size=100, overlap=10) == ["Olá mundo."]


def test_empty_text_has_no_chunks():
    assert split_text("   \n\n  ", chunk_size=100, overlap=10) == []


@pytest.mark.parametrize("size,overlap", [(50, 0), (50, 10), (200, 40)])
def test_chunks_never_exceed_size(size, overlap):
    text = "\n\n".join(f"Parágrafo {i}. " + "palavra " * (i * 7) for i in range(20))
    chunks = split_text(text, chunk_size=size, overlap=overlap)
    assert chunks
    assert all(len(c) <= size for c in chunks)


def test_overlap_repeats_end_of_previous_chunk():
    text = "\n\n".join(f"Este é o parágrafo número {i}." for i in range(10))
    chunks = split_text(text, chunk_size=60, overlap=15)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev[-15:].strip() in nxt


def test_all_content_is_kept():
    paragraphs = [f"Facto {i} sobre comboios." for i in range(30)]
    joined = "\n\n".join(split_text("\n\n".join(paragraphs), chunk_size=80, overlap=0))
    assert all(p in joined for p in paragraphs)


def test_invalid_overlap():
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=10, overlap=10)
