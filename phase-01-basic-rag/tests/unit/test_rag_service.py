import asyncio

from rag_from_scratch.services.rag import NO_CONTEXT_ANSWER, RagService
from tests.fakes import FakeChat, FakeEmbedder, InMemoryStore

DOC = """\
The Velaro train has a maximum speed of 320 km/h.

Preventive maintenance of the bogies is done every 1.2 million km.

The signalling system used on the line is ETCS level 2.
"""


def make_service() -> tuple[RagService, InMemoryStore, FakeChat]:
    store, chat = InMemoryStore(), FakeChat()
    service = RagService(FakeEmbedder(), store, chat, chunk_size=80, chunk_overlap=0, top_k=2)
    return service, store, chat


def test_ingest_creates_chunks():
    service, store, _ = make_service()
    result = asyncio.run(service.ingest("velaro.txt", DOC))
    assert result.chunks_indexed == 3
    assert len(store.items) == 3
    assert {c.source for c, _ in store.items} == {"velaro.txt"}


def test_ask_sends_most_relevant_chunk_to_llm():
    service, _, chat = make_service()
    asyncio.run(service.ingest("velaro.txt", DOC))

    result = asyncio.run(service.ask("Which signalling system is used on the line?"))

    assert result.answer == "answer [1]"
    assert "ETCS" in result.sources[0].excerpt  # the right chunk comes first
    _, user_prompt = chat.calls[0]
    assert "ETCS level 2" in user_prompt


def test_ask_without_documents_does_not_call_llm():
    service, _, chat = make_service()
    result = asyncio.run(service.ask("What is the maximum speed?"))
    assert result.answer == NO_CONTEXT_ANSWER
    assert result.sources == []
    assert chat.calls == []
