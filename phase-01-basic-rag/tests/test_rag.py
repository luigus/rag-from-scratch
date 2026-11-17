import asyncio

from fakes import FakeChat, FakeEmbedder, InMemoryStore

from rag_from_scratch.rag import NO_CONTEXT_ANSWER, RagService

DOC = """\
O comboio Velaro tem velocidade máxima de 320 km/h.

A manutenção preventiva dos bogies é feita a cada 1,2 milhões de km.

O sistema de sinalização usado na linha é o ETCS nível 2.
"""


def make_service(chat: FakeChat | None = None) -> tuple[RagService, InMemoryStore, FakeChat]:
    store, chat = InMemoryStore(), chat or FakeChat()
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

    result = asyncio.run(service.ask("Qual é o sistema de sinalização da linha?"))

    assert result.answer == "resposta [1]"
    assert "ETCS" in result.sources[0].excerpt  # o chunk certo vem em primeiro
    _, user_prompt = chat.calls[0]
    assert "ETCS nível 2" in user_prompt


def test_ask_without_documents_does_not_call_llm():
    service, _, chat = make_service()
    result = asyncio.run(service.ask("Qual é a velocidade?"))
    assert result.answer == NO_CONTEXT_ANSWER
    assert result.sources == []
    assert chat.calls == []
