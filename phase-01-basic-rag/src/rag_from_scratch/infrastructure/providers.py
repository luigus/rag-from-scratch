"""Builds an LLM provider (Gemini, OpenAI or a local Ollama) from the settings.

This is the only place that knows about specific providers. Everything else
works with the Embedder and ChatModel protocols.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from rag_from_scratch.core.config import ProviderName, Settings
from rag_from_scratch.domain.ports import ChatModel, Embedder


@dataclass
class LLMProvider:
    name: ProviderName
    embedder: Embedder
    make_chat: Callable[[str], ChatModel]  # model name → chat model (the judge uses another model)
    close: Callable[[], Awaitable[None]]


def build_provider(name: ProviderName, settings: Settings) -> LLMProvider:
    # imports inside the branches: only the SDK of the chosen provider is loaded
    if name == "gemini":
        from google import genai

        from rag_from_scratch.infrastructure.gemini_client import GeminiChat, GeminiEmbedder

        gemini = genai.Client(api_key=settings.gemini_api_key)
        return LLMProvider(
            name=name,
            embedder=GeminiEmbedder(gemini, settings.embedding_model, settings.embedding_dim),
            make_chat=lambda model: GeminiChat(gemini, model),
            close=gemini.aio.aclose,
        )

    from openai import AsyncOpenAI

    from rag_from_scratch.infrastructure.openai_client import OpenAIChat, OpenAIEmbedder

    if name == "ollama":
        # Ollama speaks the OpenAI API, so the OpenAI classes work unchanged:
        # only the address changes, and the key is ignored by the local server
        client = AsyncOpenAI(base_url=f"{settings.ollama_url.rstrip('/')}/v1", api_key="ollama")
    else:
        client = AsyncOpenAI(api_key=settings.openai_api_key)

    return LLMProvider(
        name=name,
        embedder=OpenAIEmbedder(client, settings.embedding_model),
        make_chat=lambda model: OpenAIChat(client, model),
        close=client.close,
    )
