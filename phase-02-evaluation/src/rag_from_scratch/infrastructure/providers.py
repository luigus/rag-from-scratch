"""Chooses the LLM provider from the settings.

This is the only place that knows about Gemini or OpenAI. Everything else
works with the Embedder and ChatModel protocols.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from rag_from_scratch.core.config import Settings
from rag_from_scratch.domain.ports import ChatModel, Embedder


@dataclass
class LLMProvider:
    embedder: Embedder
    make_chat: Callable[[str], ChatModel]  # model name → chat model (the judge uses another model)
    close: Callable[[], Awaitable[None]]


def build_provider(settings: Settings) -> LLMProvider:
    # imports inside the branches: only the SDK of the chosen provider is loaded
    if settings.llm_provider == "gemini":
        from google import genai

        from rag_from_scratch.infrastructure.gemini_client import GeminiChat, GeminiEmbedder

        gemini = genai.Client(api_key=settings.gemini_api_key)
        return LLMProvider(
            embedder=GeminiEmbedder(gemini, settings.embedding_model, settings.embedding_dim),
            make_chat=lambda model: GeminiChat(gemini, model),
            close=gemini.aio.aclose,
        )

    from openai import AsyncOpenAI

    from rag_from_scratch.infrastructure.openai_client import OpenAIChat, OpenAIEmbedder

    openai = AsyncOpenAI(api_key=settings.openai_api_key)
    return LLMProvider(
        embedder=OpenAIEmbedder(openai, settings.embedding_model),
        make_chat=lambda model: OpenAIChat(openai, model),
        close=openai.close,
    )
