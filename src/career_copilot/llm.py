"""One place to construct the chat model, so every node builds it the same way."""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from career_copilot.config import Settings


def get_chat_model(
    settings: Settings,
    model_name: str | None = None,
    temperature: float = 0.0,
) -> ChatOpenAI:
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=model_name or settings.chat_model,
        temperature=temperature,
    )
