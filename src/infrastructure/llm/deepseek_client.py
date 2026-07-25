"""Shared DeepSeek LLM client via langchain-openai (OpenAI-compatible API)."""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI

from src.core.config import settings

logger = logging.getLogger("uvicorn.error")


def get_deepseek_llm(
    model: str | None = None,
    *,
    max_tokens: int | None = None,
    temperature: float = 0,
    **kwargs,
) -> ChatOpenAI:
    """Create a DeepSeek LLM instance via the OpenAI-compatible endpoint.

    Parameters
    ----------
    model : str, optional
        Override the default model name from settings.
    max_tokens : int, optional
        Override the default max tokens.
    temperature : float
        Sampling temperature (default: 0 for deterministic output).
    """
    if not settings.deepseek_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")

    resolved_model = model or settings.deepseek_model
    resolved_max_tokens = max_tokens or settings.deepseek_max_tokens

    logger.debug(
        "[DeepSeek Client] Creating LLM: model=%s max_tokens=%d",
        resolved_model,
        resolved_max_tokens,
    )

    return ChatOpenAI(
        model=resolved_model,
        base_url=settings.deepseek_base_url,
        api_key=settings.deepseek_api_key,
        max_tokens=resolved_max_tokens,
        temperature=temperature,
        timeout=settings.deepseek_timeout_seconds,
        **kwargs,
    )
