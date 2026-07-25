"""DeepSeek integration for Double Diamond structured generation."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

import httpx

from src.core.config import settings


class LLMProvider(Protocol):
    model_name: str

    async def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]: ...


class DeepSeekProvider:
    model_name = settings.deepseek_model

    async def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        if not settings.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured")
        async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
            response = await client.post(
                f"{settings.deepseek_base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.deepseek_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": f"{user_prompt}\n\nJSON_SCHEMA:\n{json.dumps(schema, ensure_ascii=False)}",
                        },
                    ],
                    "temperature": 0.2,
                    "max_tokens": settings.deepseek_max_tokens,
                    "response_format": {"type": "json_object"},
                    "stream": False,
                },
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            return _load_json_object(content)


def _load_json_object(content: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("LLM response must be a JSON object")
    return value
