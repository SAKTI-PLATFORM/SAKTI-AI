"""Runtime settings for SAKTI-AI."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "development")
    port: int = int(os.getenv("PORT", "8001"))
    llm_api_url: str = os.getenv("LLM_API_URL", "https://openrouter.ai/api/v1")
    llm_api_key: str | None = os.getenv("LLM_API_KEY")
    llm_model: str = os.getenv("LLM_MODEL", "minimax/minimax-m2.5:free")


settings = Settings()
