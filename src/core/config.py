"""Runtime settings for SAKTI-AI."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

_DEFAULT_SCORING_WEIGHTS = {
    "skill": 0.4,
    "experience": 0.2,
    "education": 0.2,
    "riasec": 0.1,
    "ocean": 0.05,
    "preference": 0.05,
}

_DEFAULT_LLM_MODEL_PER_NODE = {
    "_explains": "deepseek-chat",
    "recommends_course": "deepseek-chat",
    "recommends_cert": "deepseek-chat",
    "search_roles": "deepseek-chat",
    "market_demand": "deepseek-chat",
}


def _parse_json_env(key: str, default: dict) -> dict:
    raw = os.getenv(key)
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return default


@dataclass(frozen=True)
class Settings:
    # ── App ──────────────────────────────────────────────
    app_env: str = os.getenv("APP_ENV", "development")
    port: int = int(os.getenv("PORT", "8001"))

    # ── DeepSeek LLM ────────────────────────────────────
    deepseek_base_url: str = os.getenv(
        "DEEPSEEK_BASE_URL",
        "https://api.deepseek.com",
    )
    deepseek_api_key: str | None = os.getenv("DEEPSEEK_API_KEY")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    deepseek_max_tokens: int = int(os.getenv("DEEPSEEK_MAX_TOKENS", "12000"))
    deepseek_timeout_seconds: float = float(
        os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "120")
    )

    # ── Langfuse Observability ──────────────────────────
    langfuse_public_key: str | None = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = os.getenv("LANGFUSE_SECRET_KEY")
    langfuse_host: str = os.getenv(
        "LANGFUSE_HOST", "https://cloud.langfuse.com"
    )

    # ── Redis ───────────────────────────────────────────
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "86400"))

    # ── JobMatcher Scoring ──────────────────────────────
    match_threshold: float = float(os.getenv("MATCH_THRESHOLD", "0.70"))
    jobmatcher_scoring_weights: dict = field(
        default_factory=lambda: _parse_json_env(
            "JOBMATCHER_SCORING_WEIGHTS", _DEFAULT_SCORING_WEIGHTS
        )
    )

    # ── LLM per Node ───────────────────────────────────
    llm_model_per_node: dict = field(
        default_factory=lambda: _parse_json_env(
            "LLM_MODEL_PER_NODE", _DEFAULT_LLM_MODEL_PER_NODE
        )
    )

    # ── Embedding / Vector DB ───────────────────────────
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "text-embedding-3-small"
    )

    # ── TalentForger ────────────────────────────────────
    min_resources_per_gap: int = int(
        os.getenv("MIN_RESOURCES_PER_GAP", "3")
    )


settings = Settings()
