"""DeepSeek client for structured CV extraction."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import httpx

from src.core.config import settings
from src.domain.cvparser.schema import ParsedCVResult

logger = logging.getLogger("uvicorn.error")

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


BASE_SYSTEM_PROMPT = _load_prompt("cv_extraction_system.txt")
SKILL_HOURS_PROMPT = _load_prompt("cv_skill_hours.txt")
SYSTEM_PROMPT = f"{BASE_SYSTEM_PROMPT}\n\n{SKILL_HOURS_PROMPT}"
JSON_SCHEMA_GUIDE = _load_prompt("cv_extraction_json_schema.txt")
COMPACT_OUTPUT_INSTRUCTION = _load_prompt("cv_extraction_compact_retry.txt")


def is_deepseek_configured() -> bool:
    return bool(settings.deepseek_api_key)


async def extract_cv_with_deepseek(cv_text: str) -> ParsedCVResult:
    if not settings.deepseek_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")

    endpoint = f"{settings.deepseek_base_url.rstrip('/')}/chat/completions"
    logger.info(
        (
            "[DeepSeek] Preparing CV extraction: model=%s text_length=%d "
            "max_tokens=%d"
        ),
        settings.deepseek_model,
        len(cv_text),
        settings.deepseek_max_tokens,
    )

    async with httpx.AsyncClient(timeout=settings.deepseek_timeout_seconds) as client:
        for attempt in range(2):
            compact_output = attempt == 1
            logger.info(
                "[DeepSeek] Sending request: attempt=%d compact_output=%s",
                attempt + 1,
                compact_output,
            )
            response = await client.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {settings.deepseek_api_key}",
                    "Content-Type": "application/json",
                },
                json=_build_request_payload(cv_text, compact_output),
            )

            logger.info(
                "[DeepSeek] Response received: attempt=%d status=%d",
                attempt + 1,
                response.status_code,
            )
            if response.is_error:
                logger.error(
                    "[DeepSeek] Request failed: status=%d body=%s",
                    response.status_code,
                    response.text[:1000],
                )
            response.raise_for_status()
            payload = response.json()
            finish_reason = _read_finish_reason(payload)
            logger.info(
                (
                    "[DeepSeek] Completion metadata: attempt=%d id=%s "
                    "finish_reason=%s usage=%s"
                ),
                attempt + 1,
                payload.get("id"),
                finish_reason,
                payload.get("usage"),
            )

            if finish_reason == "length":
                if not compact_output:
                    logger.warning(
                        "[DeepSeek] Output truncated; retrying with compact output"
                    )
                    continue
                raise ValueError(
                    "DeepSeek response remained truncated after compact retry"
                )

            try:
                content = _read_message_content(payload)
                result = ParsedCVResult.model_validate(_load_json_object(content))
            except (json.JSONDecodeError, ValueError) as exc:
                if not compact_output:
                    logger.warning(
                        "[DeepSeek] Invalid structured output; retrying compactly: %s",
                        exc,
                    )
                    continue
                raise

            _log_success(result)
            return result

    raise RuntimeError("DeepSeek CV extraction exhausted all attempts")


def _build_request_payload(cv_text: str, compact_output: bool) -> dict:
    user_instructions = JSON_SCHEMA_GUIDE
    if compact_output:
        user_instructions = f"{COMPACT_OUTPUT_INSTRUCTION}\n{user_instructions}"

    return {
        "model": settings.deepseek_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{user_instructions}\n\n"
                    "CV_TEXT:\n"
                    f"{cv_text[:25000]}"
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": settings.deepseek_max_tokens,
        "response_format": {"type": "json_object"},
        "stream": False,
    }


def _read_finish_reason(payload: dict) -> str | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None
    finish_reason = first_choice.get("finish_reason")
    return finish_reason if isinstance(finish_reason, str) else None


def _log_success(result: ParsedCVResult) -> None:
    logger.info(
        (
            "[DeepSeek] CV extraction successful: confidence=%.2f "
            "educations=%d experiences=%d projects=%d certifications=%d skills=%d"
        ),
        result.confidence_score,
        len(result.educations),
        len(result.experiences),
        len(result.projects),
        len(result.certifications),
        len(result.skills),
    )


def _read_message_content(payload: dict) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("DeepSeek response does not contain choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("DeepSeek choice must be an object")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("DeepSeek choice does not contain message")

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("DeepSeek message content is empty")
    return content


def _load_json_object(content: str) -> dict:
    cleaned = _strip_reasoning_and_fences(content)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))

    if not isinstance(value, dict):
        raise ValueError("LLM response must be a JSON object")
    return value


def _strip_reasoning_and_fences(content: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    cleaned = re.sub(r".*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    return cleaned
