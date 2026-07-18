"""OpenRouter client for structured CV extraction."""

from __future__ import annotations

import json
import logging
import re

import httpx

from src.core.config import settings
from src.domain.cvparser.schema import ParsedCVResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Kamu adalah CVParser untuk onboarding job seeker.

Tugasmu hanya mengekstrak data dari CV mentah menjadi JSON yang valid.
Jangan mengarang data. Kalau data tidak ada, isi null atau array kosong.
Jangan menulis markdown, penjelasan, heading, atau teks di luar JSON.

Aturan field:
- confidence_score: angka 0 sampai 1 berdasarkan kelengkapan dan kejelasan CV.
- education_level hanya boleh salah satu: SMA, D3, S1, S2, S3, atau null.
- experience_type gunakan WORK, INTERNSHIP, ORGANIZATION, VOLUNTEER, FREELANCE, atau OTHER.
- start_date dan end_date gunakan YYYY-MM-DD jika tanggal lengkap, YYYY-01-01 jika hanya tahun, atau null.
- skills belum menjadi fokus utama, ekstrak seperlunya saja dari CV.
"""

JSON_SCHEMA_GUIDE = """Return JSON dengan bentuk persis seperti ini:
{
  "confidence_score": 0.85,
  "educations": [
    {
      "education_level": "S1",
      "institution": "Universitas Indonesia",
      "major": "Computer Science",
      "degree": "Bachelor of Computer Science",
      "start_year": 2020,
      "end_year": 2024,
      "gpa": 3.8,
      "is_current": false
    }
  ],
  "experiences": [
    {
      "title": "Backend Developer",
      "organization": "Acme",
      "experience_type": "WORK",
      "start_date": "2024-01-01",
      "end_date": null,
      "is_current": true,
      "duration_months": null,
      "description": "Built REST APIs"
    }
  ],
  "projects": [
    {
      "project_name": "BI Hackathon Platform",
      "description": "Built onboarding backend",
      "tools_used": "NestJS, MySQL",
      "start_date": null,
      "end_date": null
    }
  ],
  "certifications": [
    {
      "certification_name": "AWS Cloud Practitioner",
      "issuer": "Amazon Web Services",
      "issued_year": 2024,
      "certificate_url": null
    }
  ],
  "skills": [
    {
      "detected_text": "TypeScript",
      "inferred_level": null,
      "confidence_score": 0.7,
      "evidence_source": "cv_text",
      "evidence_strength": "medium"
    }
  ]
}
"""


def is_openrouter_configured() -> bool:
    return bool(settings.llm_api_key)


async def extract_cv_with_openrouter(cv_text: str) -> ParsedCVResult:
    if not settings.llm_api_key:
        raise RuntimeError("LLM_API_KEY is not configured")

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{settings.llm_api_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.llm_api_key}",
                "Content-Type": "application/json",
                "X-Title": "SAKTI-AI CVParser",
            },
            json={
                "model": settings.llm_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"{JSON_SCHEMA_GUIDE}\n\n"
                            "CV_TEXT:\n"
                            f"{cv_text[:25000]}"
                        ),
                    },
                ],
                "temperature": 0,
                "max_tokens": 3000,
                "response_format": {"type": "json_object"},
                "reasoning": {"enabled": False},
            },
        )
    response.raise_for_status()
    payload = response.json()
    content = _read_message_content(payload)
    return ParsedCVResult.model_validate(_load_json_object(content))


def _read_message_content(payload: dict) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenRouter response does not contain choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("OpenRouter choice must be an object")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("OpenRouter choice does not contain message")

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("OpenRouter message content is empty")
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
