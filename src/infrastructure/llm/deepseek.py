"""DeepSeek client for structured CV extraction."""

from __future__ import annotations

import json
import logging
import re

import httpx

from src.core.config import settings
from src.domain.cvparser.schema import ParsedCVResult

logger = logging.getLogger("uvicorn.error")

SYSTEM_PROMPT = """Kamu adalah CVParser untuk onboarding job seeker.

Tugasmu hanya mengekstrak data dari CV mentah menjadi JSON yang valid.
Jangan mengarang data. Kalau data tidak ada, isi null atau array kosong.
Jangan menulis markdown, penjelasan, heading, atau teks di luar JSON.

Aturan field:
- confidence_score: angka 0 sampai 1 berdasarkan kelengkapan dan kejelasan CV.
- personal_info hanya diambil dari header, bagian profil, atau kontak CV.
- full_name adalah nama kandidat, bukan nama perusahaan atau institusi.
- professional_headline adalah jabatan/tagline profesional singkat kandidat.
- email dan phone_number harus persis berdasarkan kontak yang tertulis di CV.
- domicile adalah kota/lokasi domisili kandidat, bukan lokasi perusahaan.
- linkedin_url hanya URL atau handle LinkedIn kandidat; jangan mengarang URL.
- profile_summary adalah ringkasan/about/profile kandidat, maksimal 600 karakter.
- education_level hanya boleh salah satu: SMA, D3, S1, S2, S3, atau null.
- experience_type gunakan WORK, INTERNSHIP, ORGANIZATION, VOLUNTEER, FREELANCE, atau OTHER.
- start_date dan end_date gunakan YYYY-MM-DD jika tanggal lengkap, YYYY-01-01 jika hanya tahun, atau null.
- skills belum menjadi fokus utama, ekstrak seperlunya saja dari CV.
"""

JSON_SCHEMA_GUIDE = """Return JSON dengan bentuk persis seperti ini:
{
  "confidence_score": 0.85,
  "personal_info": {
    "full_name": "Anargya Isadhi Maheswara",
    "professional_headline": "Software Engineer & AI Enthusiast",
    "email": "anargya@example.com",
    "phone_number": "+62 812-3456-7890",
    "domicile": "Bogor, Indonesia",
    "linkedin_url": "linkedin.com/in/anargya",
    "profile_summary": "Software engineer experienced in scalable web applications and AI-assisted products."
  },
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

COMPACT_OUTPUT_INSTRUCTION = """Respons sebelumnya terlalu panjang atau tidak valid.
Kembalikan satu objek JSON lengkap dan valid dengan format yang diminta.
Buat setiap description maksimal 200 karakter, hilangkan duplikasi, dan jangan
menambahkan detail yang tidak tersedia di CV. Prioritaskan JSON selesai daripada
deskripsi panjang.
"""


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
