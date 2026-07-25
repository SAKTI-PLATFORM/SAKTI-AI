"""Prompt construction, structured validation, and retry orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from src.core.config import settings
from src.domain.double_diamond.schema import (
    AnalyzePhaseRequest,
    AnalyzePhaseResponse,
    DoubleDiamondPhase,
    GenerateQuestionRequest,
    GenerateQuestionResponse,
)
from src.infrastructure.llm.double_diamond_provider import (
    DeepSeekProvider,
    LLMProvider,
)

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "infrastructure" / "prompts"
T = TypeVar("T", bound=BaseModel)

PHASE_INSTRUCTIONS = {
    DoubleDiamondPhase.DIVERGE_1: "Fokus pada aktivitas, nilai kerja, lingkungan, dampak, pengalaman, serta breadth versus depth. Jangan meminta memilih role.",
    DoubleDiamondPhase.CONVERGE_1: "Gunakan jawaban DIVERGE_1 untuk mengonfirmasi 3–5 bidang karier dan selalu sediakan opsi bidang custom.",
    DoubleDiamondPhase.DIVERGE_2: "Gunakan bidang terpilih untuk mendalami tipe masalah, cara berpikir, posisi dalam alur kerja, cara belajar, kontribusi tim, ambiguitas, dan lintas fungsi.",
    DoubleDiamondPhase.CONVERGE_2: "Bentuk kandidat role lalu konfirmasi role target, kesiapan, hambatan, kekuatan, waktu persiapan, dan hari kerja ideal.",
}

ANALYSIS_INSTRUCTIONS = {
    DoubleDiamondPhase.DIVERGE_1: "Simpulkan 3–5 detected_fields dan strengths dari jawaban eksplorasi luas.",
    DoubleDiamondPhase.CONVERGE_1: "Evaluasi konsistensi bidang yang dipilih dengan evidence dan perbarui strengths bila perlu.",
    DoubleDiamondPhase.DIVERGE_2: "Simpulkan kandidat recommended_roles awal berdasarkan bidang terpilih dan pola kontribusi.",
    DoubleDiamondPhase.CONVERGE_2: "Hasilkan recommended_roles, strengths, barriers, career_summary, work_style_summary, dan readiness_summary final.",
}


class DoubleDiamondService:
    def __init__(self, provider: LLMProvider | None = None) -> None:
        self.provider = provider or DeepSeekProvider()

    async def generate_questions(
        self,
        request: GenerateQuestionRequest,
    ) -> GenerateQuestionResponse:
        context = sanitize_context(request.context.model_dump())
        user_prompt = json.dumps(
            {
                "task": "generate_double_diamond_questions",
                "phase": request.phase,
                "question_count": request.question_count,
                "phase_instruction": PHASE_INSTRUCTIONS[request.phase],
                "context": context,
                "required_output": {
                    "model_name": self.provider.model_name,
                    "prompt_version": request.prompt_version,
                    "questions": "exactly requested question_count items",
                },
            },
            ensure_ascii=False,
        )
        result = await self._generate_validated(
            system_prompt=_load_prompt("dd_question_v1.txt"),
            user_prompt=user_prompt,
            model=GenerateQuestionResponse,
            expected_question_count=request.question_count,
        )
        result.model_name = self.provider.model_name
        result.prompt_version = request.prompt_version
        return result

    async def analyze_phase(
        self,
        request: AnalyzePhaseRequest,
    ) -> AnalyzePhaseResponse:
        user_prompt = json.dumps(
            {
                "task": "analyze_double_diamond_phase",
                "phase": request.phase,
                "phase_instruction": ANALYSIS_INSTRUCTIONS[request.phase],
                "context": sanitize_context(request.context.model_dump()),
                "phase_answers": request.phase_answers,
                "selection": request.selection,
                "user_facing_language_rules": {
                    "language": "Bahasa Indonesia natural dan mudah dipahami",
                    "never_expose_internal_codes": True,
                    "describe_evidence_semantically": True,
                    "machine_code_only_allowed_in_code_fields": True,
                },
            },
            ensure_ascii=False,
        )
        result = await self._generate_validated(
            system_prompt=_load_prompt("dd_analyze_v1.txt"),
            user_prompt=user_prompt,
            model=AnalyzePhaseResponse,
            expected_phase=request.phase,
        )
        result.phase = request.phase
        return result

    async def _generate_validated(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: type[T],
        expected_question_count: int | None = None,
        expected_phase: DoubleDiamondPhase | None = None,
    ) -> T:
        last_error: Exception | None = None
        for attempt in range(settings.llm_max_retries + 1):
            retry_instruction = ""
            if attempt:
                retry_instruction = (
                    "\nOutput sebelumnya tidak valid. Keluarkan ulang JSON lengkap "
                    "yang persis mengikuti schema."
                )
            try:
                payload = await self.provider.generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt + retry_instruction,
                    schema=model.model_json_schema(),
                )
                result = model.model_validate(payload)
                if (
                    expected_question_count is not None
                    and isinstance(result, GenerateQuestionResponse)
                    and len(result.questions) != expected_question_count
                ):
                    raise ValueError("LLM returned an unexpected question count")
                if (
                    expected_phase is not None
                    and isinstance(result, AnalyzePhaseResponse)
                    and result.phase != expected_phase
                ):
                    raise ValueError("LLM returned an unexpected analysis phase")
                return result
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
        raise ValueError("Invalid structured output after retry") from last_error


def sanitize_context(value: Any) -> Any:
    sensitive_keys = {
        "email",
        "phone",
        "phone_number",
        "address",
        "password",
        "token",
        "raw_cv",
    }
    if isinstance(value, dict):
        return {
            key: sanitize_context(item)
            for key, item in value.items()
            if key.lower() not in sensitive_keys
        }
    if isinstance(value, list):
        return [sanitize_context(item) for item in value]
    return value


def _load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()
