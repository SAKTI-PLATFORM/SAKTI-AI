from __future__ import annotations

from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase

from pydantic import ValidationError

from src.domain.double_diamond.schema import (
    AnalyzePhaseRequest,
    GenerateQuestionRequest,
    GeneratedQuestion,
)
from src.domain.double_diamond.service import (
    DoubleDiamondService,
    sanitize_context,
)
from src.infrastructure.llm.double_diamond_provider import DeepSeekProvider


class FakeProvider:
    model_name = "fake-model"

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.calls = 0
        self.prompts: list[str] = []
        self.system_prompts: list[str] = []

    async def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        del schema
        self.system_prompts.append(system_prompt)
        self.prompts.append(user_prompt)
        response = self.responses[self.calls]
        self.calls += 1
        return response


def valid_question(code: str, order: int) -> dict[str, Any]:
    return {
        "question_code": code,
        "question_text": "Pertanyaan eksplorasi karier?",
        "helper_text": None,
        "response_type": "SINGLE_CHOICE",
        "options": [
            {"code": "A", "label": "Pilihan A"},
            {"code": "B", "label": "Pilihan B"},
        ],
        "min_selection": None,
        "max_selection": None,
        "scale_min": None,
        "scale_max": None,
        "question_order": order,
        "is_required": True,
    }


class DoubleDiamondSchemaTest(TestCase):
    def test_rejects_choice_without_options(self) -> None:
        with self.assertRaises(ValidationError):
            GeneratedQuestion.model_validate(
                {
                    "question_code": "D1-Q1",
                    "question_text": "Pilih aktivitas",
                    "response_type": "SINGLE_CHOICE",
                    "options": [],
                    "question_order": 1,
                }
            )

    def test_rejects_max_selection_above_option_count(self) -> None:
        with self.assertRaises(ValidationError):
            GeneratedQuestion.model_validate(
                {
                    "question_code": "D1-Q1",
                    "question_text": "Pilih aktivitas",
                    "response_type": "MULTI_CHOICE",
                    "options": [
                        {"code": "A", "label": "A"},
                        {"code": "B", "label": "B"},
                    ],
                    "min_selection": 1,
                    "max_selection": 3,
                    "question_order": 1,
                }
            )

    def test_sanitizes_pii_recursively(self) -> None:
        result = sanitize_context(
            {
                "profile": {
                    "email": "person@example.com",
                    "phone_number": "0812",
                    "summary": "Engineer",
                },
                "previous_answers": [{"token": "secret", "answer": "A"}],
            }
        )
        self.assertEqual(result["profile"], {"summary": "Engineer"})
        self.assertEqual(result["previous_answers"], [{"answer": "A"}])


class DoubleDiamondServiceTest(IsolatedAsyncioTestCase):
    async def test_uses_deepseek_as_the_default_provider(self) -> None:
        service = DoubleDiamondService()

        self.assertIsInstance(service.provider, DeepSeekProvider)

    async def test_retries_invalid_structured_output_once(self) -> None:
        invalid = {
            "model_name": "fake-model",
            "prompt_version": "DD-QUESTION-V1.0",
            "questions": [],
        }
        valid = {
            "model_name": "fake-model",
            "prompt_version": "DD-QUESTION-V1.0",
            "questions": [valid_question("D1-Q1", 1), valid_question("D1-Q2", 2)],
        }
        provider = FakeProvider([invalid, valid])
        service = DoubleDiamondService(provider)
        request = GenerateQuestionRequest.model_validate(
            {
                "onboarding_session_id": "session-1",
                "phase": "DIVERGE_1",
                "framework_version": "DD-V1.0",
                "prompt_version": "DD-QUESTION-V1.0",
                "language": "id",
                "question_count": 2,
                "context": {
                    "profile": {"summary": "Engineer", "email": "hidden@example.com"},
                    "assessment": {},
                },
            }
        )

        result = await service.generate_questions(request)

        self.assertEqual(provider.calls, 2)
        self.assertEqual(len(result.questions), 2)
        self.assertNotIn("hidden@example.com", provider.prompts[-1])
        self.assertIn("Output sebelumnya tidak valid", provider.prompts[-1])

    async def test_analysis_prompt_forbids_internal_codes_in_user_text(self) -> None:
        provider = FakeProvider(
            [
                {
                    "phase": "CONVERGE_2",
                    "recommended_roles": [
                        {
                            "code": "ML_ENGINEER",
                            "label": "Machine Learning Engineer",
                            "score": 0.9,
                            "reason": "Pengalaman teknis dan minat membangun solusi mendukung peran ini.",
                        }
                    ],
                    "strengths": ["Kemampuan teknis"],
                    "barriers": ["Perlu memperluas jejaring profesional"],
                    "career_summary": "Profil sesuai untuk peran teknis berbasis produk.",
                    "work_style_summary": "Produktif dalam kolaborasi terstruktur.",
                    "readiness_summary": "Siap dengan penguatan portofolio.",
                    "confidence_score": 0.9,
                }
            ]
        )
        service = DoubleDiamondService(provider)
        request = AnalyzePhaseRequest.model_validate(
            {
                "onboarding_session_id": "session-1",
                "phase": "CONVERGE_2",
                "framework_version": "DD-V1.0",
                "prompt_version": "DD-QUESTION-V1.1",
                "context": {"profile": {}, "assessment": {}},
                "phase_answers": [
                    {
                        "question_code": "D2-Q1",
                        "question_text": "Kontribusi seperti apa yang Anda sukai?",
                        "answer": "Terlibat dari awal hingga akhir pengembangan produk",
                    }
                ],
                "selection": "Machine Learning Engineer",
            }
        )

        await service.analyze_phase(request)

        self.assertIn("never_expose_internal_codes", provider.prompts[-1])
        self.assertIn("Jangan tampilkan atau mengutip kode internal", provider.system_prompts[-1])
