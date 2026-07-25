"""Validated internal API contracts for Double Diamond generation."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class DoubleDiamondPhase(StrEnum):
    DIVERGE_1 = "DIVERGE_1"
    CONVERGE_1 = "CONVERGE_1"
    DIVERGE_2 = "DIVERGE_2"
    CONVERGE_2 = "CONVERGE_2"


class ResponseType(StrEnum):
    SINGLE_CHOICE = "SINGLE_CHOICE"
    MULTI_CHOICE = "MULTI_CHOICE"
    RANKING = "RANKING"
    SCALE = "SCALE"
    TEXT = "TEXT"


class DoubleDiamondContext(BaseModel):
    profile: dict[str, Any] = Field(default_factory=dict)
    assessment: dict[str, Any] = Field(default_factory=dict)
    previous_answers: list[dict[str, Any]] = Field(default_factory=list)
    selected_field: str | None = None
    current_result: dict[str, Any] = Field(default_factory=dict)


class GenerateQuestionRequest(BaseModel):
    onboarding_session_id: str = Field(min_length=1, max_length=64)
    phase: DoubleDiamondPhase
    framework_version: str = Field(min_length=1, max_length=40)
    prompt_version: str = Field(min_length=1, max_length=40)
    language: str = Field(default="id", pattern="^id$")
    question_count: int = Field(ge=2, le=7)
    context: DoubleDiamondContext


class AnalyzePhaseRequest(BaseModel):
    onboarding_session_id: str = Field(min_length=1, max_length=64)
    phase: DoubleDiamondPhase
    framework_version: str = Field(min_length=1, max_length=40)
    prompt_version: str = Field(min_length=1, max_length=40)
    context: DoubleDiamondContext
    phase_answers: list[dict[str, Any]] = Field(min_length=1)
    selection: str | None = None


class GeneratedOption(BaseModel):
    code: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=240)


class GeneratedQuestion(BaseModel):
    question_code: str = Field(min_length=1, max_length=40)
    question_text: str = Field(min_length=1)
    helper_text: str | None = None
    response_type: ResponseType
    options: list[GeneratedOption] = Field(default_factory=list)
    min_selection: int | None = None
    max_selection: int | None = None
    scale_min: float | None = None
    scale_max: float | None = None
    question_order: int = Field(ge=1)
    is_required: bool = True

    @model_validator(mode="after")
    def validate_by_type(self) -> "GeneratedQuestion":
        choice_types = {
            ResponseType.SINGLE_CHOICE,
            ResponseType.MULTI_CHOICE,
            ResponseType.RANKING,
        }
        if self.response_type in choice_types and len(self.options) < 2:
            raise ValueError("Choice question requires at least two options")
        option_codes = [option.code for option in self.options]
        if len(option_codes) != len(set(option_codes)):
            raise ValueError("Option codes must be unique")
        if self.response_type == ResponseType.MULTI_CHOICE:
            if self.min_selection is None or self.max_selection is None:
                raise ValueError("MULTI_CHOICE requires min_selection and max_selection")
            if not 0 <= self.min_selection <= self.max_selection <= len(self.options):
                raise ValueError("Invalid MULTI_CHOICE selection bounds")
        if self.response_type == ResponseType.SCALE:
            if self.scale_min is None or self.scale_max is None:
                raise ValueError("SCALE requires min and max")
            if self.scale_min >= self.scale_max:
                raise ValueError("scale_min must be lower than scale_max")
        return self


class GenerateQuestionResponse(BaseModel):
    model_name: str
    prompt_version: str
    questions: list[GeneratedQuestion]

    @model_validator(mode="after")
    def validate_uniqueness(self) -> "GenerateQuestionResponse":
        codes = [question.question_code for question in self.questions]
        orders = [question.question_order for question in self.questions]
        if len(codes) != len(set(codes)):
            raise ValueError("Question codes must be unique")
        if len(orders) != len(set(orders)):
            raise ValueError("Question order must be unique")
        return self


class CareerCandidate(BaseModel):
    code: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=240)
    score: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1)


class AnalyzePhaseResponse(BaseModel):
    phase: DoubleDiamondPhase
    detected_fields: list[CareerCandidate] | None = None
    recommended_roles: list[CareerCandidate] | None = None
    strengths: list[str] = Field(default_factory=list)
    barriers: list[str] | None = None
    career_summary: str | None = None
    work_style_summary: str | None = None
    readiness_summary: str | None = None
    confidence_score: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def reject_diagnostic_language(self) -> "AnalyzePhaseResponse":
        combined = " ".join(
            value
            for value in [
                self.career_summary,
                self.work_style_summary,
                self.readiness_summary,
            ]
            if value
        ).lower()
        forbidden = ("diagnosis", "gangguan mental", "penyakit mental")
        if any(term in combined for term in forbidden):
            raise ValueError("Analysis must not contain diagnostic claims")
        if self.phase == DoubleDiamondPhase.DIVERGE_1 and not self.detected_fields:
            raise ValueError("DIVERGE_1 analysis requires detected_fields")
        if self.phase in {
            DoubleDiamondPhase.DIVERGE_2,
            DoubleDiamondPhase.CONVERGE_2,
        } and not self.recommended_roles:
            raise ValueError(f"{self.phase} analysis requires recommended_roles")
        if self.phase == DoubleDiamondPhase.CONVERGE_2 and not all(
            [self.career_summary, self.work_style_summary, self.readiness_summary]
        ):
            raise ValueError("CONVERGE_2 analysis requires final summaries")
        return self
