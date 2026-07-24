"""Skill evidence value object — produced by SkillParser, consumed by JobMatcher."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SkillEvidence(BaseModel):
    """A single skill detected from CV parsing with confidence metadata."""

    detected_text: str
    confidence_score: float = Field(ge=0, le=1)
    learning_hours: int | None = None
    working_hours: int | None = None
    evidence_source: str = "cv_text"
    evidence_strength: Literal["low", "medium", "high"] = "medium"

    # Optional IDs when loaded from the profile database
    user_skill_id: str | None = None
    skill_id: str | None = None
