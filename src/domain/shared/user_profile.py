"""Unified user profile input — aggregates CV, assessment, and double-diamond data."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.domain.shared.skill_evidence import SkillEvidence


# ── Assessment value objects ────────────────────────────


class OceanScores(BaseModel):
    """Big Five personality trait scores (0-100 scale)."""

    openness: float = 0
    conscientiousness: float = 0
    extraversion: float = 0
    agreeableness: float = 0
    neuroticism: float = 0


class RiasecScores(BaseModel):
    """Holland RIASEC interest scores (0-100 scale)."""

    realistic: float = 0
    investigative: float = 0
    artistic: float = 0
    social: float = 0
    enterprising: float = 0
    conventional: float = 0
    dominant_code: str = ""


# ── User preferences ───────────────────────────────────


class UserPreferences(BaseModel):
    """Career & work-style preferences extracted from onboarding."""

    preferred_industries: list[str] = Field(default_factory=list)
    preferred_locations: list[str] = Field(default_factory=list)
    work_mode_pref: Literal["Remote", "Hybrid", "On-site", "Any"] = "Any"
    career_goal: str | None = None
    selected_field: str | None = None
    selected_role: str | None = None
    learning_style: str | None = None


# ── Double Diamond output ──────────────────────────────


class DetectedField(BaseModel):
    """A career field detected by Double Diamond analysis."""

    code: str
    label: str
    score: float = 0
    reason: str = ""


class RecommendedRole(BaseModel):
    """A career role recommended by Double Diamond analysis."""

    code: str
    label: str
    score: float = 0
    reason: str = ""


# ── Unified profile ────────────────────────────────────


class UserProfileInput(BaseModel):
    """
    Aggregated user profile consumed by JobMatcher and TalentForger.

    Built from the full onboarding export (CV parse + assessment + double diamond).
    """

    user_id: str
    cv_id: str
    onboarding_session_id: str

    # Identity
    full_name: str = ""
    professional_headline: str | None = None
    profile_summary: str | None = None

    # Parsed CV data
    skills: list[SkillEvidence] = Field(default_factory=list)
    educations: list[dict] = Field(default_factory=list)
    experiences: list[dict] = Field(default_factory=list)
    projects: list[dict] = Field(default_factory=list)
    certifications: list[dict] = Field(default_factory=list)

    # Assessment data
    ocean: OceanScores | None = None
    riasec: RiasecScores | None = None

    # Double Diamond data
    preferences: UserPreferences | None = None
    detected_fields: list[DetectedField] = Field(default_factory=list)
    recommended_roles: list[RecommendedRole] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    barriers: list[str] = Field(default_factory=list)
    career_summary: str | None = None
    work_style_summary: str | None = None
