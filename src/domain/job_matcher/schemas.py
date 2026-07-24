"""Pydantic output schemas for the JobMatcher pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class RoleReference(BaseModel):
    """A candidate career role to match against the user profile."""

    role_id: str
    role_name: str
    role_category: str
    role_level: str = "Junior"
    description: str = ""
    required_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    riasec_ideal: str | None = None  # e.g. "RIA"
    min_experience_months: int = 0


class SimilarityResult(BaseModel):
    """Raw skill-overlap similarity between user and a role."""

    role_id: str
    raw_similarity: float = Field(ge=0, le=1)
    matched_skills: list[str] = Field(default_factory=list)
    unmatched_skills: list[str] = Field(default_factory=list)


class ScoreDetail(BaseModel):
    """Per-dimension breakdown of the career match score."""

    match_id: str
    skill_match_score: float = 0
    experience_project_score: float = 0
    education_score: float = 0
    riasec_fit_score: float = 0
    ocean_workstyle_score: float = 0
    preference_score: float = 0


class CareerMatchResult(BaseModel):
    """Final career match result linking a user to a role."""

    match_id: str
    user_id: str
    role_id: str
    role_name: str
    total_match_score: float = Field(ge=0, le=100)
    match_reason: str = ""
    created_at: datetime = Field(default_factory=_now_utc)


class SkillGapResult(BaseModel):
    """Identified gap between user's current skill level and role requirement."""

    gap_id: str
    match_id: str
    skill_name: str
    current_level: Literal["None", "Beginner", "Intermediate", "Advanced"]
    required_level: Literal["Beginner", "Intermediate", "Advanced"]
    gap_level: Literal["Low", "Medium", "High"]
    priority: Literal["Low", "Medium", "High"]
    reason: str = ""


class MarketDemand(BaseModel):
    """Market demand signal for a skill."""

    skill_name: str
    demand_score: float = Field(ge=0, le=1)
    trending: bool = False


# ── LLM structured output schemas ──────────────────────


class GapReason(BaseModel):
    """LLM-generated reason for a specific skill gap."""

    gap_id: str
    reason: str


class MatchExplanation(BaseModel):
    """LLM-generated explanation for a career match."""

    match_id: str
    match_reason: str
    gap_reasons: list[GapReason] = Field(default_factory=list)


class RoleSearchResult(BaseModel):
    """LLM-generated list of candidate roles."""

    roles: list[RoleReference]


class MarketDemandResult(BaseModel):
    """LLM-generated market demand analysis."""

    demands: list[MarketDemand]
