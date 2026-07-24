"""Pydantic output schemas for the TalentForger pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class LearningPath(BaseModel):
    """A structured learning roadmap targeting a specific career role."""

    learning_path_id: str
    match_id: str
    target_role: str
    learning_path_type: str = "structured"
    estimated_duration_weeks: int = 4
    created_at: datetime = Field(default_factory=datetime.utcnow)


class LearningPathStep(BaseModel):
    """A single step within a learning path."""

    step_id: str
    learning_path_id: str
    gap_id: str
    step_order: int
    week: int
    topic: str
    objective: str
    related_skill_name: str


class LearningResource(BaseModel):
    """A course, certification, article, or video resource."""

    resource_id: str
    skill_name: str
    resource_title: str
    resource_type: Literal["Course", "Certification", "Article", "Video"]
    provider: str
    difficulty_level: Literal["Beginner", "Intermediate", "Advanced"]
    estimated_duration_hours: int = 0
    url: str = ""


class ResourceRecommendation(BaseModel):
    """Links a learning resource to a learning path step with reasoning."""

    recommendation_id: str
    step_id: str
    resource_id: str
    recommendation_reason: str = ""
    priority_order: int = 1


# ── LLM structured output schemas ──────────────────────


class CourseSearchResult(BaseModel):
    """LLM-generated course search results."""

    resources: list[LearningResource]


class CertSearchResult(BaseModel):
    """LLM-generated certification search results."""

    resources: list[LearningResource]


class LearningPathPlan(BaseModel):
    """LLM-generated learning path with steps and recommendations."""

    learning_path: LearningPath
    steps: list[LearningPathStep]
    recommendations: list[ResourceRecommendation]
