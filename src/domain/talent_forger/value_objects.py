"""LangGraph state definition for the TalentForger graph."""

from __future__ import annotations

import operator
from typing import Annotated

from pydantic import BaseModel
from typing_extensions import TypedDict

from src.domain.shared.user_profile import UserPreferences, UserProfileInput
from src.domain.talent_forger.schemas import (
    LearningPath,
    LearningPathStep,
    LearningResource,
    ResourceRecommendation,
)


class TalentForgerState(TypedDict):
    """Shared state flowing through all TalentForger graph nodes.

    Fields with ``Annotated[list, operator.add]`` use the *append* reducer
    so that every node can return a partial list that gets accumulated.
    """

    # ── Inputs (set once at START) ──────────────────────
    match_id: str
    user_profile: UserProfileInput
    preferences: UserPreferences | None
    skill_gaps: list  # list of SkillGapResult dicts from JobMatcher

    # ── Accumulated by nodes ────────────────────────────
    role_references: Annotated[list, operator.add]
    course_candidates: Annotated[list, operator.add]
    cert_candidates: Annotated[list, operator.add]
    learning_paths: Annotated[list, operator.add]
    learning_path_steps: Annotated[list, operator.add]
    resource_recommendations: Annotated[list, operator.add]

    # ── Overwrite fields ────────────────────────────────
    progress_step: int
    trace_id: str | None


class TalentForgerOutput(BaseModel):
    """Structured output returned to the application layer."""

    learning_paths: list[LearningPath]
    learning_path_steps: list[LearningPathStep]
    learning_resources: list[LearningResource]
    resource_recommendations: list[ResourceRecommendation]
