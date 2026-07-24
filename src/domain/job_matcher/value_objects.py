"""LangGraph state definition for the JobMatcher graph."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel

from src.domain.job_matcher.schemas import (
    CareerMatchResult,
    ScoreDetail,
    SkillGapResult,
)
from src.domain.shared.user_profile import UserPreferences, UserProfileInput


class JobMatcherState(TypedDict):
    """Shared state flowing through all JobMatcher graph nodes.

    Fields with ``Annotated[list, operator.add]`` use the *append* reducer
    so that every node can return a partial list that gets accumulated.
    """

    # ── Inputs (set once at START) ──────────────────────
    user_profile: UserProfileInput
    preferences: UserPreferences | None

    # ── Accumulated by nodes ────────────────────────────
    candidate_roles: Annotated[list, operator.add]
    similarity_results: Annotated[list, operator.add]
    score_details: Annotated[list, operator.add]
    career_matches: Annotated[list, operator.add]
    skill_gaps: Annotated[list, operator.add]

    # ── Overwrite fields ────────────────────────────────
    market_demand: dict
    progress_step: int
    trace_id: str | None


class JobMatcherOutput(BaseModel):
    """Structured output returned to the application layer."""

    career_match_results: list[CareerMatchResult]
    career_match_score_details: list[ScoreDetail]
    skill_gap_results: list[SkillGapResult]
