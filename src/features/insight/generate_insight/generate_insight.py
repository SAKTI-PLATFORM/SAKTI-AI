"""RAG-B — POST /ml/insight: grounded "Market Ready" narrative for the dashboard.

Stateless: NestJS passes the candidate facts (already computed). We try Gemini
for fluent phrasing, but always fall back to the deterministic grounded template
so numbers stay truthful and the endpoint never fails on a missing LLM key.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.core.http import HTTPDataResponse
from src.domain.insight.insight import (
    GapFact,
    InsightFacts,
    build_grounded_insight,
    build_grounding_prompt,
)
from src.infrastructure.llm.gemini import generate_text

generate_insight_router = APIRouter(prefix="/ml", tags=["insight"])


class GapFactDto(BaseModel):
    skill: str
    gap_hours: int
    priority: str


class InsightRequest(BaseModel):
    full_name: str
    target_role: str
    employability_score: float
    profile_completeness: float = 0.0
    matched_count: int = 0
    top_match_title: str | None = None
    top_match_score: float | None = Field(default=None, ge=0, le=1)
    strengths: list[str] = Field(default_factory=list)
    top_gaps: list[GapFactDto] = Field(default_factory=list)
    holland_code: str | None = None

    def to_facts(self) -> InsightFacts:
        return InsightFacts(
            full_name=self.full_name,
            target_role=self.target_role,
            employability_score=self.employability_score,
            profile_completeness=self.profile_completeness,
            matched_count=self.matched_count,
            top_match_title=self.top_match_title,
            top_match_score=self.top_match_score,
            strengths=self.strengths,
            top_gaps=[GapFact(g.skill, g.gap_hours, g.priority) for g in self.top_gaps],
            holland_code=self.holland_code,
        )


class InsightResultDto(BaseModel):
    narrative: str
    market_ready: bool
    source: str  # "gemini" | "grounded-template"


@generate_insight_router.post(
    "/insight", response_model=HTTPDataResponse[InsightResultDto]
)
async def generate_insight(
    request: InsightRequest,
) -> HTTPDataResponse[InsightResultDto]:
    facts = request.to_facts()

    llm_text = generate_text(build_grounding_prompt(facts))
    narrative = llm_text or build_grounded_insight(facts)
    source = "gemini" if llm_text else "grounded-template"

    return HTTPDataResponse[InsightResultDto](
        message="Insight generated successfully",
        data=InsightResultDto(
            narrative=narrative,
            market_ready=facts.market_ready,
            source=source,
        ),
    )
